"""Étape 3 : texte avec Ollama, figures avec CLIP, stockage dans ChromaDB."""

import hashlib

import chromadb
import pymupdf
from chromadb.utils.data_loaders import ImageLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CLIP_CHECKPOINT,
    CLIP_MODEL,
    DATA,
    EMBEDDING_MODEL,
    embeddings,
    embeddings_clip,
)


def base_documentaire():
    # Une autre configuration de modèles utilise une autre collection.
    version = hashlib.sha256(f"v2:{EMBEDDING_MODEL}".encode()).hexdigest()[:12]
    return Chroma(
        collection_name=f"manuels_{version}",
        persist_directory=str(DATA / "chroma"),
        embedding_function=embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def base_figures():
    # CLIP et les embeddings textuels ont des dimensions différentes.
    version = hashlib.sha256(f"v1:{CLIP_MODEL}:{CLIP_CHECKPOINT}".encode()).hexdigest()[
        :12
    ]
    client = chromadb.PersistentClient(path=str(DATA / "chroma"))
    return client.get_or_create_collection(
        name=f"figures_clip_{version}",
        embedding_function=embeddings_clip(),
        data_loader=ImageLoader(),
        metadata={"hnsw:space": "cosine"},
    )


def extraire_pdf(pdf):
    documents = []
    dossier = DATA / "figures" / pdf.stem
    dossier.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf) as manuel:
        for numero, page in enumerate(manuel, start=1):
            metadata = {"source": pdf.name, "page": numero, "type": "texte"}
            texte = page.get_text().strip()
            if texte:
                documents.append(Document(page_content=texte, metadata=metadata))

            # Les blocs image correspondent aux figures intégrées au PDF.
            for index, bloc in enumerate(page.get_text("dict")["blocks"], start=1):
                if bloc["type"] != 1:
                    continue
                image = dossier / f"page_{numero}_figure_{index}.png"
                pixmap = pymupdf.Pixmap(bloc["image"])
                if pixmap.colorspace is None or pixmap.colorspace.n > 3:
                    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
                pixmap.save(image)
                documents.append(
                    Document(
                        # Le texte de la même page permet d'expliquer la figure retrouvée.
                        page_content=f"Figure du manuel {pdf.name}, page {numero}.\n{texte[:2000]}",
                        metadata={
                            **metadata,
                            "type": "figure",
                            "image": str(image.relative_to(DATA)),
                        },
                    )
                )
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    textes = [doc for doc in documents if doc.metadata["type"] == "texte"]
    figures = [doc for doc in documents if doc.metadata["type"] == "figure"]
    return splitter.split_documents(textes) + figures


def main():
    pdfs = sorted((DATA / "manuels").glob("*.pdf"))
    if not pdfs:
        raise ValueError(
            "Ajoute un manuel PDF dans data/manuels, ou importe-le depuis Streamlit."
        )
    documents = []
    for pdf in pdfs:
        print(f"Extraction : {pdf.name}", flush=True)
        documents.extend(extraire_pdf(pdf))
    if not documents:
        raise ValueError("Aucun texte ni figure extrait : vérifier les PDF.")

    textes = [doc for doc in documents if doc.metadata["type"] == "texte"]
    figures = [doc for doc in documents if doc.metadata["type"] == "figure"]
    base = base_documentaire()
    images = base_figures()
    # Reconstruction explicite : un manuel supprimé ne reste pas dans l'index.
    anciens = base.get()["ids"]
    if anciens:
        base.delete(ids=anciens)
    anciens = images.get()["ids"]
    if anciens:
        images.delete(ids=anciens)
    if textes:
        base.add_documents(textes, ids=[f"texte_{i}" for i in range(len(textes))])
    if figures:
        # Les URI chargent les pixels : CLIP encode les images, pas leur légende.
        uris = [str(DATA / doc.metadata["image"]) for doc in figures]
        vecteurs = embeddings_clip()(input=ImageLoader()(uris))
        images.add(
            ids=[f"figure_{i}" for i in range(len(figures))],
            embeddings=vecteurs,
            uris=uris,
            documents=[doc.page_content for doc in figures],
            metadatas=[doc.metadata for doc in figures],
        )
    print(
        f"{len(textes)} passages et {len(figures)} figures CLIP indexés dans data/chroma."
    )
    return {"passages": len(textes), "figures": len(figures)}


if __name__ == "__main__":
    try:
        main()
    except ValueError as erreur:
        raise SystemExit(str(erreur)) from erreur
