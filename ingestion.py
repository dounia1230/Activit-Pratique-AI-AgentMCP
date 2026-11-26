"""Étape 3 : texte avec Ollama, figures avec CLIP, stockage dans ChromaDB."""

import hashlib
from uuid import uuid4

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
    anciens_textes = base.get()["ids"]
    version = uuid4().hex
    nouveaux_textes, nouvelles_figures = [], []
    images = None
    try:
        # Un long manuel ne doit pas devenir une requête Ollama géante.
        for debut in range(0, len(textes), 16):
            lot = textes[debut : debut + 16]
            ids = [f"{version}_texte_{debut + i}" for i in range(len(lot))]
            nouveaux_textes.extend(ids)
            base.add_documents(lot, ids=ids)
            print(f"Texte : {debut + len(lot)}/{len(textes)}", flush=True)

        # Charger CLIP après le calcul textuel limite le pic de mémoire.
        images = base_figures()
        anciennes_figures = images.get()["ids"]
        for debut in range(0, len(figures), 4):
            lot = figures[debut : debut + 4]
            ids = [f"{version}_figure_{debut + i}" for i in range(len(lot))]
            uris = [str(DATA / doc.metadata["image"]) for doc in lot]
            # Charger seulement les pixels du lot, jamais toutes les figures à la fois.
            vecteurs = embeddings_clip()(input=ImageLoader()(uris))
            nouvelles_figures.extend(ids)
            images.add(
                ids=ids,
                embeddings=vecteurs,
                uris=uris,
                documents=[doc.page_content for doc in lot],
                metadatas=[doc.metadata for doc in lot],
            )
            print(f"Figures : {debut + len(lot)}/{len(figures)}", flush=True)
    except Exception as erreur:
        # Les identifiants uniques permettent de retirer uniquement l'essai raté.
        if nouveaux_textes:
            base.delete(ids=nouveaux_textes)
        if images is not None and nouvelles_figures:
            images.delete(ids=nouvelles_figures)
        raise RuntimeError(
            "Indexation interrompue ; les nouveaux lots ont été retirés et l'index "
            "précédent est conservé. Vérifiez Ollama, puis réessayez. "
            f"Détail : {erreur}"
        ) from erreur

    # Ne retirer l'ancien index qu'après le succès de tous les lots.
    if anciens_textes:
        base.delete(ids=anciens_textes)
    if anciennes_figures:
        images.delete(ids=anciennes_figures)
    print(
        f"{len(textes)} passages et {len(figures)} figures CLIP indexés dans data/chroma."
    )
    return {"passages": len(textes), "figures": len(figures)}


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as erreur:
        raise SystemExit(str(erreur)) from erreur
