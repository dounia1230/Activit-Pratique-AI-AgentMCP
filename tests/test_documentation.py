import asyncio

import numpy as np
import pytest
from chromadb.api.types import EmbeddingFunction
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, ToolMessage

import ingestion
import mcp_server
import rag
from examples import creer_demo


@pytest.fixture
def bases(tmp_path, monkeypatch):
    class ClipSimule(EmbeddingFunction):
        def __init__(self):
            self.images_encodees = 0

        def __call__(self, input):
            self.images_encodees += sum(isinstance(item, np.ndarray) for item in input)
            return [np.array([1.0, 0.0, 0.0], dtype=np.float32) for item in input]

        @staticmethod
        def name():
            return "clip_simule"

        def get_config(self):
            return {}

        @staticmethod
        def build_from_config(config):
            return ClipSimule()

    clip = ClipSimule()
    monkeypatch.setattr(ingestion, "DATA", tmp_path)
    monkeypatch.setattr(
        ingestion, "embeddings", lambda: DeterministicFakeEmbedding(size=16)
    )
    monkeypatch.setattr(ingestion, "embeddings_clip", lambda: clip)
    texte = ingestion.base_documentaire()
    images = ingestion.base_figures()
    monkeypatch.setattr(rag, "base_documentaire", lambda: texte)
    monkeypatch.setattr(rag, "base_figures", lambda: images)
    return texte, images, clip


def test_extraction_texte_et_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(creer_demo, "DATA", tmp_path)
    creer_demo.main()
    monkeypatch.setattr(ingestion, "DATA", tmp_path)
    documents = ingestion.extraire_pdf(tmp_path / "manuels" / "manuel_demo_px200.pdf")
    assert {d.metadata["type"] for d in documents} == {"texte", "figure"}
    assert all(d.metadata["page"] == 1 for d in documents)
    assert any(
        "bac a papier est vide" in " ".join(d.page_content.split()) for d in documents
    )
    figure = next(d for d in documents if d.metadata["type"] == "figure")
    assert (tmp_path / figure.metadata["image"]).is_file()


def test_rag_recherche_texte_et_figure_et_sources(bases, monkeypatch):
    # Vraie base Chroma ; embeddings et génération simulés, aucun appel réseau.
    base, images, _ = bases
    base.add_documents(
        [
            Document(
                page_content="E02 : ajouter du papier.",
                metadata={"source": "manuel.pdf", "page": 4, "type": "texte"},
            ),
        ]
    )
    images.add(
        ids=["figure"],
        embeddings=[[1.0, 0.0, 0.0]],
        documents=["Voyant papier rouge."],
        metadatas=[
            {
                "source": "manuel.pdf",
                "page": 4,
                "type": "figure",
                "image": "figures/voyant.png",
            }
        ],
    )
    monkeypatch.setattr(rag, "modele", lambda: None)

    def agent_simule(model, tools, system_prompt):
        class Agent:
            async def ainvoke(self, etat, **kwargs):
                contenu = await tools[0].ainvoke({"requete": "E02 papier"})
                assert "ajouter du papier" in contenu
                assert "figures/voyant.png" in contenu
                return {
                    "messages": [
                        ToolMessage(content=contenu, tool_call_id="test"),
                        AIMessage(content="Ajouter du papier [manuel.pdf, page 4]."),
                    ]
                }

        return Agent()

    monkeypatch.setattr(rag, "create_agent", agent_simule)
    resultat = asyncio.run(rag.repondre_documentation("Que signifie E02 ?"))
    assert {s["type"] for s in resultat["sources"]} == {"texte", "figure"}
    assert all(s["source"] == "manuel.pdf" for s in resultat["sources"])
    assert "page 4" in resultat["reponse"]


def test_base_vide_ne_genere_pas_de_reponse(bases):
    resultat = asyncio.run(rag.repondre_documentation("Question sans source"))
    assert resultat["sources"] == []
    assert "vide" in resultat["reponse"]


def test_ingestion_et_recherche_par_photo_utilisent_les_pixels(
    bases, tmp_path, monkeypatch
):
    textes, images, clip = bases
    monkeypatch.setattr(creer_demo, "DATA", tmp_path)
    creer_demo.main()
    ingestion.main()
    assert textes.get()["ids"]
    assert images.count() == 1
    assert (
        clip.images_encodees == 1
    )  # La légende ne doit pas remplacer l'image à indexer.
    resultat = images.query(
        query_uris=[str(tmp_path / "photos" / "img_px200_01.png")],
        n_results=1,
        include=["metadatas"],
    )
    assert clip.images_encodees == 2
    assert resultat["metadatas"][0][0]["source"] == "manuel_demo_px200.pdf"


def test_mcp_conserve_les_references_sans_llm(monkeypatch):
    async def resultat_rag(question, photo_reference):
        return {
            "reponse": "Ajouter du papier.",
            "sources": [
                {
                    "source": "manuel.pdf",
                    "page": 4,
                    "type": "figure",
                    "image": "figures/voyant.png",
                }
            ],
        }

    monkeypatch.setattr(mcp_server, "repondre_documentation", resultat_rag)
    texte = asyncio.run(mcp_server.rag_multimodal("E02 ?"))
    assert texte.startswith("Ajouter du papier.")
    assert "manuel.pdf, page 4, figure : figures/voyant.png" in texte
