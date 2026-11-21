import asyncio

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, ToolMessage

import ingestion
import mcp_server
import rag
from examples import creer_demo


def test_extraction_texte_et_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(creer_demo, "DATA", tmp_path)
    creer_demo.main()
    monkeypatch.setattr(ingestion, "DATA", tmp_path)
    monkeypatch.setattr(
        ingestion,
        "decrire_image",
        lambda *args: "Voyant rouge, point d'exclamation, E02",
    )
    documents = ingestion.extraire_pdf(tmp_path / "manuels" / "manuel_demo_px200.pdf")
    assert {d.metadata["type"] for d in documents} == {"texte", "figure"}
    assert all(d.metadata["page"] == 1 for d in documents)
    assert any(
        "bac a papier est vide" in " ".join(d.page_content.split()) for d in documents
    )
    figure = next(d for d in documents if d.metadata["type"] == "figure")
    assert (tmp_path / figure.metadata["image"]).is_file()


def test_rag_recherche_texte_et_figure_et_sources(tmp_path, monkeypatch):
    # Vraie base Chroma ; embeddings et génération simulés, aucun appel réseau.
    monkeypatch.setattr(ingestion, "DATA", tmp_path)
    monkeypatch.setattr(
        ingestion, "embeddings", lambda: DeterministicFakeEmbedding(size=16)
    )
    base = ingestion.base_documentaire()
    base.add_documents(
        [
            Document(
                page_content="E02 : ajouter du papier.",
                metadata={"source": "manuel.pdf", "page": 4, "type": "texte"},
            ),
            Document(
                page_content="Voyant papier rouge.",
                metadata={
                    "source": "manuel.pdf",
                    "page": 4,
                    "type": "figure",
                    "image": "figures/voyant.png",
                },
            ),
        ]
    )
    monkeypatch.setattr(rag, "base_documentaire", lambda: base)
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


def test_base_vide_ne_genere_pas_de_reponse(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion, "DATA", tmp_path)
    monkeypatch.setattr(
        ingestion, "embeddings", lambda: DeterministicFakeEmbedding(size=16)
    )
    monkeypatch.setattr(rag, "base_documentaire", ingestion.base_documentaire)
    resultat = asyncio.run(rag.repondre_documentation("Question sans source"))
    assert resultat["sources"] == []
    assert "vide" in resultat["reponse"]


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
