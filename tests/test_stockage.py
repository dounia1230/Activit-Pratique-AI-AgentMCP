"""Essai PGVector réel, activé explicitement avec RUN_PG_TESTS=1."""

import asyncio
import os
from uuid import uuid4

import pytest
from llama_index.core.schema import TextNode
from psycopg import sql

import stockage


@pytest.mark.skipif(
    os.getenv("RUN_PG_TESTS") != "1",
    reason="PostgreSQL réel optionnel : RUN_PG_TESTS=1",
)
def test_pgvector_metadonnees_tableaux_et_publication_atomique(monkeypatch):
    schema = "test_sav_" + uuid4().hex
    monkeypatch.setattr(stockage, "SCHEMA", schema)
    stockage.preparer_stockage()
    index = stockage.IndexPG("essai", 3)
    try:
        node = TextNode(
            text="Code E02 : papier",
            embedding=[1.0, 0.0, 0.0],
            metadata={
                "version": "v1",
                "source": "manuel.pdf",
                "page": 4,
                "type": "tableau",
                "lignes": [["Code", "Action"], ["E02", "Papier"]],
            },
        )
        index.ajouter([node])
        stockage.publier_index("v1", {"passages": 0, "tableaux": 1, "figures": 0})
        assert index.documents()[0].metadata["lignes"][1] == ["E02", "Papier"]
        assert index.proches([1.0, 0.0, 0.0])[0].metadata["page"] == 4
        assert not index.proches([1.0, 0.0, 0.0], sources={"autre.pdf"})
        index.ajouter(
            [
                TextNode(
                    text="Nouvelle version",
                    embedding=[0.0, 1.0, 0.0],
                    metadata={"version": "v2", "source": "nouveau.pdf", "page": 5},
                )
            ]
        )
        assert index.documents()[0].page_content == "Code E02 : papier"
        stockage.publier_index("v2", {"passages": 1, "tableaux": 0, "figures": 0})
        assert index.documents()[0].page_content == "Nouvelle version"
        index.supprimer("v1")
        assert not index.documents(version="v1")
    finally:
        asyncio.run(index.store.close())
        with stockage.connexion() as db:
            db.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
