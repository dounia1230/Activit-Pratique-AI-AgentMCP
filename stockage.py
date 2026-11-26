"""Deux index LlamaIndex/PGVector et publication atomique de leur version active."""

import hashlib
from functools import cache

import psycopg
from langchain_core.documents import Document
from llama_index.core.schema import MetadataMode
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.postgres import PGVectorStore
from psycopg import sql
from sqlalchemy.engine import make_url

from config import (
    CLIP_CHECKPOINT,
    CLIP_MODEL,
    DATABASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    VISION_MODEL,
)

SCHEMA = "sav_mcp"


def cle_index():
    configuration = f"v1:{EMBEDDING_MODEL}:{EMBEDDING_DIM}:{VISION_MODEL}:{CLIP_MODEL}:{CLIP_CHECKPOINT}"
    return hashlib.sha256(configuration.encode()).hexdigest()[:12]


def connexion():
    try:
        return psycopg.connect(DATABASE_URL, connect_timeout=5)
    except psycopg.Error as erreur:
        raise RuntimeError(
            "Connexion PostgreSQL impossible. Vérifiez DATABASE_URL dans .env et le serveur PostgreSQL."
        ) from erreur


def preparer_stockage():
    with connexion() as db:
        if not db.execute(
            "SELECT 1 FROM pg_extension WHERE extname='vector'"
        ).fetchone():
            raise RuntimeError(
                "L'extension pgvector manque. Un administrateur doit exécuter CREATE EXTENSION vector dans cette base."
            )
        db.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA))
        )
        db.execute(
            sql.SQL("""CREATE TABLE IF NOT EXISTS {}.index_actif (
            cle TEXT PRIMARY KEY, version TEXT NOT NULL, compteurs JSONB NOT NULL
        )""").format(sql.Identifier(SCHEMA))
        )


def etat_index():
    preparer_stockage()
    with connexion() as db:
        ligne = db.execute(
            sql.SQL(
                "SELECT version, compteurs FROM {}.index_actif WHERE cle=%s"
            ).format(sql.Identifier(SCHEMA)),
            (cle_index(),),
        ).fetchone()
    return {"version": ligne[0], **ligne[1]} if ligne else None


def publier_index(version, compteurs):
    from psycopg.types.json import Jsonb

    with connexion() as db:
        db.execute(
            sql.SQL("""INSERT INTO {}.index_actif (cle,version,compteurs) VALUES (%s,%s,%s)
            ON CONFLICT (cle) DO UPDATE SET version=EXCLUDED.version, compteurs=EXCLUDED.compteurs""").format(
                sql.Identifier(SCHEMA)
            ),
            (cle_index(), version, Jsonb(compteurs)),
        )


class IndexPG:
    def __init__(self, nom, dimension):
        url = make_url(DATABASE_URL)
        self.store = PGVectorStore.from_params(
            connection_string=url.set(
                drivername="postgresql+psycopg2"
            ).render_as_string(hide_password=False),
            async_connection_string=url.set(
                drivername="postgresql+asyncpg"
            ).render_as_string(hide_password=False),
            table_name=f"{nom}_{cle_index()}",
            schema_name=SCHEMA,
            embed_dim=dimension,
            use_jsonb=True,
        )

    def filtres(self, version=None, criteres=None):
        if version is None:
            etat = etat_index()
            version = etat["version"] if etat else "aucune_version_active"
        return MetadataFilters(
            filters=[
                MetadataFilter(key="version", value=version),
                *[
                    MetadataFilter(key=cle, value=valeur)
                    for cle, valeur in (criteres or {}).items()
                ],
            ]
        )

    def ajouter(self, noeuds):
        return self.store.add(noeuds)

    def supprimer(self, version):
        self.store.delete_nodes(filters=self.filtres(version))

    def documents(self, version=None, criteres=None):
        noeuds = self.store.get_nodes(filters=self.filtres(version, criteres))
        return [
            Document(
                page_content=n.get_content(metadata_mode=MetadataMode.NONE),
                metadata=n.metadata,
            )
            for n in noeuds
        ]

    def proches(self, vecteur, nombre=6, sources=None):
        filtres = self.filtres()
        if sources:
            filtres.filters.append(
                MetadataFilter(key="source", value=sorted(sources), operator="in")
            )
        resultat = self.store.query(
            VectorStoreQuery(
                query_embedding=vecteur, similarity_top_k=nombre, filters=filtres
            )
        )
        return [
            Document(
                page_content=n.get_content(metadata_mode=MetadataMode.NONE),
                metadata=n.metadata,
            )
            for n in resultat.nodes or []
        ]


@cache
def base_documentaire():
    return IndexPG("passages", EMBEDDING_DIM)


@cache
def base_figures():
    return IndexPG("figures", 512)
