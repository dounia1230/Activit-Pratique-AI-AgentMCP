"""Configuration des modèles locaux et des chemins de données."""

import os
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from llama_index.embeddings.ollama import OllamaEmbedding

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:4b")
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_CHECKPOINT = os.getenv("CLIP_CHECKPOINT", "openai")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
VISION_MODEL = os.getenv("VISION_MODEL", CHAT_MODEL)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@127.0.0.1:5432/sav_mcp")
MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:24000/mcp")


def modele():
    return ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
        reasoning=False,
        num_ctx=8192,
        num_predict=1500,
        client_kwargs={"timeout": 180},
    )


@cache
def embeddings():
    # Le petit modèle d'embeddings reste sur CPU, sans démarrage CUDA.
    nomic = EMBEDDING_MODEL.startswith("nomic-embed-text")
    return OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
        embed_batch_size=16,
        text_instruction="search_document: " if nomic else None,
        query_instruction="search_query: " if nomic else None,
        ollama_additional_kwargs={"num_gpu": 0, "num_ctx": 2048},
        client_kwargs={"timeout": 90},
    )


def trouver_photo(reference: str) -> Path:
    """Une référence désigne uniquement un fichier du dossier data/photos."""
    for photo in sorted((DATA / "photos").glob("*")):
        if photo.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            if reference in {photo.name, photo.stem}:
                return photo
    raise FileNotFoundError(f"Photo '{reference}' absente de data/photos.")
