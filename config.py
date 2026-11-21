"""Configuration des modèles locaux et des chemins de données."""

import os
from functools import cache
from pathlib import Path

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:4b")
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_CHECKPOINT = os.getenv("CLIP_CHECKPOINT", "openai")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
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


def embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)


@cache
def embeddings_clip():
    """Charger CLIP une seule fois ; les calculs restent locaux sur CPU."""
    return OpenCLIPEmbeddingFunction(
        model_name=CLIP_MODEL,
        checkpoint=CLIP_CHECKPOINT,
        device="cpu",
    )


def trouver_photo(reference: str) -> Path:
    """Une référence désigne uniquement un fichier du dossier data/photos."""
    for photo in sorted((DATA / "photos").glob("*")):
        if photo.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            if reference in {photo.name, photo.stem}:
                return photo
    raise FileNotFoundError(f"Photo '{reference}' absente de data/photos.")
