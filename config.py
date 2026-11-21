"""Configuration commune : tous les modèles tournent dans Ollama."""

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:4b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3.5:4b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:24000/mcp")


def modele(vision=False):
    return ChatOllama(
        model=VISION_MODEL if vision else CHAT_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
        reasoning=False,
        num_ctx=8192,
        num_predict=1500,
        client_kwargs={"timeout": 180},
    )


def embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)


def decrire_image(image: Path, consigne: str) -> str:
    """Le modèle vision reçoit réellement les pixels de l'image."""
    mime = "image/jpeg" if image.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    if image.suffix.lower() == ".webp":
        mime = "image/webp"
    contenu = base64.b64encode(image.read_bytes()).decode()
    message = HumanMessage(
        content=[
            {"type": "text", "text": consigne},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{contenu}"},
            },
        ]
    )
    return modele(vision=True).invoke([message]).text


def trouver_photo(reference: str) -> Path:
    """Une référence désigne uniquement un fichier du dossier data/photos."""
    for photo in sorted((DATA / "photos").glob("*")):
        if photo.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            if reference in {photo.name, photo.stem}:
                return photo
    raise FileNotFoundError(f"Photo '{reference}' absente de data/photos.")
