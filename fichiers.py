"""Importer les documents et retrouver uniquement les figures locales citées."""

import hashlib
import re
from io import BytesIO
from pathlib import Path

import pymupdf
from langchain_mcp_adapters.client import MultiServerMCPClient
from PIL import Image

from config import DATA, MCP_URL


def enregistrer_fichier(nom, contenu):
    # Le nom reçu du navigateur ne doit jamais choisir un chemin sur le disque.
    nom = nom.replace("\\", "/").rsplit("/", 1)[-1]
    extension = Path(nom).suffix.lower()
    if not contenu or len(contenu) > 50 * 1024 * 1024:
        raise ValueError(
            "Le fichier doit contenir des données et ne pas dépasser 50 Mo."
        )
    try:
        if extension == ".pdf":
            with pymupdf.open(stream=contenu, filetype="pdf") as pdf:
                if pdf.needs_pass or not len(pdf):
                    raise ValueError("PDF protégé ou sans page.")
            dossier = "manuels"
        elif extension in {".png", ".jpg", ".jpeg", ".webp"}:
            with Image.open(BytesIO(contenu)) as image:
                attendu = {
                    ".png": "PNG",
                    ".jpg": "JPEG",
                    ".jpeg": "JPEG",
                    ".webp": "WEBP",
                }
                if image.format != attendu[extension]:
                    raise ValueError(
                        "Le contenu ne correspond pas à l'extension de l'image."
                    )
                image.verify()
            dossier = "photos"
        else:
            raise ValueError("Formats acceptés : PDF, PNG, JPEG et WebP.")
    except Exception as erreur:
        raise ValueError(f"Fichier invalide : {erreur}") from erreur
    radical = re.sub(r"[^\w-]", "_", Path(nom).stem)[:80] or "document"
    empreinte = hashlib.sha256(contenu).hexdigest()[:12]
    chemin = DATA / dossier / f"{radical}_{empreinte}{extension}"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    # Le contenu détermine le nom : deux documents différents ne s'écrasent pas.
    try:
        with chemin.open("xb") as fichier:
            fichier.write(contenu)
    except FileExistsError:
        if chemin.read_bytes() != contenu:
            raise ValueError("Un fichier différent porte déjà ce nom.") from None
    return chemin


async def indexer_manuels():
    # L'index reste géré par le même processus que les recherches documentaires.
    client = MultiServerMCPClient(
        {
            "documentation": {
                "transport": "streamable_http",
                "url": MCP_URL,
                "sse_read_timeout": 1800,
            }
        }
    )
    async with client.session("documentation") as session:
        resultat = await session.call_tool("indexer_manuels", {})
    texte = "\n".join(bloc.text for bloc in resultat.content if bloc.type == "text")
    if resultat.isError:
        raise RuntimeError(texte)
    return texte


def figures_reponse(reponse):
    # Ce pied de réponse est ajouté par MCP, après la génération du texte.
    _, separateur, references = reponse.rpartition("\n\nSources consultées :\n")
    if not separateur:
        return []
    dossier = (DATA / "figures").resolve()
    figures, deja_vues = [], set()
    for source, page, reference in re.findall(
        r"^- (.+), page (\d+), figure : (.+)$", references, flags=re.MULTILINE
    ):
        chemin = (DATA / reference.strip()).resolve()
        if (
            not chemin.is_relative_to(dossier)
            or not chemin.is_file()
            or chemin.suffix.lower() != ".png"
            or chemin in deja_vues
        ):
            continue
        deja_vues.add(chemin)
        figures.append(
            {
                # Garder les pixels dans l'historique, même après une réindexation.
                "contenu": chemin.read_bytes(),
                "legende": f"{source}, page {page} — figure : {reference}",
            }
        )
        if len(figures) == 2:
            break
    return figures
