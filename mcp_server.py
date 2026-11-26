"""Étape 4 : publication de l'outil RAG en Streamable HTTP."""

import asyncio

from mcp.server.fastmcp import FastMCP

from ingestion import main as reconstruire_index
from rag import repondre_documentation

mcp = FastMCP(
    name="sav-documentation",
    host="127.0.0.1",
    port=24000,
    stateless_http=True,
    json_response=True,
)
verrou_index = asyncio.Lock()


@mcp.tool()
async def indexer_manuels() -> str:
    """Reconstruire l'index des PDF de data/manuels, à la demande de l'utilisateur."""
    if verrou_index.locked():
        raise ValueError(
            "Une recherche ou une indexation est déjà en cours. "
            "Attendez sa fin avant de relancer l'indexation."
        )
    async with verrou_index:
        resultat = await asyncio.to_thread(reconstruire_index)
    return (
        f"Indexation terminée : {resultat['passages']} passages et "
        f"{resultat['figures']} figures décrites et {resultat.get('tableaux', 0)} tableaux. "
        "Vous pouvez poser votre question."
    )


@mcp.tool()
async def rag_multimodal(question: str, photo_reference: str = "") -> str:
    """Dépanner un appareil à partir des manuels et d'une éventuelle photo dans data/photos.

    Transmettre la question complète et la référence exacte de la photo si elle est fournie.
    Renvoie une réponse en français accompagnée des sources (manuel, page, figure).
    """
    async with verrou_index:
        resultat = await repondre_documentation(question, photo_reference)
    references = []
    for source in resultat["sources"]:
        reference = f"- {source['source']}, page {source['page']}"
        if source["type"] == "figure":
            reference += f", figure : {source['image']}"
        references.append(reference)
    if not references:
        return resultat["reponse"]
    return resultat["reponse"] + "\n\nSources consultées :\n" + "\n".join(references)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
