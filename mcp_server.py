"""Étape 4 : publication de l'outil RAG en Streamable HTTP."""

from mcp.server.fastmcp import FastMCP

from rag import repondre_documentation

mcp = FastMCP(
    name="sav-documentation",
    host="127.0.0.1",
    port=24000,
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def rag_multimodal(question: str, photo_reference: str = "") -> str:
    """Dépanner un appareil à partir des manuels et d'une éventuelle photo dans data/photos.

    Transmettre la question complète et la référence exacte de la photo si elle est fournie.
    Renvoie une réponse en français accompagnée des sources (manuel, page, figure).
    """
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
