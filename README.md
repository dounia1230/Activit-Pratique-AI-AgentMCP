# SAV — LangGraph et MCP

Activité pratique du cours de Pr. Youssfi : un agent répond aux questions sur les manuels, un autre vérifie la garantie dans un registre JSON en lecture seule. L'orchestrateur choisit un agent, récupère sa réponse puis termine.

## Installation

Python 3.11+, [uv](https://docs.astral.sh/uv/) et [Ollama](https://ollama.com/) sont nécessaires.

```powershell
uv sync
Copy-Item .env.example .env
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
```

Démarrer Ollama si nécessaire avec `ollama serve`. Les modèles et les adresses se règlent dans `.env`. Le modèle de chat doit accepter les outils et les sorties JSON ; le modèle vision doit accepter les images.

## Préparer les documents

Placer les manuels PDF dans `data/manuels/` et les photos dans `data/photos/`.

Pour créer un manuel PX-200 et une photo de test :

```powershell
uv run examples/creer_demo.py
uv run ingestion.py
```

Les données de démonstration sont fictives. Le générateur refuse de remplacer des fichiers existants ; s'ils sont déjà présents, lancer seulement l'ingestion.

ChromaDB indexe le texte et les descriptions des figures intégrées aux PDF. Les dessins vectoriels et l'OCR complet des documents scannés ne sont pas pris en charge. Relancer l'ingestion après modification des manuels ou des modèles, sans requête en cours : la collection active est reconstruite.

## Lancer

Dans un premier terminal, démarrer le serveur MCP sur `http://127.0.0.1:24000/mcp` :

```powershell
uv run mcp_server.py
```

Dans un second terminal :

```powershell
uv run main.py
```

Ou poser directement une question :

```powershell
uv run main.py "Mon imprimante PX200-0042 est-elle sous garantie ?"
uv run main.py "Que signifie le code E02 sur mon imprimante PX-200 ?" --photo img_px200_01
```

`--photo` désigne le nom d'un fichier PNG, JPEG ou WebP de `data/photos/`, avec ou sans extension. Chaque question est indépendante. Le parcours garantie fonctionne sans serveur MCP ; son registre est `data/garanties.json`.

Pour tester la garantie sans LLM :

```powershell
uv run garantie.py PX200-0042
```

## Tests

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Les tests utilisent des modèles simulés et une base Chroma temporaire. Aucun modèle Ollama n'est nécessaire pour les exécuter.

## Références

Support « Résumé Agentic AI et MCP », Pr. Youssfi, et exemples [mcp-demo](https://github.com/mohamedYoussfi/mcp-demo) et [agent-rag-as-tool-mcp-v2](https://github.com/mohamedYoussfi/agent-rag-as-tool-mcp-v2/blob/main/mcp-server.py).
