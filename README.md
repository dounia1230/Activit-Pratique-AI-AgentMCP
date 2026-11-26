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

Démarrer Ollama si nécessaire avec `ollama serve`. Les modèles et les adresses se règlent dans `.env`. Le modèle de chat doit accepter les outils et les sorties JSON. La vision utilise CLIP (`ViT-B-32`, poids `openai`), téléchargé au premier lancement puis exécuté localement sur CPU.

## Préparer les documents

Placer les manuels PDF dans `data/manuels/` et les photos dans `data/photos/`.

Pour créer un manuel PX-200 et une photo de test :

```powershell
uv run examples/creer_demo.py
uv run ingestion.py
```

Les données de démonstration sont fictives. Le générateur refuse de remplacer des fichiers existants ; s'ils sont déjà présents, lancer seulement l'ingestion.

ChromaDB conserve deux collections : texte avec les embeddings Ollama, figures avec CLIP. Une photo recherche les figures visuellement proches ; sans photo, CLIP utilise la question. La réponse s'appuie sur le texte des pages retrouvées. CLIP ne lit pas les codes d'erreur et ne produit pas de descriptions. Les dessins vectoriels et l'OCR complet des documents scannés ne sont pas pris en charge. Relancer l'ingestion après modification des manuels ou des modèles, sans requête en cours : les collections actives sont reconstruites.

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

### Interface Streamlit

Pour utiliser le SAV dans le navigateur, lancer dans le second terminal à la place de `main.py` :

```powershell
uv run streamlit run app.py
```

Ouvrir [http://localhost:8501](http://localhost:8501). Ollama doit être démarré ; les questions documentaires nécessitent aussi le serveur MCP du premier terminal et l'ingestion des manuels.

Dans la barre latérale :

1. Choisir un **PDF ou une photo** (PNG, JPEG, WebP, 50 Mo maximum), puis cliquer sur **Importer le fichier**. Les fichiers sont vérifiés et enregistrés localement dans `data/manuels/` ou `data/photos/`, avec une empreinte dans leur nom pour éviter d'écraser un autre document.
2. Pour un PDF, cliquer sur **Indexer les PDF** et attendre la confirmation avant de poser une question. Tous les manuels présents sont réindexés : texte avec Ollama, figures intégrées avec CLIP. Le premier chargement de CLIP peut télécharger ses poids. Les PDF protégés par mot de passe sont refusés.
3. Une photo importée est sélectionnée automatiquement, avec aperçu, pour la prochaine question. Elle sert à rechercher les figures visuellement proches des manuels ; elle n'est pas ajoutée à l'index des manuels. Choisir « Sans photo » pour poser une question uniquement textuelle.
4. Poser la question en précisant l'appareil. La réponse affiche le texte sourcé et jusqu'à deux figures retrouvées, lorsqu'elles existent, avec leur fichier, page et référence. Une figure similaire ne constitue pas à elle seule un diagnostic. Les PDF sans figures intégrées produisent une réponse textuelle ; l'OCR et les dessins vectoriels restent hors périmètre.

**Redémarrer le serveur MCP après cette mise à jour**, pour rendre disponible l'outil `indexer_manuels`. L'interface déclenche l'indexation via MCP, sans demander au modèle de la lancer. Le serveur attend la fin d'une recherche avant de reconstruire l'index et attend la fin de cette reconstruction avant de lancer une nouvelle recherche. Ne pas lancer simultanément l'ingestion en ligne de commande ni plusieurs serveurs MCP sur la même base.

L'historique conserve les questions, réponses, figures et parcours des agents. Chaque question est indépendante : l'historique affiché n'est pas envoyé aux agents. Il reste en mémoire dans la session du navigateur ; le bouton « Effacer l'historique » le vide, sans supprimer les documents ni l'index. Les fichiers importés restent locaux et sont exclus de Git.

## Tests

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Les tests utilisent des modèles simulés et une base Chroma temporaire. Les tests de l'interface utilisent `AppTest` de Streamlit avec un graphe simulé. Aucun modèle Ollama ni téléchargement CLIP n'est nécessaire pour les exécuter.

## Références

Support « Résumé Agentic AI et MCP », Pr. Youssfi, et exemples [mcp-demo](https://github.com/mohamedYoussfi/mcp-demo) et [agent-rag-as-tool-mcp-v2](https://github.com/mohamedYoussfi/agent-rag-as-tool-mcp-v2/blob/main/mcp-server.py).
