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

Un vrai document pour commencer : le [guide officiel Epson ET-2850 en français](https://files.support.epson.com/docid/cpd6/cpd60207.pdf), avec 279 pages et des images intégrées. Il couvre notamment le chargement du papier, le Wi-Fi, l'entretien et la résolution des pannes.

Le fichier téléchargé est `data/manuels/epson_et2850_guide_fr.pdf`. Pour le récupérer sur une nouvelle installation, puis l'indexer :

```powershell
Invoke-WebRequest -Uri "https://files.support.epson.com/docid/cpd6/cpd60207.pdf" -OutFile "data/manuels/epson_et2850_guide_fr.pdf"
uv run ingestion.py
```

Si le PDF est déjà présent, lancer seulement l'ingestion, ou utiliser **Indexer les PDF** dans Streamlit. Le PDF reste local, hors de Git. Les références de page correspondent aux pages du PDF complet. Le registre de garantie fourni reste un jeu de données pédagogique fictif, indépendant de ce manuel Epson.

ChromaDB conserve deux collections : texte avec les embeddings Ollama, figures avec CLIP. La recherche textuelle combine les mots-clés des pages indexées (BM25 et racines françaises avec Snowball) et la recherche sémantique. Elle conserve les mots de la question d'origine, reconnaît un modèle présent dans le nom du manuel et inclut les pages voisines pour retrouver les étapes et précautions d'une procédure. Les tables des matières riches en pointillés sont écartées du classement lexical. La sélection textuelle est limitée à six pages. Cette recherche utilise l'index existant, sans réindexation nécessaire pour ce changement.

Une photo recherche les figures visuellement proches ; sans photo, les illustrations sont choisies uniquement parmi les pages retrouvées, pour éviter les captures ou schémas sans rapport. La réponse s'appuie sur le texte de ces pages. CLIP ne lit pas les codes d'erreur et ne produit pas de descriptions. Les dessins vectoriels ne sont pas indexés comme figures séparées ; en l'absence de figure extraite disponible, Streamlit affiche jusqu'à deux aperçus des pages citées, qui conservent ces dessins. L'OCR complet des documents scannés n'est pas pris en charge. Relancer l'ingestion après modification des manuels ou des modèles, sans requête en cours : les collections actives sont reconstruites.

L'indexation envoie 16 passages à la fois à Ollama et traite les figures par lots de 4 pour limiter la mémoire utilisée. Le modèle d'embeddings utilise le CPU, un contexte de 2048 tokens et un délai réseau de 90 secondes ; le modèle de chat garde sa configuration habituelle. Un long manuel peut prendre plusieurs minutes ; l'appel d'indexation MCP accepte jusqu'à 30 minutes d'attente. L'ancien index est retiré seulement après le succès de tous les lots. Si le calcul ou l'ajout d'un lot échoue, les nouveaux lots sont supprimés et les entrées de l'ancien index restent disponibles. Cette protection concerne les erreurs gérées par le programme, pas un arrêt brutal du processus.

Si une erreur mentionne `/tokenize` et un port local variable, le processus interne du modèle Ollama est devenu indisponible ; ne pas utiliser ce port dans `.env`. Vérifier avec `ollama list`, puis essayer `ollama stop nomic-embed-text` et relancer l'indexation. Si nécessaire, quitter et relancer l'application Ollama. Après une modification du code, redémarrer également `uv run mcp_server.py`. Les erreurs GPU et les détails sont consultables dans `%LOCALAPPDATA%\Ollama\server.log` sous Windows.

Si l'interface affiche une erreur MCP, les sous-erreurs des groupes `TaskGroup` sont dépliées pour montrer la cause (connexion refusée, délai d'attente, etc.). Les délais de lecture HTTP et de réponse MCP pour l'indexation sont tous deux fixés à 30 minutes. Après une mise à jour, redémarrer aussi Streamlit avec `uv run streamlit run app.py` pour recharger les modules Python. Une déconnexion du navigateur ne garantit pas l'arrêt du calcul côté serveur : ne pas relancer plusieurs fois l'indexation. Le serveur refuse une nouvelle indexation lorsqu'une opération documentaire est déjà en cours.

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
uv run main.py "Comment retirer un bourrage papier sur mon Epson ET-2850 ?"
```

`--photo` désigne le nom d'un fichier PNG, JPEG ou WebP de `data/photos/`, avec ou sans extension. Chaque question est indépendante. Le parcours garantie fonctionne sans serveur MCP ; son registre est `data/garanties.json`.

Questions possibles avec le manuel Epson indexé : « Comment charger le papier dans mon Epson ET-2850 ? », « Comment configurer le Wi-Fi ? » ou « Comment nettoyer la tête d'impression ? ». Les figures retrouvées apparaissent sous la réponse dans Streamlit lorsqu'elles sont disponibles.

Pour tester la garantie sans LLM :

```powershell
uv run garantie.py PX200-0042
```

### Interface Streamlit

Pour utiliser le SAV dans le navigateur, lancer dans le second terminal à la place de `main.py` :

```powershell
uv run streamlit run app.py
```

Ouvrir [http://127.0.0.1:8502](http://127.0.0.1:8502). L'adresse IPv4 et le port sont fixés dans `.streamlit/config.toml` pour distinguer cette interface des autres applications locales sur le port 8501. Ollama doit être démarré ; les questions documentaires nécessitent aussi le serveur MCP du premier terminal et l'ingestion des manuels.

Dans la barre latérale :

1. Choisir un **PDF ou une photo** (PNG, JPEG, WebP, 50 Mo maximum), puis cliquer sur **Importer le fichier**. Les fichiers sont vérifiés et enregistrés localement dans `data/manuels/` ou `data/photos/`, avec une empreinte dans leur nom pour éviter d'écraser un autre document.
2. Pour un PDF, cliquer sur **Indexer les PDF** et attendre la confirmation avant de poser une question. Tous les manuels présents sont réindexés : texte avec Ollama, figures intégrées avec CLIP. Le premier chargement de CLIP peut télécharger ses poids. Les PDF protégés par mot de passe sont refusés.
3. Une photo importée est sélectionnée automatiquement, avec aperçu, pour la prochaine question. Elle sert à rechercher les figures visuellement proches des manuels ; elle n'est pas ajoutée à l'index des manuels. Choisir « Sans photo » pour poser une question uniquement textuelle.
4. Poser la question en précisant l'appareil. La réponse affiche le texte sourcé et jusqu'à deux figures retrouvées, avec leur fichier, page et référence. Si aucune figure extraite n'est disponible, des aperçus des pages citées sont affichés à la place. Une figure similaire ne constitue pas à elle seule un diagnostic. Ces aperçus ne réalisent pas d'OCR.

**Redémarrer le serveur MCP après cette mise à jour**, pour rendre disponible l'outil `indexer_manuels`. L'interface déclenche l'indexation via MCP, sans demander au modèle de la lancer. Le serveur attend la fin d'une recherche avant de reconstruire l'index et attend la fin de cette reconstruction avant de lancer une nouvelle recherche. Ne pas lancer simultanément l'ingestion en ligne de commande ni plusieurs serveurs MCP sur la même base.

L'historique conserve les questions, réponses, figures et parcours des agents. Chaque question est indépendante : l'historique affiché n'est pas envoyé aux agents. Il reste en mémoire dans la session du navigateur ; le bouton « Effacer l'historique » le vide, sans supprimer les documents ni l'index. Les fichiers importés restent locaux et sont exclus de Git.

## Tests

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Les tests utilisent des modèles simulés et une base Chroma temporaire. Les tests de l'interface utilisent `AppTest` de Streamlit avec un graphe simulé. Aucun modèle Ollama ni téléchargement CLIP n'est nécessaire pour les exécuter.

Le bouton d'indexation est aussi testé avec un vrai serveur MCP HTTP local : succès d'un outil simulé et expiration de connexion, avec vérification du message affiché dans Streamlit.

## Références

Support « Résumé Agentic AI et MCP », Pr. Youssfi, et exemples [mcp-demo](https://github.com/mohamedYoussfi/mcp-demo) et [agent-rag-as-tool-mcp-v2](https://github.com/mohamedYoussfi/agent-rag-as-tool-mcp-v2/blob/main/mcp-server.py).
