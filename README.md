# SAV — LlamaIndex, PGVector, LangChain et MCP

Projet pédagogique SAV : les PDF techniques sont analysés avec LlamaIndex, leurs tableaux sont extraits et leurs images décrites par un modèle de vision local. PostgreSQL/PGVector conserve les embeddings et les métadonnées. Un orchestrateur LangGraph choisit l'agent documentaire ou l'agent de garantie, récupère sa réponse puis termine.

Le parcours reste **orchestrateur → un agent spécialisé → orchestrateur → END**. Le registre `data/garanties.json` est un jeu de données fictif consulté uniquement en lecture.

## Installation

Python 3.11+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/) et PostgreSQL avec [pgvector](https://github.com/pgvector/pgvector) sont nécessaires. Un serveur PostgreSQL existant peut être utilisé avec une base dédiée à ce projet.

```powershell
uv sync
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
```

Si `.env` existe déjà, le conserver et ajouter seulement les nouvelles variables indiquées dans `.env.example`. Ce fichier contient la connexion PostgreSQL et reste exclu de Git. Les modèles s'exécutent localement ; démarrer Ollama avec `ollama serve` si nécessaire.

Avec un compte administrateur PostgreSQL, préparer une base et activer l'extension :

```sql
CREATE ROLE sav_user LOGIN PASSWORD 'choisir_un_mot_de_passe';
CREATE DATABASE sav_mcp OWNER sav_user;
-- Se connecter ensuite à sav_mcp avec le compte administrateur :
CREATE EXTENSION IF NOT EXISTS vector;
```

Configurer `DATABASE_URL` dans `.env` avec le nom de la base, l'utilisateur, le mot de passe et le port réels. Les caractères réservés du mot de passe doivent être encodés dans l'URL. L'application crée son schéma `sav_mcp` et ses tables dans cette base ; elle ne prépare pas un serveur PostgreSQL à elle seule.

Configuration principale :

| Variable | Rôle | Valeur par défaut |
| --- | --- | --- |
| `OLLAMA_URL` | Serveur Ollama local | `http://localhost:11434` |
| `CHAT_MODEL` | Routage et génération de réponses | `qwen3.5:4b` |
| `VISION_MODEL` | Description des pixels des images | Valeur de `CHAT_MODEL` |
| `EMBEDDING_MODEL` | Embeddings textuels open source | `nomic-embed-text` |
| `EMBEDDING_DIM` | Dimension du modèle textuel | `768` |
| `CLIP_MODEL` / `CLIP_CHECKPOINT` | Embeddings visuels locaux | `ViT-B-32` / `openai` |
| `DATABASE_URL` | Connexion à la base PostgreSQL dédiée | Voir `.env.example` |
| `MCP_URL` | Serveur documentaire Streamable HTTP | `http://127.0.0.1:24000/mcp` |

Le modèle de chat doit accepter les outils et les sorties JSON. Le modèle de vision doit accepter les images : Qwen3.5 convient pour les deux usages. CLIP est chargé une fois par processus sur CPU ; ses poids sont téléchargés au premier usage si absents.

## Documents, tableaux et images

Placer les manuels dans `data/manuels/` et les photos dans `data/photos/`, ou les importer dans Streamlit. Un document de départ est le [manuel officiel Epson ET-2850 en français](https://files.support.epson.com/docid/cpd6/cpd60207.pdf), comprenant 279 pages, du texte, des illustrations et des tableaux.

```powershell
Invoke-WebRequest -Uri "https://files.support.epson.com/docid/cpd6/cpd60207.pdf" -OutFile "data/manuels/epson_et2850_guide_fr.pdf"
uv run ingestion.py
```

Si le PDF est déjà présent, lancer seulement l'ingestion. Les PDF, photos, figures extraites et descriptions restent locaux et exclus de Git.

L'ingestion fonctionne ainsi :

1. Le lecteur PDF et le découpage LlamaIndex produisent les passages avec leur fichier et leur page. PyMuPDF complète le lecteur pour extraire les images intégrées, convertir les zones de dessins vectoriels en figures et détecter les tableaux par leurs bordures. Les bordures de tableaux et les petites décorations sont écartées de cette recherche de schémas.
2. Les tableaux sont conservés en lignes/colonnes dans les métadonnées JSONB. Une représentation Markdown est indexée pour répondre aux questions sur leurs valeurs. Les grands tableaux sont répartis en groupes de quinze lignes, avec les en-têtes répétés.
3. Qwen vision décrit réellement les pixels de chaque image. Ces descriptions sont indexées dans l'espace textuel pour rendre les figures interrogeables par mots-clés ou similarité sémantique. Le texte de la page n'est pas utilisé à la place d'une description visuelle.
4. CLIP calcule des embeddings dans un espace visuel séparé de 512 dimensions, utilisé pour retrouver les figures similaires à une photo. Les embeddings textuels et CLIP sont enregistrés dans deux tables PGVector distinctes, avec leurs métadonnées.

Le texte utilise les préfixes `search_document:` et `search_query:` lorsque le modèle est Nomic. Les embeddings textuels sont calculés par lots de seize sur CPU ; CLIP traite les figures par lots de quatre. Les descriptions sont mises en cache dans `data/descriptions/`, selon les pixels et le modèle de vision, pour éviter de les recalculer après une interruption. Une première indexation avec beaucoup d'images peut être longue ; suivre la progression dans le terminal MCP.

Chaque ingestion reçoit une version. La nouvelle version n'est publiée qu'après le succès du texte, des tableaux, des descriptions et des vecteurs CLIP. PostgreSQL change le pointeur de version active dans une transaction unique. En cas d'échec, l'ancienne version reste consultable ; les descriptions terminées restent en cache. Les anciennes lignes sont retirées après publication. Un arrêt brutal peut laisser des lignes non actives ; elles ne sont pas utilisées pour répondre.

PGVector est l'unique stockage vectoriel du projet. Les embeddings textuels sont produits par `nomic-embed-text` et les embeddings visuels par CLIP. Changer les modèles ou leur dimension nécessite une nouvelle ingestion.

## Recherche et réponses

La recherche combine BM25 avec racines françaises et similarité vectorielle. Elle conserve les mots de la question initiale, reconnaît les modèles dans les noms de manuels, écarte les sommaires riches en pointillés et inclut les pages voisines d'une procédure. Le contexte textuel est limité à six pages et inclut le texte, les tableaux et les descriptions d'images indexées.

Si le modèle répond sans appeler la recherche, par exemple à « c'est quoi le PDF ? », le programme exécute cet outil et redonne ses résultats à l'agent avant de répondre. L'absence d'appel d'outil n'est donc pas présentée comme une panne documentaire.

Sans photo, les illustrations proviennent des pages retrouvées. Avec une photo, CLIP recherche les figures visuellement proches. Les descriptions générées restent des interprétations susceptibles d'erreurs : elles ne remplacent pas les consignes écrites du manuel et une ressemblance ne suffit pas à établir un diagnostic. Les extraits et les images sont des données, jamais des instructions à exécuter.

L'OCR complet des PDF scannés et la reconstruction des tableaux sans bordures ne sont pas garantis. La détection des tableaux est géométrique, sans modèle de mise en page supplémentaire. Un schéma vectoriel peut être réparti en plusieurs figures ; une page contenant beaucoup de tracés devient un aperçu unique pour borner le coût du traitement. Streamlit peut aussi afficher un aperçu des pages citées si aucune figure locale n'est disponible.

## Lancer

Premier terminal, serveur MCP :

```powershell
uv run mcp_server.py
```

Second terminal, interface Streamlit :

```powershell
uv run streamlit run app.py
```

Ouvrir [http://127.0.0.1:8502](http://127.0.0.1:8502). Le port et l'adresse IPv4 sont fixés dans `.streamlit/config.toml` pour distinguer ce projet des autres applications locales. Après une mise à jour, redémarrer MCP et Streamlit pour recharger les modules et la configuration.

Dans la barre latérale : importer un PDF ou une photo (PDF, PNG, JPEG, WebP, 50 Mo maximum), cliquer sur **Importer le fichier**, puis **Indexer les PDF** pour un manuel. Attendre la confirmation avant de poser une question. Une photo importée est sélectionnée pour la prochaine question ; elle n'est pas ajoutée à l'index des manuels.

Exemples de questions :

- « Comment retirer un bourrage papier sur mon Epson ET-2850 ? Cite les pages utilisées. »
- « Quelle est la capacité de chargement du papier selon le tableau du manuel ? »
- « Retrouve les figures montrant les niveaux d'encre. »

Chaque question est indépendante : l'historique affiché n'est pas envoyé aux agents. Le bouton **Effacer l'historique** vide la session, sans supprimer les documents ni PostgreSQL. Les réponses conservent les références fichier/page/figure et affichent jusqu'à deux illustrations ou aperçus de pages. Les images des pages explicitement citées dans la réponse sont prioritaires sur les autres pages consultées. Les petites icônes sont écartées ; sans figure exploitable, un aperçu de la page citée est affiché. L'affichage conserve la taille native des images, avec une largeur maximale de 680 pixels, sans agrandir les petites captures.

Le terminal reste disponible à la place de Streamlit :

```powershell
uv run main.py
uv run main.py "Comment retirer un bourrage papier sur mon Epson ET-2850 ?"
uv run main.py "Que montre le voyant de mon appareil ?" --photo nom_de_la_photo
uv run garantie.py PX200-0042
```

`--photo` désigne un fichier de `data/photos/`, avec ou sans extension. La vérification du registre avec `garantie.py` ne nécessite aucun LLM ; le parcours agentique de garantie utilise Ollama mais pas le serveur MCP ni PostgreSQL.

## Dépannage et vérification

PostgreSQL et Ollama doivent rester démarrés. Si la connexion à PostgreSQL échoue, vérifier `DATABASE_URL` et l'extension `vector` dans la bonne base. Ne pas versionner `.env`.

Une erreur `/tokenize` avec un port local variable concerne le processus interne Ollama. Vérifier `ollama list`, essayer `ollama stop nomic-embed-text` puis relancer l'ingestion ; quitter et relancer Ollama si nécessaire. Les journaux Windows se trouvent dans `%LOCALAPPDATA%\Ollama\server.log`.

Les sous-erreurs `TaskGroup` sont affichées explicitement dans Streamlit. L'appel d'indexation attend jusqu'à trente minutes. Une indexation attend la fin d'une recherche documentaire en cours ; plusieurs clics pendant l'indexation attendent le résultat d'un seul calcul. Une déconnexion du navigateur ne libère pas le verrou et n'arrête pas ce calcul. En cas d'échec, un clic suivant peut réessayer. Si un PDF est ajouté pendant l'indexation, relancer celle-ci une fois terminée pour inclure le nouveau document. Ne pas lancer simultanément plusieurs ingestions en ligne de commande ou plusieurs serveurs MCP sur la même base.

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Les tests courants utilisent des modèles et un index en mémoire simulés, avec de vrais petits PDF pour vérifier l'extraction du texte, des images et des tableaux. Aucun appel Ollama ni téléchargement CLIP n'est nécessaire. Le parcours Streamlit/MCP est également testé en HTTP local avec un outil simulé.

Pour tester aussi la persistance PGVector réelle, utiliser une base PostgreSQL configurée dans `.env` :

```powershell
$env:RUN_PG_TESTS = "1"
uv run pytest -q tests/test_stockage.py
```

Cet essai crée puis supprime uniquement un schéma de test au nom unique. Il vérifie les métadonnées de tableaux, la recherche vectorielle et la publication de version, sans modèle réel. Les essais manuels avec Ollama/CLIP doivent être signalés séparément des tests simulés.

## Références

Support « Résumé Agentic AI et MCP », Pr. Youssfi ; [lecteurs LlamaIndex](https://developers.llamaindex.ai/python/framework/module_guides/loading/simpledirectoryreader/), [PGVectorStore](https://developers.llamaindex.ai/python/framework/integrations/vector_stores/postgres/) et [pgvector](https://github.com/pgvector/pgvector).
