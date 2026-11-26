"""Vérifier les imports et les références d'images sans modèle."""

import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
from types import SimpleNamespace

import pymupdf
import pytest
from PIL import Image

import fichiers


def image_png():
    image = BytesIO()
    Image.new("RGB", (8, 8)).save(image, format="PNG")
    return image.getvalue()


def test_pdf_valide_sans_ecrasement_ni_sortie_du_dossier(tmp_path, monkeypatch):
    monkeypatch.setattr(fichiers, "DATA", tmp_path)
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((40, 40), "Premier manuel")
        contenu = pdf.tobytes()
        chemin = fichiers.enregistrer_fichier("../../manuel.pdf", contenu)
        assert fichiers.enregistrer_fichier("manuel.pdf", contenu) == chemin
        pdf[0].insert_text((40, 80), "Nouvelle version")
        autre = fichiers.enregistrer_fichier("manuel.pdf", pdf.tobytes())
    assert chemin.parent == tmp_path / "manuels"
    assert chemin != autre
    assert chemin.read_bytes() == contenu
    assert len(list(chemin.parent.glob("*.pdf"))) == 2


@pytest.mark.parametrize(
    "nom,contenu",
    [
        ("manuel.pdf", b"invalide"),
        ("photo.png", b"invalide"),
        ("photo.jpg", image_png()),
        ("script.py", b"print(1)"),
        ("vide.pdf", b""),
    ],
)
def test_import_invalide_necrit_rien(tmp_path, monkeypatch, nom, contenu):
    monkeypatch.setattr(fichiers, "DATA", tmp_path)
    with pytest.raises(ValueError):
        fichiers.enregistrer_fichier(nom, contenu)
    assert not list(tmp_path.iterdir())


def test_figures_citees_et_pixels_conserves_sans_chemin_externe(tmp_path, monkeypatch):
    monkeypatch.setattr(fichiers, "DATA", tmp_path)
    dossier = tmp_path / "figures"
    dossier.mkdir()
    (dossier / "voyant.png").write_bytes(image_png())
    (tmp_path / "prive.png").write_bytes(image_png())
    reponse = (
        "Réponse.\n\nSources consultées :\n"
        "- manuel.pdf, page 4, figure : figures/voyant.png\n"
        "- manuel.pdf, page 4, figure : figures/voyant.png\n"
        "- autre.pdf, page 1, figure : figures/../prive.png\n"
        "- autre.pdf, page 1, figure : figures/absente.png\n"
    )
    figures = fichiers.figures_reponse(reponse)
    assert len(figures) == 1
    assert figures[0]["legende"] == "manuel.pdf, page 4 — figure : figures/voyant.png"
    (dossier / "voyant.png").unlink()
    assert figures[0]["contenu"] == image_png()
    assert fichiers.figures_reponse("Aucune source") == []


@pytest.mark.parametrize("erreur", [False, True])
def test_indexation_utilise_mcp_et_remonte_ses_erreurs(monkeypatch, erreur):
    class Session:
        async def call_tool(self, nom, arguments):
            assert nom == "indexer_manuels"
            assert arguments == {}
            return SimpleNamespace(
                isError=erreur,
                content=[SimpleNamespace(type="text", text="Résultat MCP")],
            )

    class Client:
        def __init__(self, configuration):
            assert configuration["documentation"]["transport"] == "streamable_http"

        @asynccontextmanager
        async def session(self, nom):
            assert nom == "documentation"
            yield Session()

    monkeypatch.setattr(fichiers, "MultiServerMCPClient", Client)
    if erreur:
        with pytest.raises(RuntimeError, match="Résultat MCP"):
            asyncio.run(fichiers.indexer_manuels())
    else:
        assert asyncio.run(fichiers.indexer_manuels()) == "Résultat MCP"
