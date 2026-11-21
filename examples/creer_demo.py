"""Crée un manuel et une photo FICTIFS ; aucune connexion à un LLM."""

from pathlib import Path

import pymupdf

DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    (DATA / "manuels").mkdir(parents=True, exist_ok=True)
    (DATA / "photos").mkdir(parents=True, exist_ok=True)
    photo = DATA / "photos" / "img_px200_01.png"
    manuel = DATA / "manuels" / "manuel_demo_px200.pdf"
    if photo.exists() or manuel.exists():
        raise SystemExit("Exemples déjà présents : aucun fichier remplacé.")
    # Dessiner un pictogramme, puis l'intégrer comme image dans le PDF.
    with pymupdf.open() as dessin:
        page = dessin.new_page(width=260, height=160)
        page.insert_text((20, 25), "PX-200 - E02", fontsize=18)
        page.draw_circle((130, 85), 32, color=(1, 0, 0), width=5)
        page.insert_text((123, 99), "!", fontsize=40, color=(1, 0, 0))
        page.insert_text((65, 145), "Voyant papier", fontsize=16)
        page.get_pixmap().save(photo)

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(40, 40, 555, 220),
            "MANUEL FICTIF - EXERCICE UNIQUEMENT\nImprimante PX-200\n\n"
            "Code E02 : le voyant papier rouge clignotant avec un point d'exclamation "
            "indique que le bac a papier est vide. Ajouter du papier dans le bac, "
            "puis appuyer sur Reprendre.\n"
            "Si le voyant reste allume apres le rechargement, contacter le SAV.\n"
            "Figure 1 : voyant papier et code E02.",
            fontsize=12,
        )
        page.insert_image(pymupdf.Rect(100, 240, 490, 480), filename=str(photo))
        document.save(manuel)
    print(f"Créés : {manuel.name} et {photo.name} (données fictives).")


if __name__ == "__main__":
    main()
