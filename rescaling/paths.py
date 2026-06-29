"""
paths.py — Chemins centralisés du pipeline de recalage.

Tous les chemins sont ancrés sur l'emplacement de ce fichier (et non sur le
répertoire de travail courant) : les scripts du dossier `rescaling/` se comportent
donc de la même façon quel que soit le CWD et quel que soit le système.
"""
from pathlib import Path

# Dossier rescaling/ et racine du dépôt
RESCALING_DIR = Path(__file__).resolve().parent
REPO_ROOT = RESCALING_DIR.parent

# --- Données sources ---
DATA_DIR = REPO_ROOT / "data"
COEUR_1K_DIR = DATA_DIR / "coeur_1k"               # matrices de dose 3D (~1000 patients)
COEUR_1K_SAMPLE_DIR = DATA_DIR / "coeur_1k_petit"  # échantillon de test (5 patients)

# --- Atlas (patient de référence) ---
ATLAS_ID = "ATLAS"  # identifiant du patient de référence retiré (confidentialité)
ATLAS_RAW = COEUR_1K_DIR / f"dosi_coeur_{ATLAS_ID}.csv"

# --- Dossiers de sortie du recalage ---
ORIGINAL_PROJECTED_DIR = RESCALING_DIR / "original_csv_projected"   # méthode CPD + TPS
ANTS_PROJECTED_DIR     = RESCALING_DIR / "ants_csv_projected"       # méthode ANTsPy
ANTS_TRANSFORMED_DIR   = RESCALING_DIR / "ants_csv_transformed"     # patients transformés (avant projection)
BEST_SCORE_DIR         = RESCALING_DIR / "best_score_csv"           # meilleure des deux méthodes par patient
MASK_PROJECTED_DIR     = RESCALING_DIR / "mask_csv_projected"       # recalage corrigé (piloté par le masque)

# Grille de référence (coordonnées de l'atlas projeté) pour la projection des doses
REF_GRID = ORIGINAL_PROJECTED_DIR / f"{ATLAS_ID}_projected.csv"

# --- Visualisations ---
IMAGES_DIR = RESCALING_DIR / "images"
