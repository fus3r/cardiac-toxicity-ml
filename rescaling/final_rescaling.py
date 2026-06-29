import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.spatial import cKDTree

from antspy_rescaling import antspy_rescaling
from original_rescaling import original_rescaling
from fonctions_annexes import load_and_clean_csv

def rescale_all_patients(input_dir, atlas_filename, output_dir=None, pattern="dosi_coeur_*.csv", i_start=0, i_end=None):
    """
    Recale tous les patients d'un dossier sur l'atlas et enregistre les nouveaux fichiers csv.

    Paramètres :
    - input_dir      : dossier contenant les CSV
    - atlas_filename : nom du fichier CSV de l'atlas
    - output_dir     : dossier de sortie pour les CSV recalés
    - pattern        : pattern glob pour filtrer les fichiers
    - i_start & i_end : Si l'on veut appliquer la transformation seulement sur un echantillon des donnees.
                        i_end=None (defaut) traite tous les patients a partir de i_start.
    """
    os.makedirs(output_dir, exist_ok=True)

    df_atlas  = load_and_clean_csv(os.path.join(input_dir, atlas_filename))
    csv_files = sorted(glob(os.path.join(input_dir, pattern)))

    # i_end=None -> jusqu'au bout. (l'ancien defaut -float('inf') rendait
    # range() impossible : min(len, -inf) == -inf, non entier.)
    i_end = len(csv_files) if i_end is None else min(len(csv_files), i_end)

    for k in range(i_start, i_end):
        print(k)
        file_path = csv_files[k]
        patient_id = os.path.basename(file_path).replace(".csv", "")
        
        try:
            df_moving = load_and_clean_csv(file_path)
            output_path = os.path.join(output_dir, f"{patient_id}_transformed.csv")
            antspy_rescaling(df_atlas, df_moving, output_path)

        except Exception as e:
            print(f"  ERREUR : {e}")


# PROJECTION DES DOSES

def project_doses_to_reference(ref_df, df, output_path=None, k=3,
                               max_dist=None, voxel_size=2.0):
    """
    Projette les doses d'un patient mobile sur la grille d'un patient de référence
    par interpolation pondérée des k plus proches voisins (IDW).

    Paramètres
    ----------
    ref_df      : DataFrame avec colonnes x, y, z (patient de référence)
    df          : DataFrame avec colonnes x, y, z et ID2013A (patient mobile)
    output_path : Chemin de sortie du fichier TSV (None = ne rien écrire)
    k           : Nombre de plus proches voisins à utiliser (défaut = 3)
    max_dist    : Si fourni (en mm), les points de référence dont le plus proche
                  voisin transformé est au-delà de max_dist reçoivent une dose 0.
                  Sert de garde-fou : sans cela, un recalage raté (cœur transformé
                  minuscule) est masqué car la grille atlas est tout de même remplie
                  par des voisins très lointains. Défaut None = ancien comportement.
    voxel_size  : résolution native (mm), sert seulement aux statistiques de couverture.

    Retourne
    --------
    qc : dict de contrôle qualité (couverture de la projection).
    """

    reference_points = ref_df[['x', 'y', 'z']].values
    moving_points    = df[['x', 'y', 'z']].values
    doses            = df['ID2013A'].values if 'ID2013A' in df.columns else np.ones(len(df))

    # Construction du KD-Tree sur les points du patient mobile
    tree = cKDTree(moving_points)

    # Pour chaque voxel de référence : distances et indices des k voisins
    distances, indices = tree.query(reference_points, k=k)

    # Cas k=1 : query retourne des scalaires, on uniformise en tableaux 2D
    if k == 1:
        distances = distances[:, np.newaxis]
        indices   = indices[:, np.newaxis]

    # Interpolation par pondération inverse à la distance (IDW), vectorisée.
    # clip(EPSILON) évite la division par zéro : un voisin coïncident reçoit un
    # poids ~1e10 et domine donc le résultat (== prendre sa dose directement).
    EPSILON = 1e-10
    weights      = 1.0 / np.clip(distances, EPSILON, None)        # (N, k)
    neighbor_d   = doses[indices]                                 # (N, k)
    result_doses = (weights * neighbor_d).sum(axis=1) / weights.sum(axis=1)

    # distance au plus proche voisin (1ère colonne) = base du contrôle qualité
    nn_dist = distances[:, 0]

    if max_dist is not None:
        # garde-fou : on ne remplit pas les zones atlas trop loin du cœur transformé
        result_doses = np.where(nn_dist <= max_dist, result_doses, 0.0)

    result_df = pd.DataFrame(reference_points, columns=['x', 'y', 'z'])
    result_df['ID2013A'] = result_doses
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        result_df.to_csv(output_path, index=False, sep='\t')

    return {
        'n_ref'            : int(len(reference_points)),
        'n_moving'         : int(len(moving_points)),
        'nn_dist_median'   : float(np.median(nn_dist)),
        'nn_dist_p95'      : float(np.percentile(nn_dist, 95)),
        'nn_dist_max'      : float(nn_dist.max()),
        'coverage_1vox'    : float((nn_dist <= voxel_size).mean()),
        'coverage_2vox'    : float((nn_dist <= 2 * voxel_size).mean()),
    }


# Exemple d'utilisation (chemins centralisés via paths.py, indépendants du CWD) :
#
#   import paths as P
#   from fonctions_annexes import load_and_clean_csv
#
#   # 1) recalage ANTs de tous les patients -> ants_csv_transformed/
#   rescale_all_patients(P.COEUR_1K_DIR, P.ATLAS_RAW.name, P.ANTS_TRANSFORMED_DIR)
#
#   # 2) projection des doses d'un patient sur la grille de l'atlas
#   ref = load_and_clean_csv(P.REF_GRID)
#   moving = load_and_clean_csv(P.ANTS_TRANSFORMED_DIR / "<patient>_transformed.csv")
#   project_doses_to_reference(ref, moving, P.ANTS_PROJECTED_DIR / "<patient>_projected.csv")