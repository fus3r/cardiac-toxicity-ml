import os
import numpy as np
import pandas as pd
import ants
from scipy.spatial import cKDTree
from metrique_rescaling import calculate_score

def estimate_voxel_size(df, sample_size=1000):
    """
    Estime la résolution native des données en cherchant
    la distance médiane entre points voisins.
    Sous-échantillonne pour rester rapide sur 30k points.
    """
    coords = df[['x', 'y', 'z']].values

    if len(coords) > sample_size:
        idx    = np.random.choice(len(coords), sample_size, replace=False)
        sample = coords[idx]
    else:
        sample = coords

    tree = cKDTree(sample)
    dist, _ = tree.query(sample, k=2)
    min_dist = dist[:, 1]

    return float(np.median(min_dist))

def points_to_ants_image(df, reference_origin, reference_max, voxel_size=2.0):

    mins = np.array(reference_origin, dtype=np.float32)
    maxs = np.array(reference_max,    dtype=np.float32)

    coords = df[['x','y','z']].values.astype(np.float32)
    values = df['ID2013A'].values.astype(np.float32)

    shape = (np.round((maxs - mins) / voxel_size).astype(int) + 1).tolist()

    indices = np.round((coords - mins) / voxel_size).astype(int)
    indices = np.clip(indices, 0, np.array(shape) - 1)

    volume = np.zeros(shape, dtype=np.float32)
    mask   = np.zeros(shape, dtype=np.float32)      # ← masque binaire

    volume[indices[:,0], indices[:,1], indices[:,2]] = values
    mask[indices[:,0],   indices[:,1], indices[:,2]] = 1.0   # ← 1 = voxel appartient au cœur

    img_dose = ants.from_numpy(volume, origin=list(mins), spacing=[voxel_size]*3)
    img_mask = ants.from_numpy(mask,   origin=list(mins), spacing=[voxel_size]*3)

    return img_dose, img_mask

def ants_image_to_points(img_dose, img_mask_warped):
    arr      = img_dose.numpy()
    mask_arr = img_mask_warped.numpy()

    # seuil à 0.5 : voxel appartient au cœur si le masque interpolé > 0.5
    valid = mask_arr > 0.5

    coords = np.argwhere(valid)
    values = arr[valid]

    origin  = np.array(img_dose.origin)
    spacing = np.array(img_dose.spacing)

    phys_coords = coords * spacing + origin

    return pd.DataFrame({"x": phys_coords[:, 0], "y": phys_coords[:, 1], "z": phys_coords[:, 2], "ID2013A": values})

def antspy_rescaling(df_atlas, df_moving, output_csv_path=None,
                     register_on="mask", mask_smoothing=2.0):
    """
    Recale un patient (moving) sur l'atlas (fixed) avec ANTsPy.
    Pipeline : Rigid → Affine → SyN (SyNRA).

    Paramètres :
    - df_atlas        : DataFrame du patient de référence
    - df_moving       : DataFrame du patient à recaler
    - output_csv_path : chemin de sauvegarde du CSV résultat
    - register_on     : "mask" (defaut) ou "dose".
        * "mask" : le recalage est piloté par la FORME du cœur (masque binaire
                   lissé). C'est un recalage ANATOMIQUE : la dose est ensuite
                   transportée passivement par la transformation trouvée.
        * "dose" : ancien comportement, le recalage est piloté par la carte de
                   dose elle-même. Déconseillé (cf. plus bas).
    - mask_smoothing  : sigma (mm) du lissage gaussien appliqué au masque binaire
                        avant recalage (ne sert qu'en mode "mask").

    Pourquoi recaler sur le masque et non sur la dose ?
    ----------------------------------------------------
    Recaler sur la dose revient à déformer le cœur pour faire coïncider les
    *points chauds* de dose, alors que la dose est précisément la grandeur que
    l'on veut comparer entre patients APRÈS les avoir mis dans un repère commun.
    Sur cette cohorte, le recalage piloté par la dose :
      - écrase l'anatomie là où la dose est concentrée (avant du cœur) ;
      - produit des recalages aberrants (cœur transformé réduit à quelques
        centaines de voxels) lorsque la distribution de dose est atypique ;
      - échoue complètement pour les patients à dose nulle (ITK :
        "Total Mass of the image was zero"), d'où les 2 patients à dose nulle
        manquants dans ants_csv_transformed.
    Recaler sur le masque (forme) corrige ces trois problèmes
    (validation : voir _exp_registration.py).

    Retourne :
    - df_result : DataFrame avec colonnes x, y, z, ID2013A recalées
    - Enregistre df_result sous format csv si output_csv_path est specifie
    """

    # 1. Voxelisation
    origin_atlas = df_atlas[['x','y','z']].min().values
    origin_moving = df_moving[['x','y','z']].min().values
    max_atlas = df_atlas[['x','y','z']].max().values
    max_moving = df_moving[['x','y','z']].max().values

    reference_origin = [min(a, b) for a, b in zip(origin_atlas, origin_moving)]
    reference_max = [max(a, b) for a, b in zip(max_atlas, max_moving)]

    img_atlas, img_atlas_mask  = points_to_ants_image(df_atlas, reference_origin, reference_max)
    img_moving, img_moving_mask = points_to_ants_image(df_moving, reference_origin, reference_max)

    # 2. Choix des images qui PILOTENT le recalage
    if register_on == "mask":
        # Masque binaire lissé : donne un gradient exploitable par la métrique CC
        # tout en restant guidé par la forme du cœur (et non la dose).
        fixed_reg  = ants.smooth_image(img_atlas_mask,  mask_smoothing)
        moving_reg = ants.smooth_image(img_moving_mask, mask_smoothing)
    elif register_on == "dose":
        fixed_reg, moving_reg = img_atlas, img_moving
    else:
        raise ValueError("register_on doit valoir 'mask' ou 'dose'")

    # 3. Recherche transformation Rigid + Affine + SyN
    reg = ants.registration(
        fixed=fixed_reg,
        moving=moving_reg,
        type_of_transform="SyNRA",  # Rigid + Affine + SyN interne
        random_seed = 42,
        reg_iterations=(100, 70, 50, 20),
        syn_metric="CC",             # Cross-Correlation : meilleure pour morphologie
        syn_sampling=4,              # rayon CC (pas des bins comme Mattes)
        flow_sigma=3,
        grad_step=0.1,
    )

    # 4. Application de la transformation aux images de SORTIE
    #    - dose : interpolation linéaire (grandeur continue)
    #    - masque : interpolation 'genericLabel' (préserve un masque binaire,
    #      évite l'érosion/dilatation d'une interpolation linéaire seuillée)
    warped_dose = ants.apply_transforms(
        fixed=img_atlas, moving=img_moving,
        transformlist=reg["fwdtransforms"], interpolator="linear")
    warped_mask = ants.apply_transforms(
        fixed=img_atlas_mask, moving=img_moving_mask,
        transformlist=reg["fwdtransforms"], interpolator="genericLabel")
    df_result = ants_image_to_points(warped_dose, warped_mask)

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_result.to_csv(output_csv_path, index=False, sep='\t')

    return df_result
