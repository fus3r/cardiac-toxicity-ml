import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance, pearsonr


def _safe_pearson(a, b):
    """
    Correlation de Pearson robuste.
    pearsonr leve un ConstantInputWarning et renvoie nan si l'un des
    vecteurs est constant (variance nulle) : cas frequent quand une zone
    anatomique a une dose uniforme. On renvoie alors 1.0 si les deux
    vecteurs sont quasi identiques, 0.0 sinon.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(pearsonr(a, b)[0])


###### Metrique Geometrie

def calculate_dice(df_fixed, df_moving, tolerance=2.0):
    """
    Calcule le score de Dice entre deux nuages de points.
    Parametres :
    - df_fixed, df_moving : DataFrames
    - tolerance : tolerance sur l'existance de voisin

    Retourne
    - Approximation de la metrique de Dice (2*(A inter B)/(A+B))
    """

    fixed_points = df_fixed[['x', 'y', 'z']].values.astype(np.float32)
    moving_points = df_moving[['x', 'y', 'z']].values.astype(np.float32)

    tree_fixed = cKDTree(fixed_points)
    tree_moving = cKDTree(moving_points)

    fixed_matches = tree_moving.query_ball_point(fixed_points, tolerance)
    moving_matches = tree_fixed.query_ball_point(moving_points, tolerance)

    intersection = sum(1 for m in fixed_matches if m) + sum(1 for m in moving_matches if m)

    return intersection / (len(fixed_points) + len(moving_points))


def calculate_hausdorff(df_fixed, df_moving):
    """
    Calcule les distances de Hausdorff entre les positions
    des voxels avant et après transformation.

    Paramètres :
    - df_fixed : CSV de l'atlats  (colonnes x, y, z)
    - df_moving : CSV transformé (colonnes x, y, z)

    Retourne :
    - mhd   : Mean Hausdorff Distance (moyenne symétrique)
    - hd95  : 95e percentile
    - hd100 : distance max absolue
    """
    pts_atlas = df_fixed[['x', 'y', 'z']].values
    pts_after  = df_moving[['x', 'y', 'z']].values

    # garde-fou : un recalage qui s'effondre peut produire 0 point ;
    # on renvoie alors des distances infinies plutot qu'un nan silencieux.
    if len(pts_atlas) == 0 or len(pts_after) == 0:
        return {'mhd': float('inf'), 'hd95': float('inf'), 'hd100': float('inf')}

    tree_atlas = cKDTree(pts_atlas)
    tree_after  = cKDTree(pts_after)

    dist_b2a, _ = tree_after.query(pts_atlas)
    dist_a2b, _ = tree_atlas.query(pts_after)

    all_distances = np.concatenate([dist_b2a, dist_a2b])

    return {
        'mhd'  : float((dist_b2a.mean() + dist_a2b.mean()) / 2.0),
        'hd95' : float(np.percentile(all_distances, 95)),
        'hd100': float(all_distances.max()),
    }


def calculate_jacobian(df_before, df_after, k_neighbors=20):
    """
    Compare le patient avant et après transformation
    pour vérifier la cohérence physique de la déformation.

    Paramètres :
    - df_before : CSV original du patient  (colonnes x, y, z)
    - df_after  : CSV transformé du patient (colonnes x, y, z)
    """
    pts_before = df_before[['x', 'y', 'z']].values
    pts_after  = df_after[['x', 'y', 'z']].values

    tree = cKDTree(pts_before)
    k    = min(k_neighbors + 1, len(pts_before))
    _, neigh_idx = tree.query(pts_before, k=k)

    jacobians = []

    for i in range(len(pts_before)):
        neighbors = neigh_idx[i][1:]

        src = pts_before[neighbors] - pts_before[i]
        tgt = pts_after[neighbors]  - pts_after[i]

        if len(src) < 3:
            continue

        try:
            J       = np.linalg.lstsq(src, tgt, rcond=None)[0]
            jac_det = np.linalg.det(J)
            jacobians.append(jac_det)

        except np.linalg.LinAlgError:
            continue

    # jacobians est une liste Python -> on la convertit en array numpy
    # (sinon .mean()/.min()/(< 0).mean() plus bas levaient une AttributeError)
    if len(jacobians) == 0:
        return {
            'jacobian_mean': float('nan'), 'jacobian_min': float('nan'),
            'jacobian_neg_ratio': float('nan'), 'jacobians': [], 'jac_pts': 0.0,
        }
    jacobians = np.asarray(jacobians, dtype=np.float64)

    jacobian = {
        'jacobian_mean'     : float(jacobians.mean()),
        'jacobian_min'      : float(jacobians.min()),
        'jacobian_neg_ratio': float((jacobians < 0).mean()),
        'jacobians'         : jacobians.tolist()
    }

    r = jacobian['jacobian_neg_ratio']
    alpha = 20  # réglable
    penalty = max(0, 1 - alpha * r)
    jac_pts = penalty * 25
    if jacobian['jacobian_min'] < 0:
        jac_pts *= 0.8
    jacobian['jac_pts'] = jac_pts

    return jacobian


def calculate_registration_score(df_fixed, df_after, dice_tolerance=2.0):
    """
    Calcule un score global de qualité du recalage sur 100.

    Paramètres :
    - df_fixed    : CSV du patient de référence
    - df_after    : CSV du patient après recalage
    - dice_tolerance  : rayon en mm pour le Dice (défaut 2mm)
    Les pondérations ont été distribuées sur Dice et MHD.
    """

    hausdorff = calculate_hausdorff(df_fixed, df_after)
    dice = calculate_dice(df_fixed, df_after, tolerance=dice_tolerance)

    breakdown = {}

    # DICE (60 pts)
    # Linéaire entre 0.5 (0 pt) et 1.0 (60 pts)
    # Plancher à 0.5 : en dessous le recalage est clairement raté
    dice_pts = np.clip((dice - 0.5) / 0.5, 0, 1) * 60
    breakdown['dice_pts'] = round(dice_pts, 2)

    # MHD (40 pts)
    # Pénalité linéaire entre 0mm (40 pts) et 10mm (0 pt)
    # Plafond à 10mm : au delà, l'erreur est cliniquement inacceptable
    mhd_pts = np.clip((10.0 - hausdorff['mhd']) / 10.0, 0, 1) * 40
    breakdown['mhd_pts'] = round(mhd_pts, 2)

    # SCORE FINAL
    total = round(dice_pts + mhd_pts, 2)

    breakdown.update({
        'dice'              : round(dice, 4),
        'mhd'               : round(hausdorff['mhd'],   2),
        'hd95'              : round(hausdorff['hd95'],  2),
        'hd100'             : round(hausdorff['hd100'], 2),
        'total'             : total
    })

    return breakdown


def calculate_score(fixed_points, df_after, dice_tolerance=2.0):
    """
    Renvoie uniquement le score sur 100 obtenu dans calculate_registration_score
    """
    return (calculate_registration_score(fixed_points, df_after, dice_tolerance=2.0))['total']


#### Metrique repartition des doses


def dvh_score(d_ref: np.ndarray, d_moving: np.ndarray,
              bins: int = 50, dmax: float = None) -> float:
    """
    Compare les courbes DVH cumulées : DVH(t) = fraction de voxels recevant >= t Gy.
    RMSE (racine de la moyenned es ecarts quadratiques) entre les deux courbes, normalisé → score ∈ [0, 1].

    Paramètres :
    - d_ref    : vecteur 1D des doses du patient ORIGINAL (avant tout recalage)
    - d_moving : vecteur 1D des doses du patient PROJETÉ sur l'atlas
    - bins     : nombre de seuils de dose évalués pour tracer le DVH
    - dmax     : dose maximale commune servant à borner l'axe des doses
    """
    if dmax is None:
        # dmax : dose la plus élevée observée dans les deux patients,
        # sert de borne supérieure commune pour les seuils DVH
        dmax = max(d_ref.max(), d_moving.max())

    # thresholds : liste de 'bins' valeurs de dose régulièrement espacées entre 0 et dmax
    thresholds = np.linspace(0, dmax, bins)
    # 0 si les courbes sont identiques, jusqu'à ~1 si elles sont très différentes
    dvh_ref = np.array([(d_ref    >= t).mean() for t in thresholds])
    dvh_mov = np.array([(d_moving >= t).mean() for t in thresholds])

    rmse = np.sqrt(np.mean((dvh_ref - dvh_mov) ** 2))

    return float(np.clip(1.0 - rmse, 0, 1))


def wasserstein_score(d_ref: np.ndarray, d_moving: np.ndarray,
                      dmax: float = None) -> float:
    """
    Earth Mover Distance normalisée par la plage de dose → score dans [0, 1].
    Sensible aux décalages globaux de la distribution (ex : sous-estimation systématique).

    Paramètres :
    - d_ref    : vecteur 1D des doses du patient ORIGINAL
    - d_moving : vecteur 1D des doses du patient PROJETÉ
    - dmax     : dose maximale pour normaliser la distance
    """
    if dmax is None:
        # dmax : plage totale de dose, sert à rendre la distance sans unité
        dmax = max(d_ref.max(), d_moving.max())

    if dmax <= 0:               # patient sans dose : distributions identiques (toutes nulles)
        return 1.0

    # w : distance de Wasserstein entre les deux distributions de dose (en Gy)
    # intuition : quantité de "travail" pour transformer une distribution en l'autre
    w = wasserstein_distance(d_ref, d_moving)

    # on normalise par dmax pour ramener dans [0,1], puis on inverse pour avoir un score
    return float(np.clip(1.0 - w / dmax, 0, 1))


def percentile_score(d_ref: np.ndarray, d_moving: np.ndarray,
                     percentiles: list = None, dmax: float = None) -> float:
    """
    RMSE normalisé sur les quantiles clés (D5, D10, D25, D50, D75, D90, D95).
    Capture la forme de la distribution sans surpondérer les extrêmes.

    Paramètres :
    - d_ref        : vecteur 1D des doses du patient ORIGINAL
    - d_moving     : vecteur 1D des doses du patient PROJETÉ
    - percentiles  : liste des percentiles à comparer (D5, D25, D50...)
    - dmax         : dose maximale pour normaliser les écarts
    """
    if percentiles is None:
        # percentiles : les 7 quantiles standards en dosimétrie cardiaque
        percentiles = [5, 10, 25, 50, 75, 90, 95]
    if dmax is None:
        dmax = max(d_ref.max(), d_moving.max())

    if dmax <= 0:               # patient sans dose : quantiles identiques (tous nuls)
        return 1.0

    # q_ref : valeurs des quantiles D5, D10... D95 pour le patient original (en Gy)
    # q_mov : idem pour le patient projeté
    q_ref = np.percentile(d_ref,    percentiles)
    q_mov = np.percentile(d_moving, percentiles)

    # rmse : erreur moyenne normalisée par dmax sur les 7 quantiles
    # → pénalise les décalages sur la médiane, les queues de distribution, etc.
    rmse = np.sqrt(np.mean(((q_ref - q_mov) / dmax) ** 2))

    return float(np.clip(1.0 - rmse, 0, 1))


def dose_distribution_score(
    df_ref: pd.DataFrame,       # patient ORIGINAL avant rescaling
    df_projected: pd.DataFrame, # même patient APRÈS projection sur l'atlas
    dose_col: str = "ID2013A",  # nom de la colonne de dose dans les deux DataFrames
    w_dvh: float = 0.50,        # poids du score DVH dans le score final
    w_wasserstein: float = 0.30,# poids du score Wasserstein dans le score final
    w_percentile: float = 0.20, # poids du score percentiles dans le score final
) -> dict:
    """
    Score de similarité de la DISTRIBUTION MARGINALE des doses (0–100).
    La géométrie est entièrement ignorée : seuls les vecteurs de doses sont comparés.

    Composantes :
      DVH Score         (50%) : comparaison des courbes DVH cumulées
      Wasserstein Score (30%) : Earth Mover Distance normalisée
      Percentile Score  (20%) : comparaison des quantiles D5…D95

    Retourne un dict avec 'total' (sur 100) et le détail de chaque composante.
    """
    assert abs(w_dvh + w_wasserstein + w_percentile - 1.0) < 1e-6, \
        "Les pondérations doivent sommer à 1.0"

    # d_ref  : vecteur numpy 1D des doses du patient original (toute géométrie ignorée)
    # d_proj : vecteur numpy 1D des doses du patient projeté  (toute géométrie ignorée)
    d_ref  = df_ref[dose_col].values.astype(np.float64)
    d_proj = df_projected[dose_col].values.astype(np.float64)

    # dmax : dose maximale commune aux deux patients, sert de référence de normalisation
    dmax = max(d_ref.max(), d_proj.max())

    # Cas patient sans dose (toutes les doses sont nulles) : la distribution
    # (vide) est trivialement preservee.
    # On court-circuite pour eviter les divisions par dmax == 0 (ZeroDivisionError/NaN).
    if dmax <= 0:
        return {
            "total": 100.0,
            "dvh_score": 1.0, "wasserstein_score": 1.0, "percentile_score": 1.0,
            "dvh_pts": round(w_dvh * 100, 2),
            "wasserstein_pts": round(w_wasserstein * 100, 2),
            "percentile_pts": round(w_percentile * 100, 2),
            "d_mean_ref": 0.0, "d_mean_proj": 0.0,
            "d_median_ref": 0.0, "d_median_proj": 0.0,
            "zero_dose": True,
        }

    # s_dvh, s_wass, s_pct : scores individuels ∈ [0, 1] pour chaque métrique
    s_dvh  = dvh_score(d_ref, d_proj, dmax=dmax)
    s_wass = wasserstein_score(d_ref, d_proj, dmax=dmax)
    s_pct  = percentile_score(d_ref, d_proj, dmax=dmax)

    # total : combinaison pondérée, ramenée sur 100
    total = (w_dvh * s_dvh + w_wasserstein * s_wass + w_percentile * s_pct) * 100

    return {
        "total"             : round(total, 2),
        "dvh_score"         : round(s_dvh,  4),
        "wasserstein_score" : round(s_wass, 4),
        "percentile_score"  : round(s_pct,  4),
        "dvh_pts"           : round(s_dvh  * w_dvh        * 100, 2), # contribution DVH sur 100
        "wasserstein_pts"   : round(s_wass * w_wasserstein * 100, 2), # contribution Wasserstein sur 100
        "percentile_pts"    : round(s_pct  * w_percentile  * 100, 2), # contribution percentiles sur 100
        "d_mean_ref"        : round(float(d_ref.mean()),  3), # dose moyenne du patient original (Gy)
        "d_mean_proj"       : round(float(d_proj.mean()), 3), # dose moyenne du patient projeté  (Gy)
        "d_median_ref"      : round(float(np.median(d_ref)),  3), # dose médiane du patient original (Gy)
        "d_median_proj"     : round(float(np.median(d_proj)), 3), # dose médiane du patient projeté  (Gy)
    }


# PARTIE 2 — LOCALISATION ANATOMIQUE DES DOSES  (grille normalisée)

def _normalize_coords(df: pd.DataFrame) -> np.ndarray:
    """
    Ramène les coordonnées (x,y,z) dans [0,1]^3 selon le bounding box propre
    de chaque cœur. Cela rend la comparaison indépendante de la taille et de
    la position absolue du cœur dans l'espace.

    Paramètre :
    - df : DataFrame du patient (original OU projeté), avec colonnes x, y, z
    """
    coords = df[['x', 'y', 'z']].values.astype(np.float64)

    mn, mx = coords.min(axis=0), coords.max(axis=0)

    # rng : étendue du cœur sur chaque axe (en mm)
    # le np.where évite une division par zéro si le cœur est plat sur un axe
    rng = np.where(mx - mn > 0, mx - mn, 1.0)

    # retourne les coordonnées normalisées dans [0,1]^3
    # → (0,0,0) = coin postéro-inférieur-gauche du cœur
    # → (1,1,1) = coin antéro-supérieur-droit du cœur
    return (coords - mn) / rng


def _grid_mean_dose(coords_norm: np.ndarray, doses: np.ndarray,
                    n_bins: int) -> tuple:
    """
    Discrétise le cœur normalisé en une grille n×n×n et calcule
    la dose moyenne par cellule.

    Paramètres :
    - coords_norm : coordonnées normalisées dans [0,1]^3  (N, 3)
    - doses       : valeurs de dose associées à chaque voxel
    - n_bins      : nombre de divisions par axe (ex: 5 → grille 5×5×5 = 125 zones)

    Retourne :
    - grid (n,n,n) : dose moyenne par bin (0 si vide)
    - mask (n,n,n) : True si le bin contient au moins un voxel
    """
    # idx : indices entiers (i, j, k) de la cellule de grille pour chaque voxel
    # ex: un voxel en (0.73, 0.12, 0.95) avec n_bins=5 tombe dans la cellule (3, 0, 4)
    idx = np.clip((coords_norm * n_bins).astype(int), 0, n_bins - 1)

    # grid  : accumulateur de la somme des doses par cellule
    # count : nombre de voxels par cellule (pour calculer la moyenne ensuite)
    grid  = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    count = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)

    # accumulation vectorisée : ajoute la dose de chaque voxel dans sa cellule
    np.add.at(grid,  (idx[:, 0], idx[:, 1], idx[:, 2]), doses)
    np.add.at(count, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)

    # mask : booléen indiquant quelles cellules contiennent au moins un voxel
    mask = count > 0

    # on divise par le nombre de voxels pour obtenir la dose MOYENNE par cellule
    grid[mask] /= count[mask]

    return grid, mask


def spatial_dose_score(
    df_ref: pd.DataFrame,            # patient ORIGINAL avant projection
    df_projected: pd.DataFrame,      # même patient APRÈS projection sur l'atlas
    dose_col: str = "ID2013A",       # nom de la colonne de dose
    n_bins: int = 5,                 # résolution de la grille (5 → 125 zones anatomiques)
    high_dose_percentile: float = 80,# seuil pour définir les zones "haute dose" (top 20%)
) -> dict:
    """
    Score de LOCALISATION ANATOMIQUE des doses dans le cœur (0–100).

    Principe : le cœur est découpé en une grille n×n×n après normalisation
    des coordonnées dans [0,1]^3 (indépendant de la géométrie réelle).
    On compare la dose moyenne par zone anatomique entre le patient original
    et le patient projeté.

    Trois sous-scores :
      Corrélation globale  (40%) : corrélation de Pearson sur tous les bins
      RMSE pondéré         (30%) : RMSE entre bins pondéré par la dose de
                                   référence (les zones chaudes comptent plus)
      Score haute dose     (30%) : corrélation + RMSE sur les bins du top
                                   high_dose_percentile% — vérifie que les
                                   zones à forte dose restent bien localisées

    Retourne un dict avec 'total' (sur 100) et le détail de chaque composante.
    """
    # dmax : dose max du patient original, sert de référence de normalisation
    # (on utilise uniquement df_ref pour ne pas être influencé par des artefacts de projection)
    dmax = float(df_ref[dose_col].values.max())

    # Patient sans dose : localisation trivialement preservee, on evite la
    # division par dmax == 0 (ZeroDivisionError) et la correlation indefinie.
    if dmax <= 0:
        return {
            "total": 100.0,
            "score_global_corr": 100.0, "score_weighted_rmse": 100.0,
            "score_high_dose": 100.0, "pearson_r_global": 1.0,
            "high_dose_threshold_Gy": 0.0, "n_bins_common": 0,
            "n_bins_total": int(n_bins ** 3), "zero_dose": True,
        }

    # c_ref, c_proj : coordonnées normalisées dans [0,1]^3 pour chaque cœur
    # chaque cœur est normalisé selon son propre bounding box → comparaison anatomique relative
    c_ref  = _normalize_coords(df_ref)
    c_proj = _normalize_coords(df_projected)

    # d_ref, d_proj : vecteurs de dose associés à chaque voxel
    d_ref  = df_ref[dose_col].values.astype(np.float64)
    d_proj = df_projected[dose_col].values.astype(np.float64)

    # g_ref, g_proj : grilles 5×5×5 de dose moyenne par zone anatomique
    # m_ref, m_proj : masques booléens des cellules non vides
    g_ref,  m_ref  = _grid_mean_dose(c_ref,  d_ref,  n_bins)
    g_proj, m_proj = _grid_mean_dose(c_proj, d_proj, n_bins)

    # common : cellules présentes dans LES DEUX grilles
    # → on ne compare que les zones anatomiques renseignées des deux côtés
    common = m_ref & m_proj

    # gr : vecteur 1D des doses moyennes par zone pour le patient original
    # gp : vecteur 1D des doses moyennes par zone pour le patient projeté
    # (uniquement sur les zones communes)
    gr = g_ref[common]
    gp = g_proj[common]

    # ── A. Corrélation globale ──────────────────────────────────────────────────
    if len(gr) > 2:
        # r_global : corrélation de Pearson entre les 125 zones des deux cœurs
        # → proche de 1 si les zones chaudes/froides sont aux mêmes endroits anatomiques
        r_global = _safe_pearson(gr, gp)
    else:
        r_global = 0.0  # fallback si trop peu de zones communes

    # score_global : conversion de r ∈ [-1,1] vers [0,1]
    score_global = float(np.clip((r_global + 1) / 2, 0, 1))

    # ── B. RMSE pondéré par la dose de référence ────────────────────────────────
    # weights : poids de chaque zone proportionnel à sa dose dans l'original
    # → une erreur dans une zone à 15 Gy compte beaucoup plus que dans une zone à 2 Gy
    weights = gr / gr.sum() if gr.sum() > 0 else np.ones_like(gr) / len(gr)

    # weighted_rmse : RMSE en Gy, pondéré par l'importance dosimétrique de chaque zone
    weighted_rmse = float(np.sqrt(np.sum(weights * (gr - gp) ** 2)))

    # score_wmse : conversion en score ∈ [0,1] normalisé par dmax
    score_wmse = float(np.clip(1.0 - weighted_rmse / dmax, 0, 1))

    # ── C. Score haute dose ─────────────────────────────────────────────────────
    # threshold : seuil de dose (en Gy) en dessous duquel une zone n'est pas "haute dose"
    # ex: si high_dose_percentile=80, threshold ≈ 11.8 Gy → top 20% des zones les plus irradiées
    threshold  = np.percentile(gr, high_dose_percentile)

    # high_mask : sélecteur booléen des zones à haute dose dans le patient original
    high_mask  = gr >= threshold

    if high_mask.sum() > 2:
        # gr_h : doses moyennes des zones haute dose dans le patient original
        # gp_h : doses moyennes des mêmes zones anatomiques dans le patient projeté
        gr_h, gp_h = gr[high_mask], gp[high_mask]

        # r_high : corrélation de Pearson uniquement sur les zones à forte dose
        # → mesure si les zones les plus irradiées sont bien préservées après projection
        r_high     = _safe_pearson(gr_h, gp_h)

        # rmse_high : RMSE normalisé par dmax sur les zones haute dose uniquement
        rmse_high  = float(np.sqrt(np.mean((gr_h - gp_h) ** 2))) / dmax

        # s_high_r, s_high_mse : conversion en scores ∈ [0,1]
        s_high_r   = float(np.clip((r_high + 1) / 2, 0, 1))
        s_high_mse = float(np.clip(1.0 - rmse_high, 0, 1))

        # score_high : moyenne des deux aspects (corrélation + amplitude)
        score_high = 0.5 * s_high_r + 0.5 * s_high_mse
    else:
        # trop peu de zones haute dose pour être significatif → on replie sur le score global
        score_high = score_global

    # total : combinaison pondérée des trois sous-scores, ramenée sur 100
    total = (0.40 * score_global + 0.30 * score_wmse + 0.30 * score_high) * 100

    return {
        "total"                  : round(total, 2),
        "score_global_corr"      : round(score_global * 100, 2), # corrélation globale sur 100
        "score_weighted_rmse"    : round(score_wmse   * 100, 2), # RMSE pondéré sur 100
        "score_high_dose"        : round(score_high   * 100, 2), # localisation haute dose sur 100
        "pearson_r_global"       : round(float(r_global), 4),    # valeur brute de r ∈ [-1, 1]
        "high_dose_threshold_Gy" : round(float(threshold), 3),   # seuil haute dose en Gy
        "n_bins_common"          : int(common.sum()),             # nb de zones anatomiques comparées
        "n_bins_total"           : int(n_bins ** 3),             # nb total de zones (5^3 = 125)
    }


# PARTIE 3 — SCORE COMBINÉ

def full_dose_score(
    df_ref: pd.DataFrame,            # patient ORIGINAL avant projection
    df_projected: pd.DataFrame,      # même patient APRÈS projection sur l'atlas
    dose_col: str = "ID2013A",       # nom de la colonne de dose
    w_distribution: float = 0.50,   # poids de la partie distribution marginale
    w_spatial: float = 0.50,        # poids de la partie localisation anatomique
) -> dict:
    """
    Score global combinant distribution et localisation anatomique (0–100).

    Pondération par défaut : 50% distribution marginale + 50% localisation.
    Ajustable selon les priorités cliniques.

    Retourne un dict avec :
    - 'total'        : score global sur 100
    - 'distribution' : sous-dict de dose_distribution_score
    - 'spatial'      : sous-dict de spatial_dose_score
    """
    assert abs(w_distribution + w_spatial - 1.0) < 1e-6, \
        "w_distribution + w_spatial doit valoir 1.0"

    # dist : résultats complets de la métrique de distribution marginale
    # spat : résultats complets de la métrique de localisation anatomique
    dist = dose_distribution_score(df_ref, df_projected, dose_col=dose_col)
    spat = spatial_dose_score(df_ref, df_projected, dose_col=dose_col)

    # total : moyenne pondérée des deux scores sur 100
    total = w_distribution * dist["total"] + w_spatial * spat["total"]

    return {
        "total"        : round(total, 2),
        "distribution" : dist,  # dict complet dose_distribution_score
        "spatial"      : spat,  # dict complet spatial_dose_score
    }


# PARTIE 4 — QC GEOMETRIQUE & SCORE DE SELECTION COMBINE
#
# Pourquoi cette partie ?
# -----------------------
# full_dose_score ne mesure QUE la fidélité de la dose (distribution + position
# relative) entre le patient brut et sa projection. Il NE mesure PAS si le cœur
# a bien été ramené dans le repère commun de l'atlas. Or, pour une carte de
# risque par deep learning, c'est précisément l'alignement au repère commun qui
# compte : le voxel (i,j,k) doit correspondre à la même région anatomique chez
# tous les patients.
#
# Conséquence : sélectionner la "meilleure" méthode uniquement sur la fidélité
# de dose favorise la méthode qui DÉFORME LE MOINS (une transformation identité
# obtiendrait ~100/100), ce qui va à l'encontre du but du recalage. La
# littérature en radiothérapie recommande d'évaluer le recalage à la fois
# géométriquement (Dice, HD95) ET dosimétriquement (DVH) — les métriques
# dosimétriques seules sont insuffisantes (cf. TG-132).
#
# On fournit donc :
#   - geometric_qc      : contrôle qualité géométrique d'un cœur TRANSFORMÉ
#                         (avant projection) vs atlas — détecte les recalages
#                         effondrés (cœur réduit à quelques centaines de voxels).
#   - selection_score   : score combinant fidélité de dose ET alignement
#                         anatomique, à utiliser pour choisir la méthode.

def geometric_qc(df_atlas, df_transformed, min_voxel_ratio=0.5,
                 dice_tolerance=2.0, mhd_max=10.0, dice_min=0.5):
    """
    Contrôle qualité GÉOMÉTRIQUE d'un cœur transformé (AVANT projection) par
    rapport à l'atlas. À appliquer sur ants_csv_transformed/ (et idéalement sur
    le nuage transformé de la méthode originale, à sauvegarder), PAS sur les
    projections (qui sont par construction déjà sur la grille atlas).

    Un recalage est jugé valide (passed=True) si :
      - le nombre de voxels transformés >= min_voxel_ratio * voxels atlas
        (évite les effondrements type 337 voxels) ;
      - Dice >= dice_min ;
      - MHD <= mhd_max (mm).

    Retourne un dict : dice, mhd, hd95, voxel_ratio, passed.
    """
    n_atlas = max(len(df_atlas), 1)
    voxel_ratio = len(df_transformed) / n_atlas

    if len(df_transformed) == 0:
        return {'dice': 0.0, 'mhd': float('inf'), 'hd95': float('inf'),
                'voxel_ratio': 0.0, 'passed': False}

    reg = calculate_registration_score(df_atlas, df_transformed, dice_tolerance)
    passed = (voxel_ratio >= min_voxel_ratio) and \
             (reg['dice'] >= dice_min) and (reg['mhd'] <= mhd_max)
    return {
        'dice': reg['dice'], 'mhd': reg['mhd'], 'hd95': reg['hd95'],
        'voxel_ratio': round(voxel_ratio, 3), 'passed': bool(passed),
    }


def selection_score(df_ref_raw, df_projected, df_atlas, df_transformed,
                    dose_col="ID2013A", w_dose=0.5, w_geom=0.5):
    """
    Score de SÉLECTION combinant les deux exigences (0–100) :

      - fidélité de dose  (w_dose) : full_dose_score(brut, projeté)
        -> "la dose du patient a-t-elle survécu à la projection ?"
      - alignement        (w_geom) : calculate_score(atlas, transformé)
        -> "le cœur a-t-il bien été ramené dans le repère commun ?"

    df_transformed est le nuage AVANT projection (ex: ants_csv_transformed/).
    Utiliser ce score plutôt que full_dose_score seul pour choisir entre
    méthodes destinées au repère commun du modèle DL.
    """
    dose = full_dose_score(df_ref_raw, df_projected, dose_col=dose_col)['total']
    geom = calculate_score(df_atlas, df_transformed, dice_tolerance=2.0)
    total = w_dose * dose + w_geom * geom
    return {'total': round(total, 2), 'dose_fidelity': round(dose, 2),
            'geom_alignment': round(geom, 2)}

