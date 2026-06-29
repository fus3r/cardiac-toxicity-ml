import os
import numpy as np
import pandas as pd
from skopt.space import Real
from skopt import gp_minimize
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull
from pycpd import AffineRegistration, DeformableRegistration

from metrique_rescaling import calculate_score

# 1. FONCTIONS DE GÉOMÉTRIE ET MÉTRIQUES

def extract_boundary_points(points):
    """Extrait les points de l'enveloppe convexe pour le recalage."""
    points = np.array(points)
    hull = ConvexHull(points)
    return points[hull.vertices], hull.vertices

# 2. RECALAGE (CPD + TPS)

def optimize_cpd(fixed_boundary, TY_affine):
    """Optimise les paramètres beta et lambda du recalage déformable."""
    def objective(params):
        beta, alpha = params
        deform = DeformableRegistration(X=fixed_boundary, Y=TY_affine, beta=beta, lambda_=alpha * (beta**2))
        TY_deform, _ = deform.register()

        df_fixed  = pd.DataFrame(fixed_boundary, columns=['x', 'y', 'z'])
        df_deform = pd.DataFrame(TY_deform, columns=['x', 'y', 'z'])

        score = calculate_score(df_fixed, df_deform, dice_tolerance=2.0)
        return -score
    
    diameter = np.linalg.norm(fixed_boundary.max(axis=0) - fixed_boundary.min(axis=0))
    search_space = [Real(0.03 * diameter, 0.15 * diameter, name='beta'), Real(1e-2, 1e1, prior='log-uniform', name='alpha')]
    result = gp_minimize(objective, search_space, n_calls=30, random_state=42, verbose=False)
    
    deform = DeformableRegistration(X=fixed_boundary, Y=TY_affine, beta=result.x[0], lambda_=result.x[1] * (result.x[0]**2))
    TY_deform, _ = deform.register()
    return TY_deform

def apply_tps(boundary_src, boundary_tgt, full_src):
    """Applique la transformation Thin Plate Spline au volume complet."""
    rbf_x = Rbf(boundary_src[:,0], boundary_src[:,1], boundary_src[:,2], boundary_tgt[:,0], function='thin_plate')
    rbf_y = Rbf(boundary_src[:,0], boundary_src[:,1], boundary_src[:,2], boundary_tgt[:,1], function='thin_plate')
    rbf_z = Rbf(boundary_src[:,0], boundary_src[:,1], boundary_src[:,2], boundary_tgt[:,2], function='thin_plate')
    return np.stack([rbf_x(*full_src.T), rbf_y(*full_src.T), rbf_z(*full_src.T)], axis=1)

# FONCTION FINALE DE RESCALING

def original_rescaling(fixed_df, df_moving, output_csv_path=None):
    """
    Recale un patient (moving) sur un atlas (fixed) avec la methode initiale

    Parametres : 
    - df_atlas        : DataFrame du patient de référence
    - df_moving       : DataFrame du patient à recaler
    - output_csv_path : chemin de sauvegarde du CSV résultat

    Retourne :
    - df_result : DataFrame avec colonnes x, y, z, dose recalées
    - Enregistre df_result sous format csv si output_csv_path est specifie
    """
    # Chargement
    moving_points = df_moving[['x', 'y', 'z']].values
    fixed_points = fixed_df[['x', 'y', 'z']].values

    # Extraction des contours
    f_boundary, _ = extract_boundary_points(fixed_points)
    m_boundary, _ = extract_boundary_points(moving_points)

    # Recalage
    aff = AffineRegistration(X=f_boundary, Y=m_boundary)
    TY_affine, _ = aff.register()
    TY_deform = optimize_cpd(f_boundary, TY_affine)
    
    # Propagation TPS
    full_transformed = apply_tps(m_boundary, TY_deform, moving_points)
    df_result = pd.DataFrame(full_transformed, columns=['x', 'y', 'z'])
    df_result['ID2013A'] = df_moving['ID2013A'].values

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_result.to_csv(output_csv_path, index=False, sep='\t')

    #print("Resultat orginal_rescaling : ", calculate_score(fixed_df, df_result))

    return df_result
