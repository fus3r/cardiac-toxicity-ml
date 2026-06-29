import os
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from plotly.subplots import make_subplots

from metrique_rescaling import calculate_registration_score

# VISUALISATION PLOTLY 4D
def plot_plotly_4d(df_fixed, df_moving, patient_id, dice, image_folder):
    """Génère et sauvegarde la visualisation 4D interactive."""
    fig = go.Figure()
    
    # Patient de Référence (Fixed)
    fig.add_trace(go.Scatter3d(
        x=df_fixed['x'], y=df_fixed['y'], z=df_fixed['z'],
        mode='markers', name='Fixed (Ref)',
        marker=dict(size=2, color=df_fixed['ID2013A'], colorscale='Viridis', opacity=0.3, 
                    colorbar=dict(title="Dose Fixed", x=-0.12))
    ))
    
    # Patient Recalé (Transformed)
    fig.add_trace(go.Scatter3d(
        x=df_moving['x_transformed'], y=df_moving['y_transformed'], z=df_moving['z_transformed'],
        mode='markers', name='Moving Transformed',
        marker=dict(size=3, color=df_moving['ID2013A'], colorscale='Hot', opacity=0.7, 
                    colorbar=dict(title="Dose Moving", x=1.1))
    ))
    
    fig.update_layout(
        title=f"Patient {patient_id} | Dice Post-TPS: {dice:.4f}",
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Sauvegarde en HTML (interactif sans dépendance kaleido)
    save_path = os.path.join(image_folder, f"{patient_id}_4D_interactive.html")
    fig.write_html(save_path)
    print(f"Visualisation HTML sauvegardée : {save_path}")


# PALETTE ET HELPERS

_COLORS = {
    "atlas":    "#00d4ff",   # cyan  – patient de référence
    "original": "#ff6b35",   # orange – méthode originale CPD+TPS
    "ants":     "#a259ff",   # violet – méthode ANTsPy
    "error":    "RdYlGn_r",  # colorscale erreurs de distance
}


def _scatter(df, col_x='x', col_y='y', col_z='z', dose_col="ID2013A",
             color=None, colorscale=None, cmin=None, cmax=None,
             size=2, opacity=0.5, name="", showscale=False, colorbar_x=1.0):
    
    """Crée un trace Scatter3d réutilisable."""

    marker = dict(size=size, opacity=opacity)
    if colorscale:
        marker.update(
            color=df[dose_col].values if dose_col in df.columns else color,
            colorscale=colorscale,
            cmin=cmin, cmax=cmax,
            showscale=showscale,
            colorbar=dict(title="Dose (Gy)", x=colorbar_x, len=0.5)
        )
    else:
        marker["color"] = color

    return go.Scatter3d(x=df[col_x], y=df[col_y], z=df[col_z], mode="markers", name=name, marker=marker)

def _axis_cfg(title=""):
    return dict(title=title, backgroundcolor="#0d0d0d", gridcolor="#333", showbackground=True, tickfont=dict(size=9))


def _scene_cfg(title_x="X (mm)", title_y="Y (mm)", title_z="Z (mm)"):
    """assemble les trois axes en un seul dictionnaire de scène Plotly"""
    return dict(xaxis=_axis_cfg(title_x), yaxis=_axis_cfg(title_y), zaxis=_axis_cfg(title_z), bgcolor="#0d0d0d")


def _metrics_text(metrics: dict) -> str:
    """Formate les métriques en une annotation HTML."""
    return (
        f"<b>Score</b> : {metrics['total']:.1f}/100<br>"
        f"<b>Dice</b> : {metrics['dice']:.3f}  ({metrics['dice_pts']:.1f} pts)<br>"
        f"<b>MHD</b>  : {metrics['mhd']:.2f} mm  ({metrics['mhd_pts']:.1f} pts)<br>"
        f"<b>HD95</b> : {metrics['hd95']:.2f} mm"
    )


def _point_errors(df_ref, df_result):
    """Distance au plus proche voisin de df_result vers df_ref (en mm)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(df_ref[["x", "y", "z"]].values)
    dist, _ = tree.query(df_result[["x", "y", "z"]].values)
    return dist


# FONCTION PRINCIPALE

def compare_doses(df_moving, df_original, df_ants, patient_id="patient", output_path=None, show=True):
    """
    3 subplots colorés par dose :
    Atlas | Méthode originale | ANTsPy
    """
    dose_min = df_moving["ID2013A"].min()
    dose_max = df_moving["ID2013A"].max()

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Unrescaled (référence)", "Méthode originale (CPD+TPS)", "Méthode ANTsPy"],
        horizontal_spacing=0.04,
    )

    fig.add_trace(_scatter(df_moving,    colorscale="Viridis", cmin=dose_min, cmax=dose_max, dose_col="ID2013A", showscale=True,  colorbar_x=0.30, size=2, opacity=0.6, name="Unrescaled"),    row=1, col=1)
    fig.add_trace(_scatter(df_original, colorscale="Viridis", cmin=dose_min, cmax=dose_max, dose_col="ID2013A", showscale=False,                  size=2, opacity=0.6, name="Original"), row=1, col=2)
    fig.add_trace(_scatter(df_ants,     colorscale="Viridis", cmin=dose_min, cmax=dose_max, dose_col="ID2013A", showscale=False,                  size=2, opacity=0.6, name="ANTsPy"),   row=1, col=3)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0a0a",
        title=dict(text=f"<b>Comparaison doses — {patient_id}</b>", x=0.5, xanchor="center"),
        scene=_scene_cfg(), scene2=_scene_cfg(), scene3=_scene_cfg(),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=80), height=600,
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
    if show:
        fig.show()
        
    return fig


def compare_errors(df_atlas, df_original, df_ants, patient_id="patient", output_path=None, show=True):
    """
    2 subplots colorés par erreur de distance :
    Méthode originale | ANTsPy
    """
    m_orig = calculate_registration_score(df_atlas, df_original)
    m_ants = calculate_registration_score(df_atlas, df_ants)

    err_orig = _point_errors(df_atlas, df_original)
    err_ants = _point_errors(df_atlas, df_ants)
    emax     = max(np.percentile(err_orig, 95), np.percentile(err_ants, 95), 1.0)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"Méthode originale (CPD+TPS)<br>{_metrics_text(m_orig)}",
            f"Méthode ANTsPy<br>{_metrics_text(m_ants)}",
        ],
        horizontal_spacing=0.04,
    )

    for col, (df_res, err, name) in enumerate([(df_original, err_orig, "Original"), (df_ants, err_ants, "ANTsPy")], start=1):
        df_err = df_res.copy()
        df_err["_err"] = err
        fig.add_trace(
            go.Scatter3d(
                x=df_err["x"], y=df_err["y"], z=df_err["z"],
                mode="markers", name=name,
                marker=dict(
                    size=2, opacity=0.7,
                    color=df_err["_err"],
                    colorscale=_COLORS["error"],
                    cmin=0, cmax=emax,
                    showscale=(col == 1),
                    colorbar=dict(title="Erreur (mm)", x=0.44, len=0.7),
                ),
            ),
            row=1, col=col,
        )

    winner = "ANTsPy" if m_ants["total"] > m_orig["total"] else "Original"
    delta  = abs(m_ants["total"] - m_orig["total"])

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0a0a",
        title=dict(
            text=f"<b>Erreurs de recalage — {patient_id}</b><br><span style='font-size:13px'>Meilleure méthode : <b>{winner}</b> (+{delta:.1f} pts)</span>",
            x=0.5, xanchor="center", font=dict(size=16),
        ),
        scene=_scene_cfg(), scene2=_scene_cfg(),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=100), height=600,
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
    if show:
        fig.show()

    return fig


# VUE SUPERPOSÉE (optionnel) - Atlas + Original + ANTsPy dans une seule scène

def overlay_rescaling(df_atlas, df_original, df_ants, patient_id="patient", output_path=None, show=True):
    """
    3 subplots, chacun avec 2 nuages superposés :
    Atlas + ANTsPy | Atlas + Original | ANTsPy + Original
    """
    m_orig = calculate_registration_score(df_atlas, df_original)
    m_ants = calculate_registration_score(df_atlas, df_ants)

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            "Atlas + ANTsPy",
            "Atlas + Original",
            "ANTsPy + Original",
        ],
        horizontal_spacing=0.04,
    )

    # Subplot 1 : Atlas + ANTsPy
    fig.add_trace(_scatter(df_atlas,    color=_COLORS["atlas"], size=2, opacity=0.35, name="Atlas"),                             row=1, col=1)
    fig.add_trace(_scatter(df_ants,     color=_COLORS["ants"],  size=2, opacity=0.55, name=f"ANTsPy — {m_ants['total']:.1f}"),   row=1, col=1)

    # Subplot 2 : Atlas + Original
    fig.add_trace(_scatter(df_atlas,    color=_COLORS["atlas"],    size=2, opacity=0.35, name="Atlas"),                              row=1, col=2)
    fig.add_trace(_scatter(df_original, color=_COLORS["original"], size=2, opacity=0.55, name=f"Original — {m_orig['total']:.1f}"), row=1, col=2)

    # Subplot 3 : ANTsPy + Original
    fig.add_trace(_scatter(df_ants,     color=_COLORS["ants"],     size=2, opacity=0.55, name=f"ANTsPy — {m_ants['total']:.1f}"),   row=1, col=3)
    fig.add_trace(_scatter(df_original, color=_COLORS["original"], size=2, opacity=0.55, name=f"Original — {m_orig['total']:.1f}"), row=1, col=3)

    winner = "ANTsPy" if m_ants["total"] > m_orig["total"] else "Original"
    delta  = abs(m_ants["total"] - m_orig["total"])

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0a0a",
        title=dict(
            text=f"<b>Superposition — {patient_id}</b>  |  Meilleure méthode : <b>{winner}</b> (+{delta:.1f} pts)",
            x=0.5, xanchor="center", font=dict(size=16),
        ),
        scene=_scene_cfg(), scene2=_scene_cfg(), scene3=_scene_cfg(),
        margin=dict(l=0, r=0, b=0, t=80), height=600,
        legend=dict(x=0.01, y=0.99),
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
    if show:
        fig.show()
    return fig