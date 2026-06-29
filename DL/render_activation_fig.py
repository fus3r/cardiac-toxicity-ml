"""
render_activation_fig.py — rendu PROPRE de la carte d'activation pour le rapport.

Régénère la figure à partir des données en cache (pas de ré-entraînement) :
  - DL/cache/activation_map_best.npy  (Grad-CAM moyen, 47x47x32)
  - DL/cache/grids_best.npz           (grilles de dose -> dose moyenne + masque cœur)

Corrige le défaut visuel de la version précédente (fonds NOIRS hors-cœur) :
fond BLANC opaque partout, voxels hors-cœur masqués en blanc. Sortie écrite
directement dans le dossier figures du rapport final.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
cam = np.load(ROOT / "DL/cache/activation_map_best.npy")          # (47,47,32)
grids = np.load(ROOT / "DL/cache/grids_best.npz", allow_pickle=True)["grids"]
dose_mean = grids.mean(axis=0)
heart = dose_mean > 0
cx, cy, cz = [int(round(np.average(np.where(heart)[k]))) for k in range(3)]

planes = [("Axial",    lambda a: a[:, :, cz].T),
          ("Coronal",  lambda a: a[:, cy, :].T),
          ("Sagittal", lambda a: a[cx, :, :].T)]

gray = matplotlib.colormaps["gray"].copy();  gray.set_bad("white")
turbo = matplotlib.colormaps["turbo"].copy(); turbo.set_bad("white")

def mask_outside(slc, heart_slc):
    out = slc.astype(float).copy()
    out[heart_slc < 0.5] = np.nan          # hors-cœur -> NaN -> blanc
    return out

fig, axes = plt.subplots(2, 3, figsize=(8.8, 4.4), facecolor="white",
                         gridspec_kw={"hspace": 0.04, "wspace": 0.03})
vmax = float(cam.max())
im = None
for j, (name, sl) in enumerate(planes):
    hs = sl(heart.astype(float))
    d = mask_outside(sl(dose_mean), hs)
    c = mask_outside(sl(cam), hs)
    axes[0, j].imshow(d, cmap=gray, origin="lower")
    axes[0, j].set_title(name, fontsize=11, pad=3)   # plan = en-tête de colonne
    im = axes[1, j].imshow(c, cmap=turbo, origin="lower", vmin=0, vmax=vmax)
    for r in (0, 1):
        axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        for s in axes[r, j].spines.values():
            s.set_visible(False)

# Étiquettes de ligne (à gauche), sans recouvrement
axes[0, 0].set_ylabel("Dose moyenne", fontsize=10.5)
axes[1, 0].set_ylabel("Activation\ndu risque", fontsize=10.5)

cbar = fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.022, pad=0.015)
cbar.set_label("activation (norm.)", fontsize=9)
cbar.ax.tick_params(labelsize=8)
fig.suptitle("Carte d'activation des risques cardiaques (Grad-CAM 3D)",
             fontsize=12, y=1.01)

out = ROOT / "latex/rapport_final/fig/activation_map.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Aplatir l'alpha sur fond blanc -> PNG RGB opaque (aucun rendu sombre possible)
img = Image.open(out).convert("RGBA")
bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
Image.alpha_composite(bg, img).convert("RGB").save(out)
print("OK ->", out, Image.open(out).mode, Image.open(out).size)
