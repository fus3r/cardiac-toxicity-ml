"""
activation_map.py — Carte d'activation des risques (Grad-CAM 3D)
================================================================
Objectif Phase 3 : localiser les sous-régions cardiaques les plus associées
à la cardiotoxicité, à partir du CNN3D entraîné sur les volumes de dose
harmonisés (grille commune 47x47x32).

Méthode :
  1. Entraîne un CNN3D (dose seule, sans features cliniques) sur TOUTE la
     cohorte — on cherche ici une carte explicative moyenne, pas une mesure
     de généralisation.
  2. Grad-CAM 3D sur le dernier bloc convolutif : pondère les cartes de
     features par le gradient du logit positif, ReLU, ré-échantillonne en
     47x47x32.
  3. Moyenne sur les patients événement (+) -> carte d'activation des risques.
  4. Sauvegarde : coupes orthogonales (activation + dose moyenne) + .npy.

Usage :
  python DL/activation_map.py --data ants --epochs 40
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dl_cardiac import (
    build_dataset, CardiacDataset, CNN3DLogit, GRID_SHAPE, TARGET,
    FIG_DIR, CACHE_DIR, RANDOM_STATE,
)


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_full(grids, df, device, epochs, lr=5e-4, batch=32, dropout=0.4):
    """Entraîne un CNN3D (dose seule) sur toute la cohorte."""
    events = df[TARGET].to_numpy()
    dose_scale = float(np.percentile(grids[grids > 0], 99))
    X_empty = np.empty((len(df), 0), dtype=np.float32)
    ds = CardiacDataset(grids[df["grid_pos"].to_numpy()], X_empty, events,
                        dose_scale, flatten_mask=None)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    model = CNN3DLogit(n_clinical=0, dropout=dropout).to(device)
    n_pos = int(events.sum()); n_neg = len(events) - n_pos
    pw = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for b in loader:
            v = b["voxels"].to(device); c = b["clinical"].to(device); y = b["event"].to(device)
            opt.zero_grad()
            loss = criterion(model(v, c), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(y)
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{epochs}  loss={tot/len(events):.4f}")
    return model, dose_scale


def grad_cam_3d(model, grid, device):
    """Grad-CAM 3D sur le dernier Conv3d de l'encodeur. Retourne (47,47,32).

    On capture la feature map via un forward hook + retain_grad() (plus robuste
    que full_backward_hook, notamment sur MPS)."""
    model.eval()
    last_conv = [m for m in model.encoder if isinstance(m, nn.Conv3d)][-1]
    store = {}

    def fwd_hook(_, __, out):
        out.retain_grad()
        store["fmap"] = out

    h = last_conv.register_forward_hook(fwd_hook)
    x = torch.from_numpy(grid[np.newaxis, np.newaxis].copy()).to(device)
    c = torch.empty((1, 0), device=device)
    model.zero_grad()
    logit = model(x, c)
    logit.backward()
    h.remove()

    fmap = store["fmap"].detach()                   # (1, C, d, h, w)
    g = store["fmap"].grad.detach()                 # (1, C, d, h, w)
    weights = g.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=GRID_SHAPE, mode="trilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    mx = cam.max()
    return cam / mx if mx > 0 else cam


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="ants")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    print(f"  device={device}")

    df, grids = build_dataset(args.data, args.limit)
    dose_scale = float(np.percentile(grids[grids > 0], 99))

    print("  Entraînement CNN3D (dose seule, toute la cohorte)...")
    model, dose_scale = train_full(grids, df, device, args.epochs)

    # Grad-CAM moyenné sur les patients événement (+)
    pos_idx = np.where(df[TARGET].to_numpy() == 1)[0]
    print(f"  Grad-CAM sur {len(pos_idx)} patients événement (+)...")
    cam_sum = np.zeros(GRID_SHAPE, dtype=np.float64)
    for i in pos_idx:
        g = grids[df["grid_pos"].to_numpy()[i]] / dose_scale
        cam_sum += grad_cam_3d(model, g.astype(np.float32), device)
    cam_mean = (cam_sum / len(pos_idx)).astype(np.float32)

    dose_mean = grids.mean(axis=0)
    np.save(CACHE_DIR / f"activation_map_{args.data}.npy", cam_mean)

    # ----- visualisation : coupes orthogonales -----
    heart = dose_mean > 0
    cx, cy, cz = [int(np.round(np.average(np.where(heart)[k]))) for k in range(3)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    planes = [("Axial (z)", lambda a: a[:, :, cz].T),
              ("Coronal (y)", lambda a: a[:, cy, :].T),
              ("Sagittal (x)", lambda a: a[cx, :, :].T)]
    for j, (name, sl) in enumerate(planes):
        axes[0, j].imshow(sl(dose_mean), cmap="gray", origin="lower")
        axes[0, j].set_title(f"Dose moyenne — {name}")
        axes[0, j].axis("off")
        axes[1, j].imshow(sl(dose_mean), cmap="gray", origin="lower")
        im = axes[1, j].imshow(sl(cam_mean), cmap="jet", alpha=0.55, origin="lower")
        axes[1, j].set_title(f"Activation risque — {name}")
        axes[1, j].axis("off")
    fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.025, label="activation (norm.)")
    fig.suptitle("Carte d'activation des risques (Grad-CAM 3D, CNN3D dose)", fontsize=13)
    out = FIG_DIR / f"activation_map_{args.data}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Carte sauvegardée : {out}")


if __name__ == "__main__":
    main()
