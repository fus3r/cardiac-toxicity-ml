"""
dl_cardiac.py — Pipeline Deep Learning (MLP + CNN3D) corrigé
============================================================
Prédiction de pathologie cardiaque sévère (grade >=3 CTCAE) à partir des
matrices de dose 3D recalées + features cliniques (cohorte FCCSS).

Reprise / correction du travail initial (DL/DL_MLP_CLASS.py, DL/DL_CNN3D.py,
DL/DL_MLP_CLASS_FOCAL_LOSS.py). Corrections principales :

  1. CHEMINS LOCAUX  : auto-détection de la racine du dépôt (plus de chemin
     cluster /usr/users/...).

  2. SCATTER GRILLE DENSE 47x47x32  : les CSV projetés sont des nuages de
     points (x,y,z,dose) ; l'ordre des lignes n'est PAS garanti identique
     d'un patient à l'autre (vérifié faux pour original_csv_projected), et il
     y a des coordonnées dupliquées. On reconstruit donc une grille dense
     canonique par indexation (x,y,z)->(i,j,k). -> corrige la cause racine de
     la "val loss horrible" (voxel i != même endroit selon les patients).

  3. ONE-HOT iccc_type ALIGNÉ  : le one-hot était fait séparément sur
     train/test (=> colonnes désalignées / crash possible quand une catégorie
     rare manque dans un fold). On fit les catégories sur le train et on
     réindexe le test dessus.

  4. VAL LOSS + EARLY STOPPING SANS FUITE  : split interne train/val (le fold
     de test n'est JAMAIS utilisé pour l'early stopping). Historique train/val,
     courbes 2 panneaux comme les slides, restauration du meilleur poids.

  5. STABILISATION  : BatchNorm, LR plus bas, gradient clipping, scheduler
     ReduceLROnPlateau sur la val loss, MLP plus compact (le "funnel"
     n_voxel -> n_voxel/4 -> 256 -> 64 surparamétrait : ~400M poids).

  6. VARIANCE DES FOLDS / ctr  : option StratifiedGroupKFold par centre (ctr=1
     représente 77,7% des patients, ctr=3 a un taux d'événement de 6,6% vs
     19,2% pour ctr=1). Rapport de la composition par centre de chaque fold.

Usage :
  python DL/dl_cardiac.py --model cnn3d --features clinical --data ants
  python DL/dl_cardiac.py --model mlp   --features none     --data ants
  python DL/dl_cardiac.py --model cnn3d --features clinical --cv group
  python DL/dl_cardiac.py --limit 200 --epochs 10        # test rapide
"""

import re
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#  Chemins (auto-détection de la racine du dépôt)                             #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
CLINICAL_CSV = REPO_ROOT / "data" / "RT_Thorax_v1.csv"
CACHE_DIR = REPO_ROOT / "DL" / "cache"
FIG_DIR = REPO_ROOT / "DL" / "figures"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Variantes de données disponibles (dossiers de CSV projetés sur grille commune)
DATA_VARIANTS = {
    "ants":     REPO_ROOT / "rescaling" / "ants_csv_projected",      # recalage ANTsPy (piloté dose, ancien)
    "original": REPO_ROOT / "rescaling" / "original_csv_projected",  # projection brute
    "best":     REPO_ROOT / "rescaling" / "best_score_csv",          # meilleur score recalage (ancien)
    "mask":     REPO_ROOT / "rescaling" / "mask_csv_projected",      # recalage CORRIGÉ sur masque (anatomie)
}

# Géométrie de grille (vérifiée sur les données : pas = 2.0 sur les 3 axes)
GRID_SHAPE = (47, 47, 32)
X0, Y0, Z0 = -28.0, 872.0, 48.0
STEP = 2.0

TARGET = "Pathologie_cardiaque"
CATEGORICAL_COLS = ["iccc_type", "gender"]

FEATURE_SETS = {
    "none": [],
    "clinical": ["mean", "Year_date_diag", "age_diag", "iccc_type", "anthra_1K"],
    "clin_nomean": ["Year_date_diag", "age_diag", "iccc_type", "anthra_1K"],  # ablation : sans dose moyenne
    "mean_only": ["mean"],
    "clinical_plus": ["mean", "Year_date_diag", "age_diag", "iccc_type",
                      "anthra_1K", "do_anthra_1K", "V20", "V15", "V5"],
}

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
#  Chargement + scatter vers grille dense                                     #
# --------------------------------------------------------------------------- #
def read_projected(filepath):
    """Lit un CSV projeté (x, y, z, ID2013A) ; tab puis espaces en secours."""
    df = pd.read_csv(filepath, sep="\t")
    if "ID2013A" not in df.columns:
        df = pd.read_csv(filepath, sep=r"\s+", engine="python")
    return df


def scatter_to_grid(df):
    """Nuage de points (x,y,z,dose) -> grille dense 47x47x32.

    - indexation déterministe (x,y,z) -> (i,j,k) par (coord - origine) / pas
    - agrège les coordonnées dupliquées par la moyenne (les fichiers
      'original' contiennent ~6500 doublons de coordonnées).
    """
    ix = np.rint((df["x"].to_numpy() - X0) / STEP).astype(np.int64)
    iy = np.rint((df["y"].to_numpy() - Y0) / STEP).astype(np.int64)
    iz = np.rint((df["z"].to_numpy() - Z0) / STEP).astype(np.int64)

    ok = (
        (ix >= 0) & (ix < GRID_SHAPE[0])
        & (iy >= 0) & (iy < GRID_SHAPE[1])
        & (iz >= 0) & (iz < GRID_SHAPE[2])
    )
    ix, iy, iz = ix[ok], iy[ok], iz[ok]
    dose = df["ID2013A"].to_numpy(dtype=np.float64)[ok]

    flat = (ix * GRID_SHAPE[1] + iy) * GRID_SHAPE[2] + iz
    n = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]
    acc = np.zeros(n, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.float64)
    np.add.at(acc, flat, dose)
    np.add.at(cnt, flat, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    return grid.reshape(GRID_SHAPE).astype(np.float32)


def parse_name(fname):
    m = re.match(r"^(\d+)_(\d+)_projected\.csv$", fname)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def build_or_load_grids(variant, limit=None):
    """Construit (ou recharge depuis le cache) le tenseur de grilles denses.

    Retourne un DataFrame index (ctr, numcent) + le tableau grids (N,47,47,32).
    """
    proj_dir = DATA_VARIANTS[variant]
    assert proj_dir.exists(), f"Dossier introuvable : {proj_dir}"

    cache = CACHE_DIR / f"grids_{variant}{'' if limit is None else f'_lim{limit}'}.npz"
    if cache.exists():
        print(f"  [cache] {cache.name}")
        z = np.load(cache, allow_pickle=True)
        idx = pd.DataFrame({"ctr": z["ctr"], "numcent": z["numcent"]})
        return idx, z["grids"]

    files = sorted(proj_dir.glob("*_projected.csv"))
    if limit:
        files = files[:limit]
    print(f"  Construction grilles depuis {len(files)} fichiers ({variant})...")

    ctrs, nums, grids = [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        ctr, num = parse_name(f.name)
        if ctr is None:
            continue
        grids.append(scatter_to_grid(read_projected(f)))
        ctrs.append(ctr)
        nums.append(num)
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(files)}  ({time.time() - t0:.0f}s)")

    grids = np.stack(grids).astype(np.float32)
    idx = pd.DataFrame({"ctr": ctrs, "numcent": nums})
    np.savez_compressed(cache, grids=grids, ctr=idx["ctr"].to_numpy(),
                        numcent=idx["numcent"].to_numpy())
    print(f"  Grilles : {grids.shape}  (sauvé : {cache.name}, {time.time() - t0:.0f}s)")
    return idx, grids


def build_dataset(variant, limit=None):
    """Jointure clinique <-> grilles 3D sur (ctr, numcent)."""
    clinical = pd.read_csv(CLINICAL_CSV)
    idx, grids = build_or_load_grids(variant, limit)
    idx = idx.reset_index().rename(columns={"index": "grid_pos"})
    merged = clinical.merge(idx, on=["ctr", "numcent"], how="inner")
    merged = merged.reset_index(drop=True)
    print(f"  Jointure : {len(merged)} patients "
          f"({int(merged[TARGET].sum())} événements, "
          f"{100 * merged[TARGET].mean():.1f}%)")
    return merged, grids


# --------------------------------------------------------------------------- #
#  Features cliniques (one-hot aligné train/test)                             #
# --------------------------------------------------------------------------- #
class ClinicalPrep:
    """Fit sur le train, applique au test. Garantit le MÊME espace de colonnes
    (one-hot), la même imputation (médiane train) et le même scaling."""

    def __init__(self, feature_list):
        self.feature_list = feature_list
        self.columns = None
        self.medians = None
        self.scaler = None

    def _dummies(self, df):
        sub = df[self.feature_list].copy()
        cats = [c for c in CATEGORICAL_COLS if c in self.feature_list]
        if cats:
            sub = pd.get_dummies(sub, columns=cats, drop_first=False)
        return sub

    def fit_transform(self, df):
        if not self.feature_list:
            return np.empty((len(df), 0), dtype=np.float32)
        sub = self._dummies(df)
        self.columns = list(sub.columns)
        self.medians = sub.median(numeric_only=True)
        sub = sub.fillna(self.medians)
        X = sub.to_numpy(dtype=np.float32)
        self.scaler = StandardScaler().fit(X)
        return self.scaler.transform(X).astype(np.float32)

    def transform(self, df):
        if not self.feature_list:
            return np.empty((len(df), 0), dtype=np.float32)
        sub = self._dummies(df)
        # réindexation sur les colonnes du train (catégories manquantes -> 0)
        sub = sub.reindex(columns=self.columns, fill_value=0)
        sub = sub.fillna(self.medians)
        X = sub.to_numpy(dtype=np.float32)
        return self.scaler.transform(X).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Dataset torch                                                              #
# --------------------------------------------------------------------------- #
class CardiacDataset(Dataset):
    def __init__(self, grids, X_clinical, events, dose_scale, flatten_mask=None):
        self.grids = grids
        self.X_clinical = X_clinical.astype(np.float32)
        self.events = events.astype(np.float32)
        self.dose_scale = dose_scale
        self.flatten_mask = flatten_mask  # MLP : indices des voxels actifs

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        g = self.grids[idx] / self.dose_scale
        if self.flatten_mask is not None:
            x = torch.from_numpy(g.reshape(-1)[self.flatten_mask].copy())
        else:
            x = torch.from_numpy(g[np.newaxis].copy())  # (1, 47, 47, 32)
        return {
            "voxels": x,
            "clinical": torch.from_numpy(self.X_clinical[idx]),
            "event": torch.tensor(self.events[idx]),
        }


# --------------------------------------------------------------------------- #
#  Modèles                                                                    #
# --------------------------------------------------------------------------- #
class MLPLogit(nn.Module):
    """MLP compact sur voxels actifs (+ features cliniques concaténées).

    NB : l'architecture initiale en entonnoir (n_voxel -> n_voxel/4 -> 256 ...)
    crée ~400M de poids pour ~1100 exemples => surapprentissage massif et
    val loss instable. On la remplace par un MLP régularisé (BatchNorm+Dropout).
    """

    def __init__(self, n_voxels, n_clinical, hidden=(512, 128), dropout=0.4):
        super().__init__()
        layers, d = [], n_voxels
        layers += [nn.BatchNorm1d(d)]
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.voxel_branch = nn.Sequential(*layers)
        self.head = nn.Linear(d + n_clinical, 1)

    def forward(self, voxels, clinical):
        emb = self.voxel_branch(voxels)
        x = torch.cat([emb, clinical], dim=1) if clinical.shape[1] > 0 else emb
        return self.head(x).squeeze(1)


class CNN3DLogit(nn.Module):
    """Encodeur CNN3D (avec BatchNorm) + tête MLP. Features cliniques
    concaténées à l'embedding GAP."""

    def __init__(self, n_clinical, dropout=0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(64 + n_clinical, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, voxels, clinical):
        x = self.encoder(voxels)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        x = torch.cat([x, clinical], dim=1) if clinical.shape[1] > 0 else x
        return self.head(x).squeeze(1)

    @torch.no_grad()
    def embed(self, voxels):
        """Embedding 3D (features GAP, 64-d) — pour la fusion avec un classifieur tabulaire."""
        x = self.encoder(voxels)
        return self.gap(x).flatten(1)  # (B, 64)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        a_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()


# --------------------------------------------------------------------------- #
#  Entraînement d'un fold (avec val interne + early stopping)                 #
# --------------------------------------------------------------------------- #
def train_one_fold(model, train_loader, val_loader, criterion, device,
                   max_epochs, lr, patience, weight_decay=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5)

    history = {"train": [], "val": []}
    best_val, best_state, bad = float("inf"), None, 0

    def run_epoch(loader, train):
        model.train(train)
        tot, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                v = batch["voxels"].to(device)
                c = batch["clinical"].to(device)
                y = batch["event"].to(device)
                if train:
                    optimizer.zero_grad()
                logit = model(v, c)
                loss = criterion(logit, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                tot += loss.item() * len(y)
                n += len(y)
        return tot / max(n, 1)

    for epoch in range(max_epochs):
        tr = run_epoch(train_loader, True)
        vl = run_epoch(val_loader, False)
        scheduler.step(vl)
        history["train"].append(tr)
        history["val"].append(vl)

        if vl < best_val - 1e-4:
            best_val, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, evts = [], []
    for batch in loader:
        v = batch["voxels"].to(device)
        c = batch["clinical"].to(device)
        probs.append(torch.sigmoid(model(v, c)).cpu().numpy())
        evts.append(batch["event"].numpy())
    return np.concatenate(probs), np.concatenate(evts)


# --------------------------------------------------------------------------- #
#  Validation croisée                                                         #
# --------------------------------------------------------------------------- #
def run_cv(args, df, grids, device):
    feature_list = FEATURE_SETS[args.features]
    y = df[TARGET].to_numpy()
    groups = df["ctr"].to_numpy()

    # masque des voxels actifs (pour le MLP) : union des voxels non nuls
    active_mask = None
    n_voxels = None
    if args.model == "mlp":
        active = (grids != 0).any(axis=0).reshape(-1)
        active_mask = np.where(active)[0]
        n_voxels = active_mask.size
        print(f"  MLP : {n_voxels} voxels actifs / {grids[0].size}")

    dose_scale = float(np.percentile(grids[grids > 0], 99)) if (grids > 0).any() else 1.0
    print(f"  dose_scale (p99 nonzero) = {dose_scale:.2f}")

    # CV externe
    if args.cv == "group":
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                        random_state=RANDOM_STATE)
        splits = list(splitter.split(df, y, groups))
        print(f"  CV = StratifiedGroupKFold par ctr ({args.folds} folds)")
    else:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True,
                                   random_state=RANDOM_STATE)
        splits = list(splitter.split(df, y))
        print(f"  CV = StratifiedKFold ({args.folds} folds)")

    auc_folds, histories = [], []
    oof_probs = np.full(len(df), np.nan)

    for fold, (tr_idx, te_idx) in enumerate(splits):
        # split interne train/val (anti-fuite ; le fold de test n'est pas utilisé)
        rng = np.random.RandomState(RANDOM_STATE + fold)
        tr_idx = rng.permutation(tr_idx)
        n_val = max(1, int(0.15 * len(tr_idx)))
        val_idx, tr_in = tr_idx[:n_val], tr_idx[n_val:]

        df_tr, df_val, df_te = df.iloc[tr_in], df.iloc[val_idx], df.iloc[te_idx]

        prep = ClinicalPrep(feature_list)
        X_tr = prep.fit_transform(df_tr)
        X_val = prep.transform(df_val)
        X_te = prep.transform(df_te)
        n_clin = X_tr.shape[1]

        def make_loader(d, X, shuffle):
            ds = CardiacDataset(grids[d["grid_pos"].to_numpy()], X,
                                d[TARGET].to_numpy(), dose_scale, active_mask)
            return DataLoader(ds, batch_size=args.batch, shuffle=shuffle)

        train_loader = make_loader(df_tr, X_tr, True)
        val_loader = make_loader(df_val, X_val, False)
        test_loader = make_loader(df_te, X_te, False)

        if args.model == "mlp":
            model = MLPLogit(n_voxels, n_clin, dropout=args.dropout).to(device)
        else:
            model = CNN3DLogit(n_clin, dropout=args.dropout).to(device)

        # loss
        n_pos = int(df_tr[TARGET].sum())
        n_neg = len(df_tr) - n_pos
        if args.loss == "focal":
            criterion = FocalLoss(alpha=0.75, gamma=2.0)
        else:
            pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        n_params = sum(p.numel() for p in model.parameters())
        hist = train_one_fold(model, train_loader, val_loader, criterion, device,
                              args.epochs, args.lr, args.patience)
        histories.append(hist)

        probs, evts = predict(model, test_loader, device)
        auc = roc_auc_score(evts, probs)
        auc_folds.append(auc)
        oof_probs[te_idx] = probs

        # composition centre du fold de test
        ctr_te = df_te["ctr"].value_counts().to_dict()
        print(f"  Fold {fold + 1}/{args.folds} | params={n_params:,} | "
              f"epochs={len(hist['train'])} | AUC={auc:.4f} | "
              f"test ctr={ctr_te}")

    # ----- résumé -----
    auc_folds = np.array(auc_folds)
    print(f"\n  AUC-ROC : {auc_folds.mean():.4f} ± {auc_folds.std():.4f}  "
          f"(folds: {', '.join(f'{a:.3f}' for a in auc_folds)})")

    valid = ~np.isnan(oof_probs)
    preds = (oof_probs[valid] >= 0.5).astype(int)
    yt = y[valid]
    tn, fp, fn, tp = confusion_matrix(yt, preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    print(f"\n  Matrice de confusion (OOF, seuil 0.5) :")
    print(f"                  Prédit 0    Prédit 1")
    print(f"  Réel 0 (sain)   {tn:8d}    {fp:8d}")
    print(f"  Réel 1 (event)  {fn:8d}    {tp:8d}")
    print(f"  Sensibilité : {sens:.3f}   Spécificité : {spec:.3f}")
    print(f"  AUC-ROC (OOF poolé) : {roc_auc_score(yt, oof_probs[valid]):.4f}")

    # ----- courbes train/val -----
    plot_curves(histories, args)
    return auc_folds, (sens, spec)


def plot_curves(histories, args):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    cmap = plt.cm.tab10
    for i, h in enumerate(histories):
        a1.plot(h["train"], color=cmap(i), label=f"Fold {i + 1}")
        a2.plot(h["val"], color=cmap(i), label=f"Fold {i + 1}")
    a1.set_title("Train loss — folds"); a1.set_xlabel("Epoch"); a1.set_ylabel("Loss")
    a2.set_title("Val loss — folds"); a2.set_xlabel("Epoch"); a2.set_ylabel("Loss")
    a2.legend(fontsize=8)
    title = f"{args.model.upper()} | features={args.features} | data={args.data} | cv={args.cv}"
    fig.suptitle(title)
    fig.tight_layout()
    out = FIG_DIR / f"loss_{args.model}_{args.features}_{args.data}_{args.cv}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\n  Courbes sauvegardées : {out.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Pipeline DL cardiaque (corrigé)")
    p.add_argument("--model", choices=["mlp", "cnn3d"], default="cnn3d")
    p.add_argument("--features", choices=list(FEATURE_SETS), default="clinical")
    p.add_argument("--data", choices=list(DATA_VARIANTS), default="ants")
    p.add_argument("--cv", choices=["stratified", "group"], default="stratified")
    p.add_argument("--loss", choices=["bce", "focal"], default="bce")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--limit", type=int, default=None, help="limiter le nb de patients (test rapide)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("=" * 64)
    print(f"  DL cardiaque | model={args.model} features={args.features} "
          f"data={args.data} cv={args.cv} loss={args.loss}")
    print(f"  device={device}")
    print("=" * 64)

    df, grids = build_dataset(args.data, args.limit)
    t0 = time.time()
    run_cv(args, df, grids, device)
    print(f"\n  Temps total CV : {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
