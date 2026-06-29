"""
dl_mednet.py — Transfer learning MedNet / MedicalNet (ResNet-18 3D pré-entraîné)
================================================================================
Reprend l'idée du `DL_MEDNET.py` du collègue (suggestion de l'encadrant) mais :
  - utilise les poids MedicalNet via MONAI (`resnet18(pretrained=True, ...)`),
    donc REPRODUCTIBLE EN LOCAL (pas de dépendance cluster) ;
  - applique le PROTOCOLE CORRIGÉ : grille dense, one-hot aligné, split val
    interne (pas de fuite vers le test), best-weight restore, early stopping.

ResNet-18 3D pré-entraîné sur 23 datasets médicaux (Med3D / Tencent MedicalNet),
encodeur 512-d, fine-tuné sur nos volumes de dose ; tête MLP + features cliniques.

Usage : python -u DL/dl_mednet.py --features clinical --epochs 30
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

from monai.networks.nets import resnet18
import dl_cardiac as dl

TARGET = dl.TARGET
SEED = dl.RANDOM_STATE


class MedNet(nn.Module):
    """ResNet-18 3D MedicalNet pré-entraîné (encodeur gelé/fine-tuné) + tête clinique."""
    def __init__(self, n_clinical, dropout=0.4, freeze_until="layer1"):
        super().__init__()
        self.encoder = resnet18(
            pretrained=True, spatial_dims=3, n_input_channels=1,
            feed_forward=False, shortcut_type="A", bias_downsample=True,
        )  # forward(x) -> (B, 512)
        # gel des blocs bas-niveau (génériques) pour limiter le surapprentissage
        self._frozen = []
        if freeze_until:
            order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
            stop = order.index(freeze_until)
            names = order[: stop + 1]
            for n, m in self.encoder.named_children():
                if n in names:
                    self._frozen.append(m)
                    for p in m.parameters():
                        p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(512 + n_clinical, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def freeze_eval(self):
        """Garde les blocs gelés en mode eval (BatchNorm : stats figées)."""
        for m in self._frozen:
            m.eval()

    def forward(self, voxels, clinical):
        f = self.encoder(voxels)
        if clinical.shape[1] > 0:
            f = torch.cat([f, clinical], dim=1)
        return self.head(f).squeeze(1)

    @torch.no_grad()
    def embed(self, voxels):
        return self.encoder(voxels)


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def make_loader(grids, rows, X, df, dose_scale, shuffle, batch):
    sub = df.iloc[rows]
    ds = dl.CardiacDataset(grids[sub["grid_pos"].to_numpy()], X,
                           sub[TARGET].to_numpy(), dose_scale, flatten_mask=None)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def train_fold(model, train_loader, val_loader, criterion, device, epochs, lr, patience):
    """Comme dl.train_one_fold mais garde les blocs gelés en eval (BN figé)."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=4, factor=0.5)
    best, best_state, bad = float("inf"), None, 0
    hist = {"train": [], "val": []}

    def epoch(loader, train):
        model.train(train); model.freeze_eval()
        tot, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for b in loader:
                v = b["voxels"].to(device); c = b["clinical"].to(device); y = b["event"].to(device)
                if train: opt.zero_grad()
                loss = criterion(model(v, c), y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                tot += loss.item() * len(y); n += len(y)
        return tot / max(n, 1)

    for ep in range(epochs):
        tr = epoch(train_loader, True); vl = epoch(val_loader, False)
        sched.step(vl); hist["train"].append(tr); hist["val"].append(vl)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if best_state: model.load_state_dict(best_state)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=list(dl.FEATURE_SETS), default="clinical")
    ap.add_argument("--data", default="ants")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze_until", default="layer2", help="conv1|bn1|layer1|layer2|layer3|layer4|'' (rien)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"device={device} | features={args.features} | freeze_until={args.freeze_until}")

    df, grids = dl.build_dataset(args.data, args.limit)
    y = df[TARGET].to_numpy()
    dose_scale = float(np.percentile(grids[grids > 0], 99))
    feats = dl.FEATURE_SETS[args.features]

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, y))
    oof = np.full(len(df), np.nan); aucs = []; hists = []

    for f, (tr, te) in enumerate(splits):
        rng = np.random.RandomState(SEED + f); tr = rng.permutation(tr)
        nv = max(1, int(0.15 * len(tr))); val, fit = tr[:nv], tr[nv:]

        prep = dl.ClinicalPrep(feats)
        Xf = prep.fit_transform(df.iloc[fit]); Xv = prep.transform(df.iloc[val]); Xt = prep.transform(df.iloc[te])
        nclin = Xf.shape[1]

        tl = make_loader(grids, fit, Xf, df, dose_scale, True, args.batch)
        vl = make_loader(grids, val, Xv, df, dose_scale, False, args.batch)
        el = make_loader(grids, te,  Xt, df, dose_scale, False, args.batch)

        model = MedNet(nclin, dropout=0.4, freeze_until=(args.freeze_until or None)).to(device)
        npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
        npos = int(df.iloc[fit][TARGET].sum()); nneg = len(fit) - npos
        pw = torch.tensor([nneg / max(npos, 1)], dtype=torch.float32, device=device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)

        h = train_fold(model, tl, vl, crit, device, args.epochs, args.lr, args.patience)
        hists.append(h)
        p, e = dl.predict(model, el, device)
        auc = roc_auc_score(e, p); aucs.append(auc); oof[te] = p
        print(f"  Fold {f+1}/5 | trainable={npar:,} | epochs={len(h['train'])} | "
              f"AUC={auc:.4f} | test ctr={df.iloc[te]['ctr'].value_counts().to_dict()}", flush=True)

    aucs = np.array(aucs)
    print(f"\n  MedNet ({args.features}) AUC-ROC : {aucs.mean():.4f} ± {aucs.std():.4f}  "
          f"(folds: {', '.join(f'{a:.3f}' for a in aucs)})")
    valid = ~np.isnan(oof)
    for thr in (0.5, float(y.mean())):
        tn, fp, fn, tp = confusion_matrix(y[valid], (oof[valid] >= thr).astype(int)).ravel()
        print(f"  seuil={thr:.2f} : Sens={tp/(tp+fn):.3f}  Spéc={tn/(tn+fp):.3f}")
    print(f"  AUC OOF poolée : {roc_auc_score(y[valid], oof[valid]):.4f}")

    # courbes
    import types
    a = types.SimpleNamespace(model="mednet", features=args.features, data=args.data, cv="stratified")
    dl.plot_curves(hists, a)
    out_json = dl.CACHE_DIR / f"mednet_results_{args.data}.json"
    out_json.write_text(json.dumps(
        dict(label=f"MedNet ResNet-18 + {args.features}", data=args.data,
             features=args.features, auc_mean=float(aucs.mean()), auc_std=float(aucs.std()),
             folds=[float(x) for x in aucs], freeze_until=args.freeze_until), indent=2))
    print(f"  Sauvé : {out_json}")


if __name__ == "__main__":
    main()
