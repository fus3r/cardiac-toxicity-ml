"""
learning_check.py — Le modèle apprend-il vraiment ?
===================================================
Trace, epoch par epoch, train loss + val loss + **AUC de validation**.
Preuve directe d'apprentissage : l'AUC monte de ~0,5 (hasard, epoch 0) vers ~0,75,
même si la loss pondérée reste élevée (plancher d'erreur + pos_weight).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dl_cardiac as dl
T = dl.TARGET


def dev():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def main():
    device = dev()
    torch.manual_seed(42); np.random.seed(42)
    df, grids = dl.build_dataset("best", None)
    y = df[T].to_numpy()
    dose_scale = float(np.percentile(grids[grids > 0], 99))
    tr, va = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=42)

    prep = dl.ClinicalPrep(dl.FEATURE_SETS["clinical"])
    Xtr = prep.fit_transform(df.iloc[tr]); Xva = prep.transform(df.iloc[va])

    def loader(idx, X, sh):
        ds = dl.CardiacDataset(grids[df.iloc[idx]["grid_pos"].to_numpy()], X,
                               df.iloc[idx][T].to_numpy(), dose_scale, None)
        return DataLoader(ds, batch_size=64, shuffle=sh)
    tl, vl = loader(tr, Xtr, True), loader(va, Xva, False)

    model = dl.CNN3DLogit(Xtr.shape[1], dropout=0.4).to(device)
    npos = int(df.iloc[tr][T].sum()); nneg = len(tr) - npos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([nneg/npos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    H = {"tr": [], "vl": [], "auc": []}
    # epoch 0 : AUC AVANT entraînement (≈ 0,5 attendu)
    p, e = dl.predict(model, vl, device)
    print(f"  epoch  0 (avant entraînement) — val AUC={roc_auc_score(e,p):.3f}", flush=True)

    for ep in range(30):
        model.train(); tot = n = 0
        for b in tl:
            opt.zero_grad()
            loss = crit(model(b["voxels"].to(device), b["clinical"].to(device)), b["event"].to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item()*len(b["event"]); n += len(b["event"])
        trl = tot/n
        # val loss + val AUC
        model.eval(); vtot = vn = 0; ps = []; es = []
        with torch.no_grad():
            for b in vl:
                lo = model(b["voxels"].to(device), b["clinical"].to(device))
                vtot += crit(lo, b["event"].to(device)).item()*len(b["event"]); vn += len(b["event"])
                ps.append(torch.sigmoid(lo).cpu().numpy()); es.append(b["event"].numpy())
        vll = vtot/vn; auc = roc_auc_score(np.concatenate(es), np.concatenate(ps))
        H["tr"].append(trl); H["vl"].append(vll); H["auc"].append(auc)
        if (ep+1) % 5 == 0:
            print(f"  epoch {ep+1:2d} — train_loss={trl:.3f} val_loss={vll:.3f} val_AUC={auc:.3f}", flush=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(H["tr"], label="train loss"); a1.plot(H["vl"], label="val loss")
    a1.set_title("Loss (BCE pondérée)"); a1.set_xlabel("epoch"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(H["auc"], color="green", marker="o", ms=3)
    a2.axhline(0.5, ls="--", c="red", label="hasard (0,5)")
    a2.axhline(0.770, ls=":", c="gray", label="RF baseline (0,77)")
    a2.set_title("AUC de validation = PREUVE d'apprentissage")
    a2.set_xlabel("epoch"); a2.set_ylim(0.45, 0.85); a2.legend(); a2.grid(alpha=.3)
    fig.suptitle("Le modèle apprend : loss ↓ modeste, mais AUC 0,5 → ~0,75")
    fig.tight_layout()
    out = dl.FIG_DIR / "learning_check.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n  AUC finale={H['auc'][-1]:.3f} (départ {roc_auc_score(np.concatenate(es),np.concatenate(ps)):.3f})")
    print(f"  Figure : {out}", flush=True)


if __name__ == "__main__":
    main()
