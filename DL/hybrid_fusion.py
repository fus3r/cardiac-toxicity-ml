"""
hybrid_fusion.py — Fusion (embedding 3D du CNN) + clinique -> classifieur tabulaire
===================================================================================
Question scientifique : la GÉOMÉTRIE 3D de la dose (embedding appris par un CNN3D)
ajoute-t-elle un pouvoir prédictif AU-DELÀ du clinique (qui contient déjà le
scalaire `mean`, dose cardiaque moyenne) ?

Sur petit échantillon, un classifieur tabulaire sur l'embedding d'un CNN bat
souvent le CNN bout-en-bout (découplage représentation / décision).

Protocole OOF SANS FUITE :
  CV externe 5-fold stratifiée (mêmes splits que la matrice) ; pour chaque fold :
    1. CNN3D dose-seule entraîné UNIQUEMENT sur outer-train (early stopping sur
       split val interne, restauration best-weight) ;
    2. embedding 64-d extrait pour outer-train et outer-test ;
    3. classifieurs (LogReg / RandomForest / HistGBDT) entraînés sur outer-train,
       prédits sur outer-test (les embeddings du test viennent d'un CNN qui ne
       les a jamais vus).
  -> AUC OOF poolée + AUC par fold (moyenne ± std) pour : clinique | géométrie |
     clinique+géométrie ; + CNN bout-en-bout (référence).
  -> test bootstrap apparié de ΔAUC = AUC(clinique+géom) − AUC(clinique).
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

import dl_cardiac as dl

TARGET = dl.TARGET
SEED = dl.RANDOM_STATE
CLIN = dl.FEATURE_SETS["clinical"]   # mean, Year_date_diag, age_diag, iccc_type, anthra_1K


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def make_loader(grids, rows, df, dose_scale, shuffle, batch=64):
    sub = df.iloc[rows]
    ds = dl.CardiacDataset(grids[sub["grid_pos"].to_numpy()],
                           np.empty((len(rows), 0), np.float32),
                           sub[TARGET].to_numpy(), dose_scale, flatten_mask=None)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def embed_rows(model, grids, rows, df, dose_scale, device):
    loader = make_loader(grids, rows, df, dose_scale, shuffle=False)
    out = []
    model.eval()
    for b in loader:
        out.append(model.embed(b["voxels"].to(device)).cpu().numpy())
    return np.concatenate(out)


def cnn_prob_rows(model, grids, rows, df, dose_scale, device):
    loader = make_loader(grids, rows, df, dose_scale, shuffle=False)
    out = []
    model.eval()
    c0 = torch.empty((0, 0))
    with torch.no_grad():
        for b in loader:
            v = b["voxels"].to(device)
            c = torch.empty((v.size(0), 0), device=device)
            out.append(torch.sigmoid(model(v, c)).cpu().numpy())
    return np.concatenate(out)


def train_cnn(grids, tr_rows, df, dose_scale, device, epochs=25, patience=6):
    """CNN3D dose-seule sur tr_rows, avec split val interne + early stopping."""
    rng = np.random.RandomState(SEED)
    tr_rows = rng.permutation(tr_rows)
    n_val = max(1, int(0.15 * len(tr_rows)))
    val_rows, fit_rows = tr_rows[:n_val], tr_rows[n_val:]

    train_loader = make_loader(grids, fit_rows, df, dose_scale, shuffle=True)
    val_loader = make_loader(grids, val_rows, df, dose_scale, shuffle=False)

    model = dl.CNN3DLogit(n_clinical=0, dropout=0.4).to(device)
    n_pos = int(df.iloc[fit_rows][TARGET].sum()); n_neg = len(fit_rows) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    dl.train_one_fold(model, train_loader, val_loader, crit, device,
                      max_epochs=epochs, lr=5e-4, patience=patience)
    return model


def fit_predict(clf_factory, X_tr, y_tr, X_te):
    clf = clf_factory()
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


CLASSIFIERS = {
    "LogReg": lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5),
    "RF":     lambda: RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                             random_state=SEED, n_jobs=-1),
    "HGB":    lambda: HistGradientBoostingClassifier(class_weight="balanced",
                                                     random_state=SEED, max_iter=300,
                                                     learning_rate=0.05),
}


def bootstrap_delta(y, p_base, p_new, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y)); diffs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        ys = y[s]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        diffs.append(roc_auc_score(ys, p_new[s]) - roc_auc_score(ys, p_base[s]))
    diffs = np.array(diffs)
    p_one_sided = float((diffs <= 0).mean())  # H0: pas d'amélioration
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), p_one_sided


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"device={device}")

    df, grids = dl.build_dataset("ants", args.limit)
    y = df[TARGET].to_numpy()
    dose_scale = float(np.percentile(grids[grids > 0], 99))
    print(f"dose_scale={dose_scale:.2f}")

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, y))

    variants = ["clin", "geom", "clin+geom"]
    # OOF probs : par classifieur -> par variante -> array(n)
    oof = {clf: {v: np.full(len(df), np.nan) for v in variants} for clf in CLASSIFIERS}
    oof_cnn = np.full(len(df), np.nan)          # CNN bout-en-bout (référence)
    per_fold = {clf: {v: [] for v in variants} for clf in CLASSIFIERS}

    for f, (tr, te) in enumerate(splits):
        print(f"\n=== Fold {f+1}/5 — train={len(tr)} test={len(te)} ===")
        model = train_cnn(grids, tr, df, dose_scale, device, epochs=args.epochs)

        E_tr = embed_rows(model, grids, tr, df, dose_scale, device)
        E_te = embed_rows(model, grids, te, df, dose_scale, device)
        oof_cnn[te] = cnn_prob_rows(model, grids, te, df, dose_scale, device)

        prep = dl.ClinicalPrep(CLIN)
        C_tr = prep.fit_transform(df.iloc[tr]); C_te = prep.transform(df.iloc[te])

        es = StandardScaler().fit(E_tr)
        Es_tr, Es_te = es.transform(E_tr), es.transform(E_te)

        feats = {
            "clin":      (C_tr, C_te),
            "geom":      (Es_tr, Es_te),
            "clin+geom": (np.hstack([C_tr, Es_tr]), np.hstack([C_te, Es_te])),
        }
        y_tr, y_te = y[tr], y[te]
        for clf in CLASSIFIERS:
            for v in variants:
                Xtr, Xte = feats[v]
                p = fit_predict(CLASSIFIERS[clf], Xtr, y_tr, Xte)
                oof[clf][v][te] = p
                per_fold[clf][v].append(roc_auc_score(y_te, p))
        print(f"  CNN e2e AUC(test)={roc_auc_score(y_te, oof_cnn[te]):.3f} | "
              + " ".join(f"{clf}:clin={per_fold[clf]['clin'][-1]:.3f},"
                         f"c+g={per_fold[clf]['clin+geom'][-1]:.3f}" for clf in CLASSIFIERS))

    # ---------- résumé ----------
    print("\n" + "=" * 78)
    print("  FUSION — AUC OOF poolée (et AUC moyenne ± std par fold)")
    print("=" * 78)
    print(f"  {'Classifieur':<10}{'clinique':<22}{'géométrie(3D)':<22}{'clinique+géométrie':<22}")
    results = {}
    for clf in CLASSIFIERS:
        row = {}
        cells = []
        for v in variants:
            pooled = roc_auc_score(y, oof[clf][v])
            m, s = np.mean(per_fold[clf][v]), np.std(per_fold[clf][v])
            row[v] = dict(pooled=float(pooled), mean=float(m), std=float(s))
            cells.append(f"{pooled:.3f} ({m:.3f}±{s:.3f})")
        results[clf] = row
        print(f"  {clf:<10}{cells[0]:<22}{cells[1]:<22}{cells[2]:<22}")

    auc_cnn = roc_auc_score(y, oof_cnn)
    print(f"\n  Référence CNN3D dose-seule (bout-en-bout, OOF) : {auc_cnn:.3f}")
    print(f"  Référence RF baseline (clinique agrégé)        : 0.770 ± 0.047")

    # ---------- test de l'apport géométrique (meilleur classifieur) ----------
    best_clf = max(CLASSIFIERS, key=lambda c: roc_auc_score(y, oof[c]["clin+geom"]))
    d, lo, hi, p = bootstrap_delta(y, oof[best_clf]["clin"], oof[best_clf]["clin+geom"])
    print("\n" + "=" * 78)
    print(f"  APPORT DE LA GÉOMÉTRIE 3D ({best_clf}) — bootstrap apparié 2000x")
    print("=" * 78)
    print(f"  ΔAUC (clinique+géom − clinique) = {d:+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]")
    print(f"  p (unilatéral, H0: ΔAUC<=0) = {p:.4f}  -> "
          + ("géométrie ajoute (significatif)" if p < 0.05 else "non significatif"))

    # ---------- matrice de confusion du meilleur modèle (seuil 0.5 et prévalence) ----------
    p_best = oof[best_clf]["clin+geom"]
    for thr in (0.5, float(y.mean())):
        tn, fp, fn, tp = confusion_matrix(y, (p_best >= thr).astype(int)).ravel()
        sens = tp / (tp + fn); spec = tn / (tn + fp)
        print(f"  [{best_clf} clin+geom] seuil={thr:.2f} : Sens={sens:.3f} Spéc={spec:.3f}")

    out = dl.CACHE_DIR / "hybrid_results.json"
    out.write_text(json.dumps(dict(results=results, cnn_oof=float(auc_cnn),
                                   best_clf=best_clf, delta=dict(mean=d, lo=lo, hi=hi, p=p)),
                              indent=2, ensure_ascii=False))
    print(f"\n  Sauvé : {out}")


if __name__ == "__main__":
    main()
