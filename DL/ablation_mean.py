"""
ablation_mean.py — Le CNN3D apprend-il de vraies features, ou court-circuite-t-il
via la dose moyenne scalaire des features cliniques ?
=================================================================================
Question (collègue) : retirer `mean` du clinique. Si le VOLUME 3D récupère le
signal de dose → le CNN apprend ; sinon il se repose sur le raccourci clinique.

On compare (CV 5-fold stratifiée, mêmes splits) :
  Tabulaire (sans volume) :   mean_only | clin_sans_mean | clin_avec_mean
  CNN3D (avec volume)     :   dose_seule | clin_sans_mean+vol | clin_avec_mean+vol
"""
import numpy as np
from types import SimpleNamespace
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import dl_cardiac as dl

T = dl.TARGET
SEED = dl.RANDOM_STATE
SETS = {
    "mean_only":      ["mean"],
    "clin_sans_mean": ["Year_date_diag", "age_diag", "iccc_type", "anthra_1K"],
    "clin_avec_mean": ["mean", "Year_date_diag", "age_diag", "iccc_type", "anthra_1K"],
}


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def tabular_oof(df, y, splits, feats, clf_factory):
    oof = np.full(len(df), np.nan)
    for tr, te in splits:
        prep = dl.ClinicalPrep(feats)
        Xtr = prep.fit_transform(df.iloc[tr]); Xte = prep.transform(df.iloc[te])
        clf = clf_factory(); clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(y, oof)


def main():
    device = get_device()
    torch.manual_seed(SEED); np.random.seed(SEED)
    df, grids = dl.build_dataset("ants", None)   # ants : pour comparer au 0,756 connu
    y = df[T].to_numpy()
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, y))

    print("\n=== TABULAIRE (clinique seul, SANS volume 3D) — AUC OOF ===")
    tab = {}
    for name, feats in SETS.items():
        lr = tabular_oof(df, y, splits, feats, lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
        rf = tabular_oof(df, y, splits, feats, lambda: RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=SEED, n_jobs=-1))
        tab[name] = (lr, rf)
        print(f"  {name:16s} : LogReg={lr:.3f}  RandomForest={rf:.3f}")

    print("\n=== CNN3D (AVEC volume 3D) — AUC OOF par fold ===")
    cnn = {}
    for feat_key, label in [("none", "dose_seule (volume seul)"),
                            ("clin_nomean", "clin_sans_mean + volume"),
                            ("clinical", "clin_avec_mean + volume")]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        args = SimpleNamespace(model="cnn3d", features=feat_key, data="ants", cv="stratified",
                               loss="bce", folds=5, epochs=35, patience=8, batch=64,
                               lr=5e-4, dropout=0.4, limit=None, device=None)
        auc, _ = dl.run_cv(args, df, grids, device)
        cnn[feat_key] = float(auc.mean())
        print(f"  >>> {label:28s} : {auc.mean():.3f} ± {auc.std():.3f}")

    # ---------- synthèse + interprétation ----------
    print("\n" + "=" * 70)
    print("  SYNTHÈSE ABLATION DE LA DOSE MOYENNE")
    print("=" * 70)
    A = tab["clin_avec_mean"][1]   # RF clin avec mean (~baseline)
    B = tab["clin_sans_mean"][1]   # RF clin sans mean
    M = tab["mean_only"][1]        # mean seul
    C = cnn["clin_nomean"]         # CNN volume + clin sans mean
    E = cnn["none"]                # CNN volume seul
    print(f"  [A] clinique AVEC mean (RF, sans volume)      : {A:.3f}")
    print(f"  [B] clinique SANS mean (RF, sans volume)      : {B:.3f}   (chute due au retrait de mean : {A-B:+.3f})")
    print(f"  [M] mean SEUL (1 scalaire)                    : {M:.3f}")
    print(f"  [E] volume 3D SEUL (CNN, 0 clinique)          : {E:.3f}")
    print(f"  [C] volume 3D + clinique SANS mean (CNN)      : {C:.3f}")
    print()
    print(f"  Le volume récupère-t-il la dose moyenne retirée ?  C - B = {C-B:+.3f}")
    if C >= A - 0.01:
        print("  => OUI : volume+clin_sans_mean ≈ clin_AVEC_mean → le CNN EXTRAIT le signal de dose du volume (vraies features).")
    elif C - B > 0.02:
        print("  => PARTIELLEMENT : le volume récupère une partie du signal de mean, mais pas tout.")
    else:
        print("  => NON : le volume n'ajoute ~rien même sans mean → la branche 3D est quasi du poids mort, le modèle s'appuie sur le clinique.")
    print(f"  (Rappel : volume seul {E:.3f} vs mean seul {M:.3f} → le volume {'≥' if E>=M else '<'} le scalaire mean.)")


if __name__ == "__main__":
    main()
