"""
run_all.py — Matrice d'expériences DL (charge le cache une seule fois)
======================================================================
Enchaîne les configs clés pour le rapport et imprime un tableau récapitulatif.
Lancer en non-bufferisé pour suivre la progression :  python -u DL/run_all.py
"""
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

import dl_cardiac as dl


def make_args(**kw):
    base = dict(model="cnn3d", features="clinical", data="ants", cv="stratified",
                loss="bce", folds=5, epochs=30, patience=6, batch=64,
                lr=5e-4, dropout=0.4, limit=None, device=None)
    base.update(kw)
    return SimpleNamespace(**base)


# (label, overrides)  — toutes sur 'ants'. Ordre = par priorité pour le rapport :
# les 2 résultats critiques (headline + test ctr) sortent en premier.
CONFIGS = [
    ("CNN3D + clinique  (CV stratifiée)",       dict(model="cnn3d", features="clinical", cv="stratified")),
    ("CNN3D + clinique  (CV par centre/ctr)",    dict(model="cnn3d", features="clinical", cv="group")),
    ("MLP   + dose seule (CV stratifiée)",       dict(model="mlp",   features="none",     cv="stratified")),
    ("CNN3D + dose seule (CV stratifiée)",       dict(model="cnn3d", features="none",     cv="stratified")),
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()
    torch.manual_seed(dl.RANDOM_STATE); np.random.seed(dl.RANDOM_STATE)
    print(f"device = {device}")

    # charge les données 'ants' une seule fois
    df, grids = dl.build_dataset("ants", None)

    results = []
    for label, ov in CONFIGS:
        args = make_args(**ov)
        print("\n" + "#" * 70)
        print(f"#  {label}")
        print("#" * 70)
        t0 = time.time()
        torch.manual_seed(dl.RANDOM_STATE); np.random.seed(dl.RANDOM_STATE)
        auc, (sens, spec) = dl.run_cv(args, df, grids, device)
        dt = time.time() - t0
        results.append(dict(label=label, model=args.model, features=args.features,
                            cv=args.cv, auc_mean=float(auc.mean()),
                            auc_std=float(auc.std()), sens=float(sens),
                            spec=float(spec), secs=round(dt)))
        print(f"  [{dt:.0f}s]")

    # ----- tableau récapitulatif -----
    print("\n" + "=" * 78)
    print("  RÉCAPITULATIF")
    print("=" * 78)
    print(f"  {'Config':<42}{'AUC-ROC':<18}{'Sens':<7}{'Spéc':<7}")
    print("  " + "-" * 74)
    for r in results:
        print(f"  {r['label']:<42}{r['auc_mean']:.3f} ± {r['auc_std']:.3f}     "
              f"{r['sens']:.3f}  {r['spec']:.3f}")

    out = dl.CACHE_DIR / "results_summary.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  Résultats JSON : {out}")


if __name__ == "__main__":
    main()
