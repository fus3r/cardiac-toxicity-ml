"""
regen_corrected_parallel.py — Régénère le rescaling CORRIGÉ (recalage sur MASQUE)
=================================================================================
Pipeline corrigé (cf. diagnostic vérifié) : pour chaque patient,
  1. antspy_rescaling(df_atlas, df_moving, register_on="mask")  -> recalage
     ANATOMIQUE (Dice 0,62 -> 0,92) au lieu de piloté par la dose ;
  2. project_doses_to_reference(ref_atlas_grid, warped)         -> grille commune.
Sortie : rescaling/mask_csv_projected/{ctr}_{numcent}_projected.csv
(même format que ants_csv_projected -> directement utilisable par le DL).

Parallélisé 12 cœurs (chaque patient est indépendant ; ITK forcé à 1 thread/worker
pour éviter la sur-souscription). Reprenable (saute les fichiers déjà produits).

Lancer :  python -u rescaling/regen_corrected_parallel.py --n_jobs 12
"""
import os
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import re
import sys
import glob
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import paths as P

HERE = str(P.RESCALING_DIR)          # conservé pour sys.path des workers
RAW_DIR = str(P.COEUR_1K_DIR)
ATLAS_RAW = str(P.ATLAS_RAW)
REF_GRID = str(P.REF_GRID)
OUT_DIR = str(P.MASK_PROJECTED_DIR)


def robust_load(fp):
    """Lecture tolérante (tab puis espaces) + dédoublonnage (x,y,z)->mean."""
    df = pd.read_csv(fp, sep="\t")
    if "ID2013A" not in df.columns:
        df = pd.read_csv(fp, sep=r"\s+", engine="python")
    df.columns = df.columns.str.strip()
    req = ["x", "y", "z", "ID2013A"]
    df[req] = df[req].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=req)
    return df.groupby(["x", "y", "z"], as_index=False)["ID2013A"].mean()


# chargés une fois par worker (globals au niveau module)
_ATLAS = None
_REF = None
def _get_refs():
    global _ATLAS, _REF
    if _ATLAS is None:
        _ATLAS = robust_load(ATLAS_RAW)
        _REF = robust_load(REF_GRID)[["x", "y", "z"]]
    return _ATLAS, _REF


def process(fp):
    m = re.match(r"^dosi_coeur_(\d+)_(\d+)\.csv$", os.path.basename(fp))
    if not m:
        return ("skip-name", fp, 0.0)
    ctr, num = m.group(1), m.group(2)
    out = os.path.join(OUT_DIR, f"{ctr}_{num}_projected.csv")
    if os.path.exists(out):
        return ("cached", f"{ctr}_{num}", 0.0)
    try:
        from antspy_rescaling import antspy_rescaling
        from final_rescaling import project_doses_to_reference
        atlas, ref = _get_refs()
        df_moving = robust_load(fp)
        t0 = time.time()
        warped = antspy_rescaling(atlas, df_moving, output_csv_path=None, register_on="mask")
        project_doses_to_reference(ref, warped, output_path=out)
        return ("ok", f"{ctr}_{num}", time.time() - t0)
    except Exception as e:
        return (f"ERR:{type(e).__name__}:{str(e)[:80]}", f"{ctr}_{num}", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="ne traiter que les N premiers (smoke test)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    sys.path.insert(0, HERE)  # pour importer antspy_rescaling/final_rescaling dans les workers

    assert os.path.exists(ATLAS_RAW), f"atlas introuvable: {ATLAS_RAW}"
    assert os.path.exists(REF_GRID), f"ref grid introuvable: {REF_GRID}"

    files = sorted(glob.glob(os.path.join(RAW_DIR, "dosi_coeur_*.csv")))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} fichiers à traiter | n_jobs={args.n_jobs} | sortie={OUT_DIR}", flush=True)

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(process)(fp) for fp in files
    )
    dt = time.time() - t0

    ok = [r for r in results if r[0] == "ok"]
    cached = [r for r in results if r[0] == "cached"]
    errs = [r for r in results if r[0].startswith("ERR")]
    times = [r[2] for r in ok if r[2] > 0]
    print(f"\n=== TERMINÉ en {dt/60:.1f} min ===", flush=True)
    print(f"  ok={len(ok)}  cached={len(cached)}  erreurs={len(errs)}", flush=True)
    if times:
        print(f"  temps/patient: médiane={np.median(times):.1f}s moyenne={np.mean(times):.1f}s", flush=True)
    for r in errs[:15]:
        print(f"  [ERR] {r[1]} : {r[0]}", flush=True)
    print(f"  fichiers produits dans {OUT_DIR}: {len(glob.glob(os.path.join(OUT_DIR,'*_projected.csv')))}", flush=True)


if __name__ == "__main__":
    main()
