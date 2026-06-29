"""
run_corrected.py — CNN3D + clinique sur le rescaling CORRIGÉ (mask), tagué JSON.
Lancé par l'orchestrateur overnight une fois la régénération terminée.
"""
import json
from types import SimpleNamespace
import numpy as np
import torch
import dl_cardiac as dl


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()
    torch.manual_seed(dl.RANDOM_STATE); np.random.seed(dl.RANDOM_STATE)
    df, grids = dl.build_dataset("mask", None)
    args = SimpleNamespace(model="cnn3d", features="clinical", data="mask", cv="stratified",
                           loss="bce", folds=5, epochs=40, patience=10, batch=64,
                           lr=5e-4, dropout=0.4, limit=None, device=None)
    auc, (sens, spec) = dl.run_cv(args, df, grids, device)
    (dl.CACHE_DIR / "mask_cnn.json").write_text(json.dumps(dict(
        label="CNN3D + clinique", data="mask", features="clinical",
        auc_mean=float(auc.mean()), auc_std=float(auc.std()),
        sens=float(sens), spec=float(spec)), indent=2, ensure_ascii=False))
    print(f"Sauvé : {dl.CACHE_DIR / 'mask_cnn.json'}")


if __name__ == "__main__":
    main()
