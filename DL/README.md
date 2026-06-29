# Deep learning

Deep models predicting severe cardiac disease from the registered 3D dose matrices and clinical variables, with production of activation maps.

## Input

The projected CSV files (the registration output, folders `rescaling/*_projected/`) are point clouds whose ordering is not guaranteed to match across patients. They are therefore converted back into a **dense 47×47×32 grid** by deterministic indexing of the coordinates `(x, y, z) → (i, j, k)`, with duplicates aggregated by mean.

Available data variants (`--data`): `ants`, `original`, `best`, `mask`.

## Main script

`dl_cardiac.py` trains an MLP or a 3D CNN under cross validation and plots the learning curves.

```bash
python DL/dl_cardiac.py --model cnn3d --features clinical --data ants
python DL/dl_cardiac.py --model mlp   --features none     --data ants
python DL/dl_cardiac.py --model cnn3d --features clinical --cv group
python DL/dl_cardiac.py --limit 200 --epochs 10            # quick test
```

Clinical feature sets (`--features`): `none`, `clinical` (`mean`, `Year_date_diag`, `age_diag`, `iccc_type`, `anthra_1K`), `clin_nomean`, `mean_only`, `clinical_plus`.

## Other scripts

| Script | Role |
|---|---|
| `dl_mednet.py` | Pretrained 3D CNN variant (MedicalNet). |
| `activation_map.py`, `render_activation_fig.py` | Compute and render the activation maps (Grad-CAM). |
| `ablation_mean.py`, `compare_all.py`, `fold_analysis.py` | Ablation studies and comparisons. |
| `hybrid_fusion.py`, `learning_check.py` | Clinical plus image fusion, learning diagnostics. |
| `run_all.py`, `run_corrected.py` | Orchestration scripts. |

Intermediate outputs (`cache/`, `figures/`) are regenerable and not versioned.

## Design choices

Compared to the earliest versions, the current pipeline notably fixes:

- the reconstruction of a **dense grid that is consistent across patients** (instead of a voxel vector whose ordering varied);
- a **one-hot encoding aligned** between training and test;
- an **early stopping on an internal split** (not on the test fold), to avoid any leak;
- paths **auto-detected** from the repository root (no more cluster-specific paths).

## Limitation

MedicalNet (`dl_mednet.py`) requires external pretrained weights, which are not provided in this repository.
