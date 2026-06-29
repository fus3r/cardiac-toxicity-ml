# 3D dose registration

## Goal

Each patient is described by a 3D point cloud of the heart, where each point carries a received dose. Hearts differ in size, shape and position. To compare doses across patients, every heart is brought into the space of a reference patient, the **atlas** (chosen as the median patient by voxel count; patient identifier removed for confidentiality).

The process has two stages: a **registration** (deformation of the patient heart toward the atlas) then a **projection** of the doses onto the atlas grid, so that all patients share the same spatial support.

## Two methods compared

1. **Original method**: outer envelope, affine registration, deformable CPD registration, then propagation to the full heart by Thin Plate Spline (TPS).
2. **ANTsPy method**: voxelization into a 3D image then registration with ANTs.

For each patient, the projection that best preserves the dose distribution (`full_dose_score`) is kept.

## Main scripts

| Script | Role |
|---|---|
| `paths.py` | Centralized paths (data, atlas, output folders), anchored on the repository and independent of the current directory. |
| `fonctions_annexes.py` | Loading and cleaning of patient CSV files (`load_and_clean_csv`). |
| `original_rescaling.py` | CPD plus TPS registration. |
| `antspy_rescaling.py` | ANTsPy registration. |
| `final_rescaling.py` | Projection of the doses onto the atlas grid. |
| `metrique_rescaling.py` | Geometric and dose metrics, combined score. |
| `visualisation.py` | 3D visualizations (Plotly). |
| `regen_corrected_parallel.py` | Parallel regeneration of the corrected projections (mask driven registration). |

## Output folders (regenerable, not versioned)

| Folder | Content |
|---|---|
| `original_csv_projected/` | Projections by the original method. |
| `ants_csv_projected/` | Projections by ANTsPy. |
| `ants_csv_transformed/` | Patients transformed by ANTs before projection (useful to check the registration). |
| `best_score_csv/` | Best of the two projections per patient, the reference result. |
| `mask_csv_projected/` | Variant of the anatomy driven registration (see below). |

## Points to watch

- **Anatomy driven registration.** The historical version registered on the dose image, which could degrade the geometry. The recommended variant registers on the **mask** (the heart shape) then transports the dose, which improves the alignment markedly. A full recompute on this basis is advised.
- **Patients with a null dose.** Two patients (identifiers removed for confidentiality) have an entirely null source dose; the dose metrics are not defined for them and they must be handled separately.
- **Projection can hide a bad registration**: also check `ants_csv_transformed/`, not only the projected CSV files.
