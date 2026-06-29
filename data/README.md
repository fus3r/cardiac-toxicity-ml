# Data

## Contents

| File / folder | Description |
|---|---|
| `RT_Thorax_v1.csv` | Main analysis table: 1,375 patients, 23 variables. Comma separated. |
| `coeur_1k/` | Voxelized 3D dose matrices, one CSV file per patient (`dosi_coeur_<ctr>_<numcent>.csv`). Useful columns: `x, y, z, ID2013A` (coordinates plus dose). Tab separated. |
| `coeur_1k_petit/` | Sample of 5 patients, handy to develop and test without loading the full set. |

The detailed variable dictionary is in `../references/dictionnaire_donnees.md`.

## Confidentiality

These data come from the FCCSS cohort and are **strictly confidential**: use is limited to the academic scope of the project, in accordance with the CNIL authorizations. They must not be shared or published.

In particular, the `numcent` field is a patient identifier: it must never be shared, even pseudonymized.

No data file is versioned (see `.gitignore`). The files above must be obtained from the supervisor and placed in this folder.

## Clinical to 3D dosimetry join

The 3D dose matrices link to the clinical table by the pair `ctr` + `numcent`, which appears in the names of the `coeur_1k/` files.
