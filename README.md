# Cardiac Toxicity Prediction after Thoracic Radiotherapy

Project P15, CentraleSupélec, in partnership with INSERM and Gustave Roussy (FCCSS cohort, the French Childhood Cancer Survivors Study).

## Context

Some survivors of childhood cancers treated with thoracic radiotherapy develop severe cardiac disease decades later. This project predicts those events (grade ≥ 3 CTCAE) from clinical, treatment and dosimetric variables, then uses the 3D radiation dose delivered to the heart to build risk activation maps.

The study cohort has **1,375 patients**, with **236 severe cardiac events (17.2%)**. Median follow-up is about **32 years**, for diagnoses made between **1950 and 2000**, so the dose maps are reconstructed rather than read from a modern scanner.

> The patient data is confidential (FCCSS cohort, strictly academic use) and is not versioned. This repository holds code only. See `LICENSE` and `data/README.md`.

## Results

| Model | Metric |
|---|---|
| Random Forest (multivariate, 5-fold stratified) | ROC-AUC **0.77 ± 0.04** |
| Cox proportional hazards | C-index **0.74** |

The dominant predictor is the mean heart dose, alongside year of diagnosis, age at diagnosis, cancer type and cumulative anthracyclines. Sex was dropped after analysis (not significant, HR around 1.02, p = 0.88). Class imbalance is handled by weighting (`class_weight` and `sample_weight`), without SMOTE oversampling.

On the 3D side, five deep learning models (a 3D CNN, a learned geometry fusion, and a MedicalNet ResNet-18 adapted by transfer learning) all reach the same ceiling near 0.77: where the dose lands adds nothing measurable beyond how much (ΔAUC +0.000, p = 0.47). The payoff is a Grad-CAM map of where in the heart the risk concentrates. The result reproduces the published FCCSS finding with a leaner model, after closing a train/test leak in earlier code and showing that splitting cross validation by treatment centre doubles the variance (one centre supplies 78% of the cohort).

A selection of figures is in `reports/figures/`.

## Data

| Item | Description |
|---|---|
| `data/RT_Thorax_v1.csv` | Analysis table: 1,375 patients, 23 variables. Target: `Pathologie_cardiaque`. |
| `data/coeur_1k/` | Voxelized 3D dose matrices (one point cloud per patient, about 1,000 patients). |
| `data/coeur_1k_petit/` | Sample of 5 patients for quick tests. |
| `references/dictionnaire_donnees.md` | Detailed dictionary of the 23 variables. |

These data folders are gitignored and are not part of the repository.

## Repository layout

```
.
├── README.md  LICENSE  Makefile  requirements.txt  environment.yml
├── data/                 dataset and sample (not versioned)
├── references/           data dictionary
├── EDA/
│   ├── notebooks/        01_EDA, 02_Classification, 03_Survival, 04_3D_Analysis
│   └── figures/
├── rescaling/            3D dose registration to a common atlas
├── DL/                   deep learning: MLP, 3D CNN, MedicalNet, activation maps
└── reports/figures/      selected key figures
```

Each pillar (`EDA/`, `rescaling/`, `DL/`) has its own `README.md`.

## Install

Python 3.13.

```bash
make setup                       # creates .venv/ and installs requirements.txt
```

or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

An equivalent conda recipe is provided (`environment.yml`).

## Usage

**1. Exploratory analysis and classical modelling.** Notebooks in `EDA/notebooks/` cover exploration, classification, survival analysis and a first 3D analysis.

```bash
make notebooks                   # or: jupyter lab EDA/notebooks
```

**2. 3D dose registration.** The `rescaling/` folder brings every heart into the space of a reference patient (the atlas), compares two registration methods (CPD with TPS, and ANTsPy), then keeps for each patient the projection that best preserves the dose. Details in `rescaling/README.md`.

**3. Deep learning.** The `DL/` folder trains models (MLP, 3D CNN, MedicalNet) on the 3D dose grids and the clinical variables, and produces Grad-CAM activation maps.

```bash
python DL/dl_cardiac.py --model cnn3d --features clinical --data ants
```

Details in `DL/README.md`.

## Methods

**Classification:** logistic regression, Random Forest and Gradient Boosting (multivariate), with an SVM in the univariate screen. Stratified 5-fold cross validation, and nested 5×3 cross validation for hyperparameter selection. **Survival:** Kaplan Meier and a Cox proportional hazards model. **3D:** registration to an atlas, extraction on a common 47×47×32 grid, deep models and activation maps.

## Known limitations

- Several scripts assume the folder layout above (relative paths, or auto-detection of the repository root).
- The historical ANTs registration was driven by the dose. A variant driven by the anatomy (the heart mask) is available, and a recompute is recommended (see `rescaling/README.md`).
- Two patients have an entirely null dose and are handled separately.
- MedicalNet needs external pretrained weights, which are not provided.

## Next step

The same pipeline extends next semester to the CANTO-RT cohort (3,914 women treated for breast cancer, 409 cardiac events), where the heart sits inside the radiation field rather than outside it, so the spatial question that came back flat here gets a fair test. Segmented organ data (labelled anatomy) arrives this autumn.

## Team and supervision

Riad Darwish, Ayoub Chikri, Thibault Cohen, Hugo Likaj and Carl Kairouz (CentraleSupélec, 2025 to 2026). Supervision: Rodrigue Allodji (INSERM and Gustave Roussy).

## References

- Chounta S., Allodji R., et al. "Dosiomics-Based Prediction of Radiation-Induced Valvulopathy after Childhood Cancer", *Cancers*, 2023. DOI: 10.3390/cancers15123107
- Bentriou M., Letort V., et al. "Combining dosiomics and machine learning methods for predicting severe cardiac diseases in childhood cancer survivors: the FCCSS", *Frontiers in Oncology*, 2024. DOI: 10.3389/fonc.2024.1241221
