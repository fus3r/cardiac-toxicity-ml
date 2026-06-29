# Exploratory analysis and modelling

Analysis notebooks, to run from the repository root (`jupyter lab EDA/notebooks`).

## Notebooks

| Notebook | Content |
|---|---|
| `01_EDA.ipynb` | Exploratory analysis: distributions, missing values, temporal trends, correlations, first signals. |
| `02_Classification.ipynb` | Supervised modelling (logistic regression, Random Forest, Gradient Boosting, SVM). Feature selection protocol, cross validation, ROC and PR curves, importances. |
| `03_Survival.ipynb` | Survival analysis: stratified Kaplan Meier and a Cox proportional hazards model (forest plot of the hazard ratios). |
| `04_3D_Analysis.ipynb` | First exploration of the 3D dose matrices (visualizations, dose volume histograms, slices). |

## Figures

Figures produced by the notebooks are written to `EDA/figures/` (not versioned). A selection is copied into `reports/figures/`.

## Key points

- Retained predictors: `mean`, `Year_date_diag`, `age_diag`, `iccc_type`, `anthra_1K`.
- `gender` dropped after analysis (not significant).
- Class imbalance handled by weighting, without SMOTE.
- Random Forest: ROC-AUC ≈ 0.77; Cox PH: C-index ≈ 0.74.
