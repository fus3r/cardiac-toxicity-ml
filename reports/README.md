# Key figures

A selection of representative figures, copied from the notebook and script outputs for quick viewing without re-running. The full figures are regenerated in `EDA/figures/` and `DL/figures/`.

## Exploratory analysis
- `figures/eda_distribution_cible.png`: distribution of the target variable.
- `figures/eda_boxplots_variables.png`: distributions of the continuous variables by status (with p-values).
- `figures/eda_matrice_correlation.png`: correlation matrix of the variables.
- `figures/eda_analyse_dosimetrique.png`: analysis of the dose volume indicators.
- `figures/eda_tendances_temporelles.png`: trends by diagnosis period.

## Classification
- `figures/classification_roc_pr.png`: ROC and precision recall curves with uncertainty (5-fold).
- `figures/classification_importance_variables.png`: variable importance (Random Forest).

## Survival
- `figures/survie_kaplan_meier.png`: stratified Kaplan Meier curves.
- `figures/survie_cox_forest_plot.png`: hazard ratios of the Cox model (forest plot).

## 3D dose matrices
- `figures/dose3d_histogramme_dvh.png`: dose volume histogram.
- `figures/dose3d_visualisation.png`: 3D visualization of the dose to the heart.

## Deep learning
- `figures/dl_courbe_apprentissage.png`: learning curves (3D CNN).
- `figures/dl_carte_activation_mask.png`: risk activation map (3D Grad-CAM, 3D CNN) on the corrected anatomical registration (`mask`).
- `figures/dl_carte_activation_best.png`: same on the best method per patient (`best`).
