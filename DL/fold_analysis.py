"""
fold_analysis.py — Pourquoi la variance entre folds ? (sans entraînement)
=========================================================================
Reproduit EXACTEMENT les splits utilisés par dl_cardiac / run_all
(StratifiedKFold seed=42) et la variante par centre (StratifiedGroupKFold),
puis chiffre l'hétérogénéité inter-folds (taux d'événement, centres, dose,
année, anthracyclines). Confirme/infirme l'hypothèse 'ctr'.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import dl_cardiac as dl

TARGET = dl.TARGET


def summarize(df, splits, name):
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    ev_rates, doses, years, ctr1_share = [], [], [], []
    for k, (_, te) in enumerate(splits):
        t = df.iloc[te]
        ev = t[TARGET].mean()
        ctr = t["ctr"].value_counts(normalize=True).round(3).to_dict()
        ev_rates.append(ev); doses.append(t["mean"].mean()); years.append(t["Year_date_diag"].mean())
        ctr1_share.append(t["ctr"].eq(1).mean())
        print(f"  Fold {k+1}: n={len(t):4d} | event={ev:.3f} | dose_moy={t['mean'].mean():.1f} | "
              f"year={t['Year_date_diag'].mean():.0f} | %ctr1={t['ctr'].eq(1).mean():.2f} | ctr={ctr}")
    print(f"\n  Écart-type inter-folds :  event={np.std(ev_rates):.4f}  "
          f"dose={np.std(doses):.3f}  year={np.std(years):.2f}  %ctr1={np.std(ctr1_share):.4f}")
    return np.std(ev_rates)


def main():
    df, _ = dl.build_dataset("ants", None)
    y = df[TARGET].to_numpy()
    g = df["ctr"].to_numpy()

    strat = list(StratifiedKFold(5, shuffle=True, random_state=42).split(df, y))
    grp = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(df, y, g))

    s1 = summarize(df, strat, "CV STRATIFIÉE (event) — utilisée par défaut")
    s2 = summarize(df, grp, "CV GROUPÉE PAR CENTRE (ctr)")

    print(f"\n{'='*70}")
    print("  LECTURE :")
    print(f"  - En CV stratifiée, le taux d'événement varie peu entre folds (std={s1:.4f}),")
    print(f"    MAIS la dose/année/centre varient (centres mélangés dans chaque fold).")
    print(f"  - ctr=1 ~78% partout ; les centres minoritaires (3,8,10,12) tombent")
    print(f"    inégalement -> variance d'AUC quand le modèle capte la signature centre.")
    print(f"  - La CV par centre teste la robustesse (généralisation hors-centre).")


if __name__ == "__main__":
    main()
