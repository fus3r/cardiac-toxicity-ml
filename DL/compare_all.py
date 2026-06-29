"""
compare_all.py — Tableau comparatif de toutes les approches DL + baseline
=========================================================================
Agrège les JSON de résultats produits par les différents scripts et imprime
un classement par AUC-ROC, pour décider du/des modèle(s) à défendre.
"""
import json
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "cache"
RF_BASELINE = ("Random Forest (clinique agrégé, baseline)", 0.770, 0.047, "clinique 5 var", "—")

rows = []  # (label, auc_mean, auc_std, entrée, données)


def add(label, m, s, entree, data):
    rows.append((label, m, s, entree, data))


# 1) Matrice (results_summary.json) — données ants
p = CACHE / "results_summary.json"
if p.exists():
    for r in json.loads(p.read_text()):
        add(r["label"], r["auc_mean"], r["auc_std"], r["features"], "ants")

# 2) Fusion (hybrid_results.json) — données ants
p = CACHE / "hybrid_results.json"
if p.exists():
    h = json.loads(p.read_text())
    bc = h.get("best_clf", "LogReg")
    res = h["results"][bc]
    add(f"Fusion {bc}: clinique", res["clin"]["pooled"], res["clin"]["std"], "clinique", "ants")
    add(f"Fusion {bc}: clinique+géométrie3D", res["clin+geom"]["pooled"], res["clin+geom"]["std"],
        "clinique+vol", "ants")

# 3) MedNet (mednet_results*.json) — peut y en avoir plusieurs variantes
for p in sorted(CACHE.glob("mednet_results*.json")):
    m = json.loads(p.read_text())
    data = m.get("data", "best" if "best" in p.name or p.name == "mednet_results.json" else "?")
    add(f"MedNet (ResNet-18 transfer, {m['features']})", m["auc_mean"], m["auc_std"],
        "clinique+vol", m.get("data", data))

# 4) Modèles sur rescaling corrigé (mask) — fichiers mask_*.json
for p in sorted(CACHE.glob("mask_*.json")):
    m = json.loads(p.read_text())
    add(m.get("label", p.stem), m["auc_mean"], m["auc_std"], m.get("features", "?"), "mask(corrigé)")

# baseline
add(*RF_BASELINE)

rows.sort(key=lambda r: r[1], reverse=True)

print("=" * 92)
print("  COMPARAISON DE TOUTES LES APPROCHES (tri par AUC-ROC décroissant)")
print("=" * 92)
print(f"  {'Approche':<46}{'AUC-ROC':<18}{'Entrée':<16}{'Données':<14}")
print("  " + "-" * 88)
for label, m, s, entree, data in rows:
    star = "  " if m >= 0.770 else ""
    print(f"  {label:<46}{m:.3f} ± {s:.3f}     {entree:<16}{data:<14}{star}")
print("\n   = au niveau ou au-dessus du baseline RF (0,770)")
best = max(rows, key=lambda r: r[1])
print(f"\n  >>> Meilleure approche : {best[0]} — AUC {best[1]:.3f} ± {best[2]:.3f} ({best[4]})")
