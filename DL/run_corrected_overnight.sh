#!/bin/bash
# Orchestrateur overnight : attend la fin de la régénération du rescaling corrigé
# (mask_csv_projected), puis lance le DL dessus + comparatif final.
# Lancer : nohup bash DL/run_corrected_overnight.sh &
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null
LOG=DL/cache/overnight.log
MASK=rescaling/mask_csv_projected
echo "=== $(date '+%H:%M') : orchestrateur démarré, attente régénération ===" >> "$LOG"

# 1) Attendre que la régénération soit finie (>=1370 fichiers OU process terminé)
while [ "$(ls $MASK/*_projected.csv 2>/dev/null | wc -l | tr -d ' ')" -lt 1370 ] \
      && pgrep -f regen_corrected_parallel >/dev/null 2>&1; do
  sleep 60
done
N=$(ls $MASK/*_projected.csv 2>/dev/null | wc -l | tr -d ' ')
echo "=== $(date '+%H:%M') : régénération terminée ($N fichiers). Lancement DL corrigé ===" >> "$LOG"

# 2) cache frais (données corrigées) + DL
rm -f DL/cache/grids_mask.npz
PYTHONPATH=DL python -u DL/run_corrected.py >> "$LOG" 2>&1
echo "=== $(date '+%H:%M') : CNN3D mask fait ===" >> "$LOG"
PYTHONPATH=DL python -u DL/dl_mednet.py --data mask --features clinical \
   --freeze_until layer3 --epochs 30 --patience 8 --batch 24 >> "$LOG" 2>&1
echo "=== $(date '+%H:%M') : MedNet mask fait ===" >> "$LOG"
PYTHONPATH=DL python -u DL/activation_map.py --data mask --epochs 30 >> "$LOG" 2>&1
echo "=== $(date '+%H:%M') : carte activation mask faite ===" >> "$LOG"

# 3) comparatif final (toutes approches, incl. corrigé)
PYTHONPATH=DL python DL/compare_all.py > DL/cache/compare_final.txt 2>&1
echo "=== $(date '+%H:%M') : OVERNIGHT TERMINÉ — voir DL/cache/compare_final.txt ===" >> "$LOG"
