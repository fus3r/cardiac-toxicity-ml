# Projet P15 — commandes utilitaires
# Utilisation : make <cible>

PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup notebooks dl-demo clean

help:
	@echo "Cibles disponibles :"
	@echo "  setup      Cree l'environnement .venv et installe les dependances"
	@echo "  notebooks  Lance JupyterLab sur les notebooks d'analyse"
	@echo "  dl-demo    Entrainement DL rapide de demonstration (echantillon reduit)"
	@echo "  clean      Supprime les caches Python et fichiers systeme"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

notebooks:
	$(PYTHON) -m jupyterlab EDA/notebooks

dl-demo:
	$(PYTHON) DL/dl_cardiac.py --model cnn3d --features clinical --data ants --limit 200 --epochs 5

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
	find . -name '.DS_Store' -delete
