# Analyse VEC Fabry

Ce projet Python analyse les données VEC (Volume Extra Cellulaire) pour l'étude de la maladie de Fabry. Il génère des visualisations statistiques et des comparaisons entre différents groupes de patients.

## Fonctionnalités

- Analyse statistique complète des données VEC
- Génération de visualisations :
  - Boîtes à moustaches pour les comparaisons entre groupes
  - Diagrammes Bullseye pour la visualisation cardiaque
- Tests statistiques automatiques (t-test ou Mann-Whitney selon la normalité)
- Export automatique des résultats en document Word

## Prérequis

- Python 3.x
- Packages Python requis (listés dans `requirements.txt`)

## Installation

1. Clonez ce dépôt
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Placez votre fichier Excel de données dans le dossier du projet
2. Exécutez le script d'analyse :
```bash
python analysis.py
```

Le script génèrera :
- Des visualisations au format PNG
- Un document Word avec l'analyse complète (`Analyse_VEC_Fabry_comparaisons.docx`)

## Structure des fichiers

- `analysis.py` : Script principal d'analyse
- `requirements.txt` : Liste des dépendances Python
- `CRF_Fabry_unique_modifCF.xlsx` : Données source (non inclus dans le dépôt)
- Fichiers de sortie générés :
  - `VEC_boxplot_*.png` : Graphiques en boîtes à moustaches
  - `VEC_bullseye_*.png` : Diagrammes Bullseye
  - `Analyse_VEC_Fabry_comparaisons.docx` : Rapport final