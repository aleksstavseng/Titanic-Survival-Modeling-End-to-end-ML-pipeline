# Titanic Survival — End-to-end ML pipeline

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-ff9900?logo=scikitlearn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)


**Results**
- 5-fold CV accuracy: **0.829 ± 0.013**
- Hold-out accuracy: **0.793**

Artifacts: `results/model_report.md`, `models/hgb_min.pkl`, `submission.csv`.
**Release:** se siste versjon under *Releases* (v0.1 med zip av artefakter).

**Quick start**
## Reproduce locally

> Krever Python 3.12+ og `pip`.

```bash
# 1) Hent kode
git clone https://github.com/aleksstavseng/Titanic-Survival-Modeling-End-to-end-ML-pipeline.git
cd Titanic-Survival-Modeling-End-to-end-ML-pipeline

# 2) (Valgfritt) Virtuelt miljø
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3) Avhengigheter
pip install -r requirements.txt

# 4) Data (legg inn Kaggle-filene)
#   data/train.csv og data/test.csv
#   (opprett mappa hvis den mangler)
mkdir -p data
# kopier inn train.csv og test.csv her

# 5) Kjør treningen
python train.py
# Artefakter: submission.csv, results/model_report.md, models/hgb_min.pkl

```bash
pip install -r requirements.txt
python train.py
