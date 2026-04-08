# Enhancing Diagnostic Accuracy for Urinary Tract Disease through Explainable SHAP-Guided Feature Selection and Classification (ijcnn_pap3345)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Research-blue)](#workflow)

#### By [Filipe Ferreira de Oliveira](mailto:filipe.f.oliveira98@gmail.com) and [Matheus Becali Rocha](mailto:matheusbecali@gmail.com)

This repository provides an end-to-end machine learning workflow for Urinary Tract Disease classification using clinical laboratory data, with emphasis on performance and interpretability.

The pipeline includes:

- model comparison across `XGBoost`, `LightGBM`, and `CatBoost`
- stratified cross-validation with metric reporting
- SHAP-based global feature importance analysis
- top-N feature reduction and re-evaluation of model performance

The project currently includes the following binary experiments:

- **Bladder Cancer vs Cystitis**
- **Bladder Cancer vs Kidney Cancer**
- **Bladder Cancer vs Others**
- **Bladder Cancer vs Prostate Cancer**
- **Bladder Cancer vs Uterus Cancer**
- **Prostate Cancer vs Others**

## Workflow

The project is organized into three sequential stages:

1. **Baseline training and evaluation** (`main.py`)
2. **SHAP analysis and feature ranking** (`shap_analysis_script.py`)
3. **Cross-validated re-evaluation using SHAP top-N features** (`main_shap.py`)

Evaluation metrics include:

- Accuracy
- Balanced Accuracy
- F1-score
- Precision
- Sensitivity
- Specificity

## Environment Setup

Recommended: Python 3.11+ in a virtual environment.

```bash
conda create --name YOUR_ENV_NAME python=3.12
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```

## Dataset

This project uses the clinical laboratory bladder cancer dataset made publicly available by the original work from I-JUNG TSAI.

- Local file in this repository: [Predict-BC-Clinical-Laboratory-Data](https://github.com/I-JUNG-TSAI/Predict-BC-Clinical-Laboratory-Data)
- Source URL data used in code: [Link](https://raw.githubusercontent.com/I-JUNG-TSAI/Predict-BC-Clinical-Laboratory-Data/main/The%20clinical%20laboratory%20data%20of%20bladder%20cancer%20.csv)

## Repository Structure

```text
.
├── main.py
├── shap_analysis_script.py
├── main_shap.py
├── run.sh
├── src/
│   ├── pipeline.py
│   └── featureSelector.py
├── utils/
│   └── helpers.py
├── best_params/
├── data_splits/
├── saved_pipelines/
├── confusion_matrices/
├── metrics_per_fold/
└── shap_analysis_results/
```

## Run

### Option 1: full pipeline script

```bash
bash run.sh
```

### Option 2: step-by-step execution

```bash
python main.py
python shap_analysis_script.py
python main_shap.py
```

## Generated Artifacts

After execution, key outputs include:

- `all_metrics.csv`: aggregated baseline metrics
- `best_params.csv` and `best_params/*.json`: best hyperparameters
- `saved_pipelines/*.joblib`: saved best pipeline per experiment
- `data_splits/*.csv`: train/test splits
- `confusion_matrices/*`: confusion matrix files (CSV and plots)
- `metrics_per_fold/*`: fold-level metrics
- `shap_analysis_results/final_summary_report.csv`: SHAP selection summary
- `shap_analysis_results/feature_importance_data/*`: SHAP feature rankings
- `shap_analysis_results/plots/*`: summary plots and performance curves
- `shap_features_cv_evaluation.csv`: CV metrics using SHAP top-N features

## Notes

- `main.py` clears previously generated artifacts before a full rerun.
- The `experiments` block includes 6 binary comparisons and can be adjusted as needed for each run.
- The SHAP feature reduction stage selects the best feature subset based on **Balanced Accuracy (BACC)**.

## Future Work

1. Expand the pipeline to datasets in another domain;
2. Test other feature selection techniques.

## Cite This Repository

If this repository supports your research, please cite:

```bibtex
@misc{oliveira_rocha_2026_bladder_shap,
  author       = {Filipe Ferreira de Oliveira and Matheus Becali Rocha},
  title        = {Enhancing Diagnostic Accuracy for Urinary Tract Disease through Explainable SHAP-Guided Feature Selection and Classification},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Available at: https://github.com/filipefolv/shap-utd-explainability-study}
}
```

## Acknowledgement

1. I-JUNG TSAI's project for making the clinical dataset publicly available.
2. The open-source communities behind `scikit-learn`, `imbalanced-learn`, `XGBoost`, `LightGBM`, `CatBoost`, and `SHAP`.
