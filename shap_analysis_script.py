# -*- coding: utf-8 -*-
"""
-------------------------------------
Authors: Filipe Ferreira de Oliveira,
        Matheus Becali Rocha
-------------------------------------
Email: filipe.f.oliveira98@gmail.com, 
       matheusbecali@gmail.com
-------------------------------------
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,f1_score,
                            precision_score, recall_score, confusion_matrix)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =========================================================================== #
# CONFIGURATION
# =========================================================================== #

os.makedirs("shap_analysis_results/plots/performance_curves", exist_ok=True)
os.makedirs("shap_analysis_results/plots/confusion_matrices", exist_ok=True)

feature_selection = "None"
SEED = 42

ALL_CAT_FEATURES = [
    "Nitrite", "Urine occult Blood", "Urine Bilirubin", "Urine Glucose",
    "Urine Ketone", "Urine Protein", "Urobilinogen", "gender", "Hyper1en1ion",
    "Diabe1es", "Smoking", "Drinking", "Bee1leNu1", "FamilyHis1ory"
]

# List to store information about BEST models to be analyzed
all_models_to_analyze = []

# scan 'best_params' folder to find .json files of best models
best_params_files = glob.glob('best_params/*.json')

if not best_params_files:
    raise FileNotFoundError(
        "No best-parameters (.json) file found in 'best_params/' folder. "
        "Check if the first notebook was executed and saved the results."
    )

# iterate over each .json file to identify best model and its pipeline
for json_path in best_params_files:
    base_name = os.path.basename(json_path)
    clean_name = base_name.replace('_best_params.json', '')
    parts = clean_name.split('_')

    model_type = parts[-1]
    experiment_name = '_'.join(parts[:-1])

    pipeline_file_path = (
        f"saved_pipelines/"
        f"{experiment_name}_{model_type}_{feature_selection}_best_pipeline.joblib"
    )

    if os.path.exists(pipeline_file_path):
        all_models_to_analyze.append({
            'experiment_name': experiment_name,
            'model_type': model_type,
            'pipeline_file': pipeline_file_path
        })
    else:
        print(
            f"WARNING: Parameters found for '{experiment_name} ({model_type})', "
            f"but matching pipeline at '{pipeline_file_path}' was not found. Skipping."
        )

print(f"\n{len(all_models_to_analyze)} best model/pipeline pairs identified for SHAP analysis:")

all_models_to_analyze.sort(key=lambda x: x['experiment_name'])

for model_info in all_models_to_analyze:
    print(f"- Experiment: {model_info['experiment_name']} ----> {model_info['model_type']}")

# load baseline metrics for comparison
try:
    baseline_metrics_df = pd.read_csv('all_metrics.csv')
    baseline_metrics_df.set_index(['experiment', 'model'], inplace=True)
    print("Baseline metrics loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(
        "File 'all_metrics.csv' was not found. "
        "Run the first notebook to generate it."
    )

# =========================================================================== #

# auxiliary transformer to convert arrays back to DataFrames (for SHAP)
class ArrayToDataFrame(BaseEstimator, TransformerMixin):
    """
    Transformer that converts a NumPy array into a pandas DataFrame.
    """
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.column_names)

# =========================================================================== #

def average_results(results_list):
    """
    Compute average values from a list of metric dictionaries.

    Args:
        results_list: list[dict] with keys 'bal_acc', 'accuracy', etc.

    Returns:
        dict with averages and n_features.
    """
    if not results_list:
        return {}

    numeric_keys = [
        'bal_acc', 'accuracy', 'roc_auc', 'f1',
        'precision', 'sensitivity', 'specificity'
    ]

    avg_results = {}

    for key in numeric_keys:
        values = [d[key] for d in results_list]
        avg_results[key] = np.mean(values)

    avg_results['n_features'] = results_list[0]['n_features']

    return avg_results

# =========================================================================== #

def run_shap_feature_selection(experiment_name, model_name, pipeline_path, baseline_metrics):
    """
    Run SHAP-based feature selection and reevaluate a specific model
    using its corresponding saved data split.
    """
    print(
        f"\n{'='*80}\n"
        f"--- SHAP analysis for: {experiment_name} (Model: {model_name}) ---\n"
        f"{'='*80}"
    )

    print(f"Loading data split: data_splits/{experiment_name}_*.csv")

    try:
        X_train = pd.read_csv(f'data_splits/{experiment_name}_X_train.csv')
        y_train = pd.read_csv(f'data_splits/{experiment_name}_y_train.csv').squeeze()
        X_test = pd.read_csv(f'data_splits/{experiment_name}_X_test.csv')
        y_test = pd.read_csv(f'data_splits/{experiment_name}_y_test.csv').squeeze()
    except FileNotFoundError as e:
        print(
            f"ERROR: Data file not found for {experiment_name}. "
            f"Check your folder structure."
        )
        print(e)
        return None

    pipeline = joblib.load(pipeline_path)

    processing_pipeline = ImbPipeline(pipeline.steps[:-1])

    # Final data (post-OHE)
    X_train_final = processing_pipeline.transform(X_train)
    X_test_final = processing_pipeline.transform(X_test)

    # Final feature names (from 'ohe' step)
    final_feature_names = pipeline.named_steps['ohe'].get_feature_names_out()

    # Clean names (remove OHE/remainder prefixes)
    final_feature_names_cleaned = [name.split('__')[-1] for name in final_feature_names]

    # Previously saved SHAP importances
    importance_csv_path = (
        "shap_analysis_results/feature_importance_data/"
        f"{experiment_name}_{model_name}_{feature_selection}_feature_importance.csv"
    )
    shap_importance_df = pd.read_csv(importance_csv_path)

    # Classifier blueprint
    clf_blueprint = clone(pipeline.named_steps['clf'])

    results_for_n = []
    n_features_to_test = list(range(2, len(final_feature_names) + 1, 1))

    for n in n_features_to_test:
        if n > len(shap_importance_df):
            continue

        print(f"Testing with top {n} features...")

        top_n_features = shap_importance_df['Feature Name'].head(n).to_list()

        try:
            top_n_indices = [
                final_feature_names_cleaned.index(f) for f in top_n_features
            ]
        except ValueError as e:
            print(
                f"Error: Feature "
                f"'{e.args[0].split(' is not in list')[0]}' from your CSV was not found "
                f"in pipeline feature names. Check consistency."
            )
            continue

        X_train_reduced = X_train_final.iloc[:, top_n_indices]
        X_test_reduced = X_test_final.iloc[:, top_n_indices]

        reduced_pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf", clone(clf_blueprint))
        ])

        reduced_pipeline.fit(X_train_reduced, y_train)

        y_pred = reduced_pipeline.predict(X_test_reduced)
        y_proba = reduced_pipeline.predict_proba(X_test_reduced)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        accuracy = float(accuracy_score(y_test, y_pred))
        bal_acc = float(balanced_accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred))
        sensitivity = float(recall_score(y_test, y_pred, pos_label=1))
        specificity = float(recall_score(y_test, y_pred, pos_label=0))

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        results_for_n.append({
            'n_features': n,
            'bal_acc': bal_acc,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1': f1,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': cm
        })

    if not results_for_n:
        return None

    results_df = pd.DataFrame(results_for_n)
    # print(results_df)
    results_df.to_csv(
        f"shap_analysis_results/"
        f"{experiment_name}_{model_name}_{feature_selection}_results.csv",
        index=False
    )

    best_n_result = results_df.loc[results_df['bal_acc'].idxmax()]
    baseline_perf = baseline_metrics.loc[(experiment_name, model_name)]

    return {
        'experiment_name': experiment_name,
        'model_type': model_name,
        'best_n': int(best_n_result['n_features']),
        'bal_acc_at_best_n': best_n_result['bal_acc'],
        'roc_auc_at_best_n': best_n_result['roc_auc'],
        'f1_at_best_n': best_n_result['f1'],
        'precision_at_best_n': best_n_result['precision'],
        'sensitivity_at_best_n': best_n_result['sensitivity'],
        'specificity_at_best_n': best_n_result['specificity'],
        'accuracy_at_best_n': best_n_result['accuracy'],
        'baseline_bal_acc': baseline_perf['test_bal_acc_mean'],
        'baseline_roc_auc': baseline_perf['test_roc_auc_mean'],
        'feature_importance': shap_importance_df,
        'performance_curve': results_df,
    }

# =========================================================================== #

def main():
    all_shap_experiments_results = []

    for model_info in all_models_to_analyze:
        result = run_shap_feature_selection(
            experiment_name=model_info['experiment_name'],
            model_name=model_info['model_type'],
            pipeline_path=model_info['pipeline_file'],
            baseline_metrics=baseline_metrics_df
        )
        if result:
            all_shap_experiments_results.append(result)

    # comparative summary table
    summary_data = []
    for r in all_shap_experiments_results:
        summary_data.append({
            "Experiment": r['experiment_name'],
            "Model": r['model_type'],
            "Baseline Bal. Acc": f"{r['baseline_bal_acc']:.4f}",
            "Optimal Features (SHAP)": r['best_n'],
            "Reduced Bal. Acc": f"{r['bal_acc_at_best_n']:.4f}",
            "Delta": r['bal_acc_at_best_n'] - r['baseline_bal_acc'],
            "Accuracy": r['accuracy_at_best_n'],
            "Balanced Accuracy": r['bal_acc_at_best_n'],
            "ROC AUC": r['roc_auc_at_best_n'],
            "F1-Score": r['f1_at_best_n'],
            "Precision": r['precision_at_best_n'],
            "Sensitivity": r['sensitivity_at_best_n'],
            "Specificity": r['specificity_at_best_n']
        })

    final_summary_df = (
        pd.DataFrame(summary_data)
        .sort_values(by=["Experiment", "Model"])
        .reset_index(drop=True)
    )
    summary_csv_path = "shap_analysis_results/final_summary_report.csv"
    final_summary_df.to_csv(summary_csv_path, encoding='utf-8', index=False)

    # performance plots vs number of features
    print(
        "\n\n--- PERFORMANCE CURVE: "
        "Balanced Accuracy vs Number of Features ---"
    )
    for r in sorted(
        all_shap_experiments_results,
        key=lambda x: (x['experiment_name'], x['model_type'])
    ):
        exp_name = r['experiment_name']
        model_type = r['model_type']
        curve_data = r['performance_curve']

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=curve_data, x='n_features', y='bal_acc',
            marker='o', label="Balanced Accuracy"
        )
        plt.axhline(
            y=r['baseline_bal_acc'], color='r', linestyle='--',
            label=f"Reference (Bal. Acc.:{r['baseline_bal_acc']:.3f})"
        )
        best_n = r['best_n']
        best_acc = r['bal_acc_at_best_n']
        plt.axvline(
            x=best_n, color='g', linestyle='--',
            label=f"Best N = {best_n} (Bal. Acc.: {best_acc:.3f})"
        )

        plt.title(
            f'Model Sensitivity Analysis: {model_type}\n'
            f'Experiment: {exp_name}',
            fontsize=14
        )
        plt.xlabel('Number of Features Used', fontsize=12)
        plt.ylabel('Balanced Accuracy', fontsize=12)
        plt.legend()
        plt.grid(True, which='both', linestyle='--')

        curve_plot_path = (
            "shap_analysis_results/plots/performance_curves/"
            f"{exp_name}_{model_type}_perf_curve.png"
        )
        plt.savefig(curve_plot_path, bbox_inches='tight')
        print(f"Performance curve plot saved at {curve_plot_path}")
        plt.close()

if __name__ == "__main__":
    main()
