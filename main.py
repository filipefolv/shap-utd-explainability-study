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
import ast
import shap
import json
import torch
import optuna
import joblib
import warnings
import logging

from typing import List
from pathlib import Path
from functools import partial
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             f1_score, precision_score, recall_score, confusion_matrix)


from src.pipeline import build_pipeline, objective
from utils.helpers import (clear_memory, make_binary_subset,
                           show_split_info, load_best_hyperparameters)

# =========================================================================== #
# CONFIGURATION
# =========================================================================== #

# Suppress non-critical informative warnings
warnings.filterwarnings('ignore', category=UserWarning, 
                      message='.*Found unknown categories.*')
warnings.filterwarnings('ignore', category=UserWarning,
                      message='.*does not have valid feature names.*')
warnings.filterwarnings('ignore', category=FutureWarning, 
                      message='.*NumPy global RNG.*')

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

cuda_available = torch.cuda.is_available()
logger.info(f"CUDA available: {cuda_available}")
if cuda_available:
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")


SEED = 42
np.random.seed(SEED)

@dataclass
class ExperimentConfig:
    """Experiment configuration values."""
    missing_threshold: float = 45.0
    n_trials: int = 100
    test_size: float = 0.2
    n_cv_folds: int = 5
    random_state: int = SEED


@dataclass
class FeatureConfig:
    """Feature group configuration."""
    categorical: List[str]
    numeric: List[str]

# =========================================================================== #

class ArrayToDataFrameAfterOHE(BaseEstimator, TransformerMixin):
    """
    Transformer that converts arrays to DataFrame after OHE.
    It gets feature names dynamically from the previous pipeline step.
    """
    def __init__(self):
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        # During fit, store the feature names
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
        else:
            # If X is an array, create generic feature names
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        return self
    
    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.feature_names_)

# =========================================================================== #

def evaluate_best_params_with_cv(best_params, exp_name, model_name, X_train, y_train, X_test, y_test, 
                                 feature_config, feature_selection="None", n_splits=5):
    """
    Evaluate best hyperparameters using K-fold CV on the training set.
    For each fold, train a model, validate on fold split, and test on fixed test set.
    
    Returns:
        dict: Validation and test metrics with mean and standard deviation
        list: Trained pipelines for each fold
        pd.DataFrame: Detailed metrics by fold
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    val_metrics = {
        'accuracy': [], 'bal_acc': [], 'roc_auc': [], 'f1': [],
        'precision': [], 'sensitivity': [], 'specificity': []
    }
    
    test_metrics = {
        'accuracy': [], 'bal_acc': [], 'roc_auc': [], 'f1': [],
        'precision': [], 'sensitivity': [], 'specificity': []
    }
    
    trained_pipelines = []
    fold_details = []
    columns_fs_pipelines = []
    
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name} with {n_splits}-Fold CV")
    print(f"{'='*70}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        print(f"\nFold {fold_idx}/{n_splits}")
        print("-" * 50)
        
        # Split fold data
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        # Build pipeline using fixed best hyperparameters
        trial_fixed = optuna.trial.FixedTrial(best_params)
        pipeline = build_pipeline(trial_fixed, exp_name, model_name, X_tr, feature_config, feature_selection)        

        # Train
        print(f" Training... (n_train={len(y_tr)}, n_val={len(y_val)})")
        pipeline.fit(X_tr, y_tr)

        columns_fs_pipelines.append("FULL FEATURES")

        trained_pipelines.append(pipeline)

        # ===== VALIDATION =====
        y_val_pred = pipeline.predict(X_val)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        
        val_acc = accuracy_score(y_val, y_val_pred)
        val_bal_acc = balanced_accuracy_score(y_val, y_val_pred)
        val_roc_auc = roc_auc_score(y_val, y_val_proba)
        val_f1 = f1_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, zero_division=0)
        val_sensitivity = recall_score(y_val, y_val_pred, pos_label=1)
        val_specificity = recall_score(y_val, y_val_pred, pos_label=0)
        
        val_metrics['accuracy'].append(val_acc)
        val_metrics['bal_acc'].append(val_bal_acc)
        val_metrics['roc_auc'].append(val_roc_auc)
        val_metrics['f1'].append(val_f1)
        val_metrics['precision'].append(val_precision)
        val_metrics['sensitivity'].append(val_sensitivity)
        val_metrics['specificity'].append(val_specificity)
        
        print(f"  Validation - Bal_Acc: {val_bal_acc:.4f}, ROC-AUC: {val_roc_auc:.4f}")

        # ===== TEST =====
        y_test_pred = pipeline.predict(X_test)
        y_test_proba = pipeline.predict_proba(X_test)[:, 1]
        
        test_acc = accuracy_score(y_test, y_test_pred)
        test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        test_f1 = f1_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_sensitivity = recall_score(y_test, y_test_pred, pos_label=1)
        test_specificity = recall_score(y_test, y_test_pred, pos_label=0)
        
        test_metrics['accuracy'].append(test_acc)
        test_metrics['bal_acc'].append(test_bal_acc)
        test_metrics['roc_auc'].append(test_roc_auc)
        test_metrics['f1'].append(test_f1)
        test_metrics['precision'].append(test_precision)
        test_metrics['sensitivity'].append(test_sensitivity)
        test_metrics['specificity'].append(test_specificity)
        
        print(f"  Test       - Bal_Acc: {test_bal_acc:.4f}, ROC-AUC: {test_roc_auc:.4f}")
        
        # Store fold-level details
        fold_details.append({
            'fold': fold_idx,
            'test_accuracy': test_acc,
            'test_bal_acc': test_bal_acc,
            'test_roc_auc': test_roc_auc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_sensitivity': test_sensitivity,
            'test_specificity': test_specificity
        })

    # Compute summary statistics
    summary = {}
    # for split_name, metrics_dict in [('val', val_metrics), ('test', test_metrics)]:
    for split_name, metrics_dict in [('test', test_metrics)]:
        for metric_name, values in metrics_dict.items():
            summary[f'{split_name}_{metric_name}_mean'] = float(np.mean(values))
            summary[f'{split_name}_{metric_name}_std'] = float(np.std(values))
    
    # Display summary
    print(f"\n{'='*70}")
    print(f"SUMMARY - {model_name}")
    print(f"{'='*70}")
    
    print(f"\nTEST (Fixed set, evaluated {n_splits}x):")
    print(f"  Balanced Accuracy: {summary['test_bal_acc_mean']:.4f} ± {summary['test_bal_acc_std']:.4f}")
    print(f"  ROC-AUC:          {summary['test_roc_auc_mean']:.4f} ± {summary['test_roc_auc_std']:.4f}")
    print(f"  F1-Score:         {summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}")
    print(f"{'='*70}\n")
    
    fold_details_df = pd.DataFrame(fold_details)
    
    return summary, trained_pipelines, fold_details_df, columns_fs_pipelines

# =========================================================================== #

def get_stable_features(trained_pipelines, threshold=1.0):
    """
    Return only stable features (present in >= threshold of folds).
    
    Args:
        threshold: 1.0 = 100% of folds, 0.8 = 80%, etc.
    """
    from collections import Counter
    
    all_features = []
    for pipeline in trained_pipelines:
        fs = pipeline.named_steps['feature_selection']
        all_features.extend(fs.selected_feature_names_)
    
    feature_counts = Counter(all_features)
    n_folds = len(trained_pipelines)
    
    stable_features = [
        feat for feat, count in feature_counts.items()
        if count >= (threshold * n_folds)
    ]
    
    print(f"\nStable features (threshold={threshold*100}%):")
    print(f"  {stable_features}")
    
    return stable_features

# =========================================================================== #

def run_experiment(df, exp_name, pos_labels, neg_labels, config_params, 
                   feature_config, feature_selection="None"):
    """Run a complete experiment pipeline."""
    logger.info(f"{'='*70}")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"{'='*70}")

    ALL_EXPERIMENTS_BEST_METRICS = []

    MISSING_THRESHOLD = config_params.missing_threshold
    random_state = config_params.random_state
    N_TRIALS = config_params.n_trials
    TEST_SIZE = config_params.test_size

    # Build binary subset for the current experiment
    data, exp_name = make_binary_subset(df, pos_labels, neg_labels, exp_name)
    X, y = data.drop(columns=['target', 'label']), data['target']

    missing_pct = (X.isnull().sum() / len(X)) * 100
    features_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    
    if features_to_drop:
        logger.info(
            f"Removing {len(features_to_drop)} features with >"
            f"{MISSING_THRESHOLD}% missing data:"
        )
        for feature in features_to_drop:
            logger.info(f"  - {feature} ({missing_pct[feature]:.2f}%)")
        X = X.drop(columns=features_to_drop)
        logger.info(f"\nX shape after removal: {X.shape}")
    else:
        logger.info("No feature exceeded the missing-data threshold")

    # Split into train+validation and fixed test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=random_state
    )

    # Save data splits
    X_train.to_csv(f'data_splits/{exp_name}_X_train.csv', index=False, header=True, encoding='utf-8')
    X_test.to_csv(f'data_splits/{exp_name}_X_test.csv', index=False, header=True, encoding='utf-8')
    y_train.to_csv(f'data_splits/{exp_name}_y_train.csv', index=False, header=True, encoding='utf-8')
    y_test.to_csv(f'data_splits/{exp_name}_y_test.csv', index=False, header=True, encoding='utf-8')
    logger.info(f"Data split saved (CSV) for experiment {exp_name}.")

    # Display split information
    show_split_info(y_train, y_test, exp_name)

    models = ['XGBoost', 'LightGBM', 'CatBoost']
    best_params = {model: None for model in models}
    all_summaries = {}
    all_fold_details = {}
    all_trained_pipelines = {}
    all_columns_fs = {}

    best_model = None
    best_test_bal_acc = -1

    # =========================================================================== #
    # hyperparameter optimization with optuna 
    # (commented out since we are loading best params from CSV)
    # =========================================================================== #

    # print("\n" + "="*70)
    # print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    # print("="*70)
    
    # for model_name in models:
    #     print(f"\nOptimizing {model_name}...")

    #     study = optuna.create_study(direction='maximize')
    #     objective_with_data = partial(objective, model_name=model_name, 
    #         X_train=X_train, y_train=y_train, 
    #         feature_config=feature_config)
        
    #     study.optimize(objective_with_data, n_trials=N_TRIALS,
    #                    show_progress_bar=True, gc_after_trial=True)

    #     print(f"{model_name} - Best Bal_Acc (Optuna CV): {study.best_value:.4f}")
    #     best_params[model_name] = study.best_trial.params

    # =========================================================================== #

    # K-FOLD evaluation with best hyperparameters
    print("\n" + "="*70)
    print("EVALUATION WITH K-FOLD CROSS-VALIDATION")
    print("="*70)
    
    best_params_path = f"./best_params.csv"

    for model_name in models:
        print(f"\n{'#'*70}")
        print(f"# Model: {model_name}")
        print(f"{'#'*70}")
        
        params = load_best_hyperparameters(best_params_path, exp_name, model_name)
        print(f"\nExperiment: {exp_name}")
        print(f"Model: {model_name}")
        print("\nHyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        # Evaluate with K-fold CV
        summary, trained_pipelines, fold_details_df, columns_fs = evaluate_best_params_with_cv(
            best_params=params,
            exp_name=exp_name,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_config=feature_config,
            feature_selection=feature_selection,
            n_splits=5
        )
        
        stable_features = "FULL FEATURES"
        all_columns_fs[model_name] = stable_features
        all_summaries[model_name] = summary
        all_trained_pipelines[model_name] = trained_pipelines
        all_fold_details[model_name] = fold_details_df
        
        # Save fold metrics
        fold_details_filename = f"metrics_per_fold/{exp_name}_{model_name}_{feature_selection}_fold_details.csv"
        os.makedirs("metrics_per_fold", exist_ok=True)
        fold_details_df.to_csv(fold_details_filename, encoding='utf-8', index=False)
        print(f"Detailed fold metrics saved at {fold_details_filename}")
        
        # Update best model based on fixed test performance
        # if summary['val_bal_acc_mean'] > best_val_bal_acc:
        if summary['test_bal_acc_mean'] > best_test_bal_acc:
            best_test_bal_acc = summary['test_bal_acc_mean']
            best_model = model_name

        # Store experiment summary for final report
        metrics_for_report = {
            'experiment': exp_name,
            'model': model_name,
            'feature selection': feature_selection,
            **summary,
            'columns_fs': stable_features
            # 'best_trial_params': best_params[model_name]
        }
        ALL_EXPERIMENTS_BEST_METRICS.append(metrics_for_report)
        
    # =========================================================================== #

    # Select best model and save final results
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_model}")
    print(f"Test Bal_Acc:      {all_summaries[best_model]['test_bal_acc_mean']:.4f} ± {all_summaries[best_model]['test_bal_acc_std']:.4f}")
    print("="*70)

    best_pipeline = all_trained_pipelines[best_model][0]  # First fold
    
    # Save best pipeline
    pipeline_filename = f"saved_pipelines/{exp_name}_{best_model}_{feature_selection}_best_pipeline.joblib"
    joblib.dump(best_pipeline, pipeline_filename)
    print(f"\nPipeline saved at {pipeline_filename}")
    
    # Save best hyperparameters
    params_filename = f"best_params/{exp_name}_{best_model}_best_params.json"
    with open(params_filename, 'w') as f:
        json.dump(best_params[best_model], f, indent=4)
    print(f"Best hyperparameters saved at {params_filename}")

    # confusion Matrix (using aggregated predictions from all folds)
    print("\nComputing aggregated confusion matrix...")
    all_test_preds = []
    
    for pipeline in all_trained_pipelines[best_model]:
        y_pred = pipeline.predict(X_test)
        all_test_preds.append(y_pred)
    
    # majority vote across fold predictions
    all_test_preds = np.array(all_test_preds)
    y_test_pred_final = np.round(all_test_preds.mean(axis=0)).astype(int)
    
    cm = confusion_matrix(y_test, y_test_pred_final, labels=[0, 1])
    
    # save non-normalized confusion matrix
    cm_df = pd.DataFrame(cm, index=[f'Actual {neg_labels}', f'Actual {pos_labels}'],
        columns=[f'Predicted {neg_labels}', f'Predicted {pos_labels}'])
    
    cm_filename = f'confusion_matrices/confusion_matrix_{exp_name}_{best_model}_{feature_selection}.csv'
    
    cm_df.to_csv(cm_filename, encoding='utf-8')
    print(f"Confusion matrix saved at {cm_filename}")

    # save normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    tick_labels_x = [f'Predicted: {neg_labels} (0)', f'Predicted: {pos_labels} (1)']
    tick_labels_y = [f'Actual: {neg_labels} (0)', f'Actual: {pos_labels} (1)']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                xticklabels=tick_labels_x, yticklabels=tick_labels_y)
    plt.title(
        f'Normalized Confusion Matrix: {exp_name} - {best_model} - {feature_selection}'
    )
    plt.xlabel('Model Prediction')
    plt.ylabel('True Class')
    plt.tight_layout()
    cm_normalized_filename = (
        f'confusion_matrices/'
        f'confusion_matrix_{exp_name}_{best_model}_{feature_selection}_normalized.png'
    )
    plt.savefig(cm_normalized_filename)
    plt.close()
    print(f"Normalized confusion matrix saved at {cm_normalized_filename}")

    # =========================================================================== #
    # shap analysis
    print("\n" + "="*70)
    print("SHAP ANALYSIS OF THE BEST MODEL")
    print("="*70)
    
    classifier = best_pipeline.named_steps['clf']
    processing_pipeline = ImbPipeline(best_pipeline.steps[:-1])
    X_test_final_transformed = processing_pipeline.transform(X_test)

    final_feature_names = best_pipeline.named_steps['ohe'].get_feature_names_out()
    cleaned_feature_names = [name.split('__')[-1] for name in final_feature_names]
    assert X_test_final_transformed.shape[1] == len(cleaned_feature_names), "Number of columns does not match the number of feature names!"

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_test_final_transformed)
    
    print("\nGenerating and saving SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_test_final_transformed, feature_names=cleaned_feature_names, 
                      show=False)
    plt.tight_layout()
    # plt.title(
    #     f"SHAP Summary: {exp_name} - {best_model} - {feature_selection}",
    #     fontsize=12
    # )
    # summary_plot_path = (
    #     f"shap_analysis_results/plots/summary_plots/"
    #     f"{exp_name}_{best_model}_{feature_selection}_shap_summary.png"
    # )
    plt.title(f"SHAP Summary: {exp_name} - {best_model}", fontsize=12)

    summary_plot_path = (
        f"shap_analysis_results/plots/summary_plots/"
        f"{exp_name}_{best_model}_{feature_selection}_shap_summary.png"
    )
    plt.savefig(summary_plot_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP plot saved at {summary_plot_path}")

    # Feature ranking
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_importance_df = pd.DataFrame(
        list(zip(cleaned_feature_names, mean_abs_shap)),
        columns=['Feature Name', 'SHAP Importance']
    ).sort_values(by='SHAP Importance', ascending=False).reset_index(drop=True)

    importance_csv_path = (f"shap_analysis_results/feature_importance_data/"
        f"{exp_name}_{best_model}_{feature_selection}_feature_importance.csv")
    shap_importance_df.to_csv(importance_csv_path, encoding='utf-8', index=False)

    print(f"Feature ranking saved at {importance_csv_path}")

    # Save final experiment summary
    print(f"\nExperiment {exp_name} completed.\n")

    results_df = pd.DataFrame(ALL_EXPERIMENTS_BEST_METRICS)
    results_df.to_csv("all_metrics.csv", mode='a',header=not Path('all_metrics.csv').exists(), 
                      encoding='utf-8', index=False)
    
    print("Final summary of all metrics saved in 'all_metrics.csv'.")

# =========================================================================== #

def main():
    """
    Main function that executes all experiments.
    """
    print("="*80)
    print(f" START OF ANALYSIS - UROLOGICAL CANCER CLASSIFICATION")
    print("="*80)
    
    # load data from GitHub original file by I JUNG TSAI
    print("\n[1/4] Loading data...")
    url = (
        "https://raw.githubusercontent.com/I-JUNG-TSAI/"
        "Predict-BC-Clinical-Laboratory-Data/main/"
        "The%20clinical%20laboratory%20data%20of%20bladder%20cancer%20.csv"
    )
    df = pd.read_csv(url)
    df.rename(columns={'Disease': 'label'}, inplace=True)
    df.drop(columns=['Patient Number'], inplace=True)
    print(f"   Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"   Classes: {df['label'].unique().tolist()}")
    print(f"   Distribution: \n{df['label'].value_counts()}")
    
    # configure features categorical vs numeric
    print("\n[2/4] Configuring features...")
    categorical_features = [
        "Nitrite", "Urine occult Blood", "Urine Bilirubin", "Urine Glucose",
        "Urine Ketone", "Urine Protein", "Urobilinogen", "gender",
        "Hyper1en1ion", "Diabe1es", "Smoking", "Drinking",
        "Bee1leNu1", "FamilyHis1ory"
    ]
    numeric_features = [
        c for c in df.columns
        if c not in categorical_features + ['label']
    ]
    
    feature_config = FeatureConfig(
        categorical=categorical_features,
        numeric=numeric_features
    )
    
    print(f"   Categorical features: {len(categorical_features)}")
    print(f"   Numeric features: {len(numeric_features)}")
    
    # configure experimental parameters
    print("\n[3/4] Configuring experiment...")
    config = ExperimentConfig(
        missing_threshold=45.0,
        n_trials=5,
        test_size=0.15,
        n_cv_folds=5,
        random_state=SEED
    )
    
    print(f"   Missing threshold: {config.missing_threshold}%")
    print(f"   Trials Optuna: {config.n_trials}")
    print(f"   CV folds: {config.n_cv_folds}")
    print(f"   Random state: {config.random_state}")
    
    # create output directories if they don't exist
    output_dirs = [
        'saved_pipelines', 
        'data_splits', 
        'best_params', 
        'confusion_matrices',
        'metrics_per_fold',
        'shap_analysis_results/plots/summary_plots',
        'shap_analysis_results/feature_importance_data'
    ]
    for folder in output_dirs:
        os.makedirs(folder, exist_ok=True)

    # define experiments with their respective positive and negative labels
    print("\n[4/4] Running experiments...")

    experiments = [
        {
            'exp_name': 'Bladder Cancer vs Prostate Cancer',
            'pos_labels': 'Bladder Cancer',
            'neg_labels': 'Prostate Cancer'
        },
        # {
        #     'exp_name': 'Bladder Cancer vs Cystitis',
        #     'pos_labels': 'Bladder Cancer',
        #     'neg_labels': 'Cystitis'
        # },
        # {
        #     'exp_name': 'Bladder Cancer vs Kidney Cancer',
        #     'pos_labels': 'Bladder Cancer',
        #     'neg_labels': 'Kidney Cancer'
        # },
        # {
        #     'exp_name': 'Bladder Cancer vs Uterus Cancer',
        #     'pos_labels': 'Bladder Cancer',
        #     'neg_labels': 'Uterus Cancer'
        # },
        # {
        #     'exp_name': 'Bladder Cancer vs Others',
        #     'pos_labels': 'Bladder Cancer',
        #     'neg_labels': 'Others'
        # },
        # {
        #     'exp_name': 'Prostate Cancer vs Others',
        #     'pos_labels': 'Prostate Cancer',
        #     'neg_labels': 'Others'
        # },
    ]
    
    print(f"\n   Total experiments to run: {len(experiments)}\n")
    
    # run each experiment in sequence, with error handling to continue even if one fails
    for idx, exp in enumerate(experiments, 1):
        print("\n" + "="*80)
        print(f" EXPERIMENT {idx}/{len(experiments)}: {exp['exp_name']}")
        print("="*80)
        
        try:
            run_experiment(
                df=df,
                exp_name=exp['exp_name'],
                pos_labels=exp['pos_labels'],
                neg_labels=exp['neg_labels'],
                config_params=config,
                feature_config=feature_config
            )
            print(f"\nExperiment {idx}/{len(experiments)} completed successfully!")
            
        except Exception as e:
            print(f"\nERROR in experiment {idx}/{len(experiments)}: {exp['exp_name']}")
            print(f"   Details: {str(e)}")
            import traceback
            traceback.print_exc()
            print("   Continuing to the next experiment...\n")
            continue
        
        finally:
            # Clear memory after each experiment
            print(f"\nClearing memory after experiment {idx}...")
            clear_memory()
            print("-"*80)
        
    # Final summary and report
    print("\n" + "="*80)
    print(" ALL EXPERIMENTS COMPLETED")
    print("="*80)
    
    if Path('all_metrics.csv').exists():
        results_df = pd.read_csv('all_metrics.csv')
        print(f"\nTotal evaluated models: {len(results_df)}")
        print("\nBest results summary per experiment:")
        print("-"*80)
        
        # Group by experiment and get best model
        for exp_name in results_df['experiment'].unique():
            exp_data = results_df[results_df['experiment'] == exp_name]
            best_row = exp_data.loc[exp_data['test_bal_acc_mean'].idxmax()]
            
            print(f"\n{exp_name}:")
            print(f"   Best model: {best_row['model']}")
            # print(f"   Validation - Bal_Acc: {best_row['val_bal_acc_mean']:.4f} ± {best_row['val_bal_acc_std']:.4f}")
            print(f"   Test      - Bal_Acc: {best_row['test_bal_acc_mean']:.4f} ± {best_row['test_bal_acc_std']:.4f}")
            print(f"   Test      - ROC-AUC: {best_row['test_roc_auc_mean']:.4f} ± {best_row['test_roc_auc_std']:.4f}")
        
        print("\n" + "-"*80)
        print("\nDetailed results saved in: 'all_metrics.csv'")
        print("Pipelines saved in: 'saved_pipelines'")
        print("SHAP analyses saved in: 'shap_analysis_results'")
        print("Confusion matrices saved in: 'confusion_matrices'")
        print("Fold metrics saved in: 'metrics_per_fold'")
    
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE!")
    print("="*80 + "\n")

# =========================================================================== #

if __name__ == "__main__":

    # Remove existing metrics file (fresh start)
    if Path('all_metrics.csv').exists():
        print("\n   Removing previous 'all_metrics.csv'...")
        os.remove('all_metrics.csv')

    if Path('shap_features_cv_evaluation.csv').exists():
        print("\n   Removing previous 'shap_features_cv_evaluation.csv'...")
        os.remove('shap_features_cv_evaluation.csv')


    dirs = [
        'saved_pipelines', 
        'data_splits', 
        'best_params', 
        'confusion_matrices',
        'metrics_per_fold',
        'shap_analysis_results',
        'shap_analysis_results/plots/summary_plots',
        'shap_analysis_results/plots/performance_curves',
        'shap_analysis_results/feature_importance_data'
    ]

    for d in dirs:
        path = Path(d)
        if not path.exists():
            continue
        for file in path.rglob("*"):
            if file.is_file():
                print(f"Removing {file}")
                file.unlink()

    # Run the main function to execute all experiments
    main()

# =========================================================================== #