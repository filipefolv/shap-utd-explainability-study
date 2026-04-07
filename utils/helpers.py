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

import gc
import ast
import json
import torch
import logging
import pandas as pd

from imblearn.pipeline import Pipeline as ImbPipeline

# =========================================================================== #
# CONFIGURATION
# =========================================================================== #

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =========================================================================== #

def clear_memory():
    """
    Clear RAM and GPU memory between experiments.
    """
    # Force Python garbage collection
    gc.collect()
    
    # Clear GPU cache if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("Memory cleared")

# =========================================================================== #

def make_binary_subset(data, pos_labels, neg_labels, name):
    """
    Create a binary subset and map classes to 0 and 1.

    Args:
        data: Input DataFrame with English columns.
        pos_labels: Positive-class labels (list or string).
        neg_labels: Negative-class labels (list or string),
            or 'Others' for all labels except pos_labels.
        name: Experiment name.

    Returns:
        tuple: (DataFrame with 'target' column, experiment name).
    """
    # Label mapping dictionary
    label_map = {
        'Bladder Cancer': 'UrinaryBladder',
        'Prostate Cancer': 'Prostate',
        'Kidney Cancer': 'Kidney',
        'Uterus Cancer': 'Uterus',
        'Cystitis': 'Cystitis'
    }


    # Convert positive labels to list if input is a single string
    if isinstance(pos_labels, str):
        pos_labels = [pos_labels]
    print(pos_labels)
    # Map positive labels to dataset naming convention
    pos_labels_en = [label_map[label] for label in pos_labels]

    if neg_labels == 'Others':
        # Select all labels that are not in positive labels
        neg_labels_en = [label for label in data['label'].unique() if label not in pos_labels_en]
    else:
        # Convert negative labels to list if input is a single string
        if isinstance(neg_labels, str):
            neg_labels = [neg_labels]
        # Map negative labels to dataset naming convention
        neg_labels_en = [label_map[label] for label in neg_labels]

    # Build subset based on mapped labels
    subset = data[data['label'].isin(pos_labels_en + neg_labels_en)].copy()
    subset['target'] = subset['label'].apply(lambda x: 1 if x in pos_labels_en else 0)
    numeric_here = subset.select_dtypes(include="number").columns.difference(['target'])
    subset[numeric_here] = subset[numeric_here].astype("float32")
    return subset, name

# =========================================================================== #

def show_split_info(
        y_train: pd.Series,
        y_test: pd.Series,
        exp_name: str
    ) -> None:
        """Display information about train/test splits."""
        logger.info(f"\n=== Distribution: {exp_name} ===")
        for tag, y in [('Train', y_train), ('Test', y_test)]:
            vc = y.value_counts()
            pct = (vc / vc.sum() * 100).round(1)
            logger.info(f"{tag}: {vc.to_dict()} ({pct.to_dict()}%) - Total: {len(y)}")

# =========================================================================== #

# Function to get data after imputation
def get_imputed_data(pipeline, X_df):
    """
    Return data after the preprocessing ('prep') step.
    
    Args:
        pipeline: Trained (fitted) pipeline.
        X_df: Original input DataFrame to transform.
    
    Returns:
        pd.DataFrame: Data after 'prep' with correct column names.
    """
    # 1. Apply trained preprocessor to input data
    X_processed = pipeline.named_steps['prep'].transform(X_df)
    
    # 2. Get column names directly from preprocessor output
    column_names = pipeline.named_steps['prep'].get_feature_names_out()
    
    return pd.DataFrame(X_processed, columns=column_names)
  
# =========================================================================== #

# Function to get data after SMOTE
def get_smote_data(pipeline, X_train, y_train):
    """
    Simulate and return TRAIN data after preprocessing and SMOTE steps.
    This function is for training-process inspection only.
    
    Args:
        pipeline: Trained (fitted) pipeline.
        X_train: Original TRAIN DataFrame.
        y_train: Original TRAIN labels.
    
    Returns:
        (pd.DataFrame, pd.Series): Tuple with resampled X DataFrame
            and resampled y Series.
    """
    # 1. Apply trained preprocessor to train data
    X_processed = pipeline.named_steps['prep'].transform(X_train)
    
    # 2. Apply fit_resample here to inspect SMOTE effect
    X_resampled, y_resampled = pipeline.named_steps['smote'].fit_resample(X_processed, y_train)

    # 3. Column names are the same as generated by 'prep'
    column_names = pipeline.named_steps['prep'].get_feature_names_out()
    
    df_resampled = pd.DataFrame(X_resampled, columns=column_names)
    
    # Return labels as well to inspect class balancing
    return df_resampled, y_resampled

# =========================================================================== #

# Function to get final classifier input (post-processing/OHE)
def get_final_data_for_classifier(pipeline, X_train, y_train):
    """
    Return TRAIN data in its final format after all transformation
    and oversampling steps, reflecting the current pipeline structure.
    
    Args:
        pipeline: Trained (fitted) pipeline.
        X_train: Original TRAIN DataFrame.
        y_train: Original TRAIN labels.
    
    Returns:
        pd.DataFrame: Final feature DataFrame (X).
                                   
    """
    # 1. Create partial pipeline with all steps except final classifier
    processing_pipeline = ImbPipeline(pipeline.steps[:-1])
    
    # 2. Fit/transform partial pipeline on train data
    X_final_array = processing_pipeline.fit_transform(X_train, y_train)
    
    # 3. Get final feature names from original trained pipeline
    final_feature_names = pipeline.named_steps['ohe'].get_feature_names_out()
    
    # 4. Build final DataFrame
    df_final = pd.DataFrame(X_final_array, columns=final_feature_names)
    
    return df_final

# =========================================================================== #

def compare_train_val(pipeline, X_train, X_val, y_train, y_val):
    """
    Compare train and validation splits side by side.
    """
    pipeline.fit(X_train, y_train)
    
    # Pipeline without classifier
    processing = ImbPipeline(pipeline.steps[:-1])
    
    X_train_t = processing.transform(X_train)
    X_val_t = processing.transform(X_val)
    
    print("\n" + "="*70)
    print("COMPARISON: TRAIN vs VALIDATION")
    print("="*70)
    
    comparison = pd.DataFrame({
        'Metric': [
            'Original samples',
            'Samples after transformation',
            'Original features',
            'Features after transformation',
            'Class 0',
            'Class 1'
        ],
        'TRAIN': [
            X_train.shape[0],
            X_train_t.shape[0],
            X_train.shape[1],
            X_train_t.shape[1],
            (y_train == 0).sum(),
            (y_train == 1).sum()
        ],
        'VALIDATION': [
            X_val.shape[0],
            X_val_t.shape[0],
            X_val.shape[1],
            X_val_t.shape[1],
            (y_val == 0).sum(),
            (y_val == 1).sum()
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    return X_train_t, X_val_t

# =========================================================================== #

def load_shap_results(shap_summary_path="shap_analysis_results/final_summary_report.csv"):
    """Load SHAP analysis summary report."""
    df = pd.read_csv(shap_summary_path)
    print(f"\nSHAP summary loaded: {len(df)} models")
    return df

# =========================================================================== #

def load_best_hyperparameters(csv_path, experiment_name, model_name):
    """
    Load best hyperparameters from a CSV filtered by experiment and model.
    
    Args:
        csv_path (str): Path to CSV file (e.g., 'all_metrics.csv')
        experiment_name (str): Experiment name
        model_name (str): Model name ('XGBoost', 'LightGBM', 'CatBoost')
    
    Returns:
        dict: Dictionary with best hyperparameters
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Filter by experiment and model
    filtered = df[(df['experiment'] == experiment_name) & (df['model'] == model_name)]
    
    if filtered.empty:
        raise ValueError(f"No results found for experiment='{experiment_name}' and model='{model_name}'")
    
    if len(filtered) > 1:
        print("Warning: Multiple results found. Using the first one.")
    
    # Extract best_trial_params (stored as string)
    best_params_str = filtered.iloc[0]['best_trial_params']
    
    # Convert string to dictionary
    try:
        best_params = ast.literal_eval(best_params_str)
    except:
        import json
        best_params = json.loads(best_params_str.replace("'", '"'))
    
    return best_params
