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
import numpy as np
import pandas as pd


from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import  StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.featureSelector import NoFeatureSelector

# =========================================================================== #
# CONFIGURATION
# =========================================================================== #

SEED = 42
np.random.seed(SEED)

# =========================================================================== #

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

class AdaptiveOHETransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies OHE only to categorical columns,
    detected dynamically by prefix.
    """
    def __init__(self, categorical_prefix='cat__', handle_unknown='ignore', 
                 sparse_output=False, drop='first'):
        self.categorical_prefix = categorical_prefix
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.drop = drop
        self.categorical_cols_ = None
        self.numeric_cols_ = None
        self.ohe_ = None
        self.feature_names_out_ = None
        
    def fit(self, X, y=None):
        """
        Detect categorical columns and fit OHE.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Split categorical and numeric columns
        self.categorical_cols_ = [col for col in X.columns if col.startswith(self.categorical_prefix)]
        self.numeric_cols_ = [col for col in X.columns if not col.startswith(self.categorical_prefix)]
        
        print("\nAdaptive OHE:")
        print(f"- Categorical: {len(self.categorical_cols_)} columns")
        print(f"- Numeric: {len(self.numeric_cols_)} columns")
        
        if len(self.categorical_cols_) > 0:
            # Create and fit OHE for categorical columns only
            self.ohe_ = ColumnTransformer(
                transformers=[
                    ('ohe', OneHotEncoder(
                        handle_unknown=self.handle_unknown,
                        sparse_output=self.sparse_output,
                        drop=self.drop
                    ), self.categorical_cols_)
                ],
                remainder='passthrough'  # Keep numeric columns
            )
            self.ohe_.fit(X)
            
            # Get final feature names
            self.feature_names_out_ = self.ohe_.get_feature_names_out()
        else:
            print("No categorical columns found, skipping OHE")
            self.feature_names_out_ = X.columns.tolist()
        
        return self
    
    def transform(self, X):
        """
        Apply OHE when categorical columns are present.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        if self.ohe_ is not None and len(self.categorical_cols_) > 0:
            # Apply OHE
            X_transformed = self.ohe_.transform(X)
            return pd.DataFrame(X_transformed, columns=self.feature_names_out_)
        else:
            # Without OHE, return as-is
            return X
    
    def get_feature_names_out(self, input_features=None):
        """
        Return feature names after OHE.
        """
        return self.feature_names_out_

# =========================================================================== #

def build_pipeline(trial, exp_name, model_name, X_df, feature_config, feature_selection="None"):
    """
    Build a complete pipeline with optional feature selection.
    
    Args:
        trial: Optuna trial
        exp_name: Experiment name (e.g., 'exp1')
        model_name: Model name ('XGBoost', 'LightGBM', 'CatBoost')
        X_df: Input DataFrame
        feature_config: Feature configuration
        feature_selection: FS type ('None')
    """
    available_cols = X_df.columns
    numeric_features = [c for c in feature_config.numeric if c in available_cols]
    categorical_features = [col for col in feature_config.categorical if col in available_cols]
    
    # Imputation + preprocessing
    knn_n_neighbors = trial.suggest_int("imputer__n_neighbors", 3, 30)

    cat_pipe = ImbPipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        # ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    num_pipe = ImbPipeline([
        ('imp', KNNImputer(n_neighbors=knn_n_neighbors)),
        ('scaler', StandardScaler())         

    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, numeric_features),
            ('cat', cat_pipe, categorical_features)
        ],
        sparse_threshold=0.0, 
    )

    # Oversampling (SMOTE)
    cat_indices_after_imputing = list(range(
        len(numeric_features), 
        len(numeric_features) + len(categorical_features)
    ))
    smote = SMOTENC(categorical_features=cat_indices_after_imputing, 
                    sampling_strategy='auto',
                    random_state=SEED
                    )
    
    intermediate_names = [f'num__{col}' for col in numeric_features] + \
                         [f'cat__{col}' for col in categorical_features]
    intermediate_cat_names = [name for name in intermediate_names if name.startswith('cat__')]

    # Feature Selection
    print(f"\n Configuring Feature Selection: {feature_selection}")
    
    feature_selector = NoFeatureSelector()

    # One-hot encoding
    ohe_adaptive = AdaptiveOHETransformer(
        categorical_prefix='cat__',
        handle_unknown='ignore',
        sparse_output=False,
        drop='first'
    )

    # Classifier
    if model_name == "XGBoost":
        clf = XGBClassifier(
            max_depth        = trial.suggest_int("clf__max_depth", 3, 10),
            n_estimators     = trial.suggest_int("clf__n_estimators", 120, 250),
            learning_rate    = trial.suggest_float("clf__lr", 0.01, 0.2, log=True),
            subsample        = trial.suggest_float("clf__subsample", 0.7, 0.95),
            colsample_bytree = trial.suggest_float("clf__colsample_bytree", 0.65, 0.95),
            min_child_weight = trial.suggest_float("clf__min_child_weight", 3, 10),
            gamma            = trial.suggest_float("clf__gamma", 0.3, 1.0),
            reg_alpha        = trial.suggest_float("clf__reg_alpha", 0.01, 1.0, log=True),
            reg_lambda       = trial.suggest_float("clf__reg_lambda", 0.1, 5.0, log=True),
            scale_pos_weight = trial.suggest_float("clf__scale_pos_weight", 0.5, 5.0),
            eval_metric      = "logloss",
            tree_method      = "hist",
            random_state     = SEED,
            n_jobs           = -1,
            # device           = "cuda"
        )

    elif model_name == "LightGBM":
        clf = LGBMClassifier(
            max_depth        = trial.suggest_int('clf__max_depth', 3, 10),
            num_leaves       = trial.suggest_int('clf__num_leaves', 20, 50),
            n_estimators     = trial.suggest_int('clf__n_estimators', 50, 200),
            learning_rate    = trial.suggest_float("clf__lr", 0.01, 0.2, log=True),
            subsample        = trial.suggest_float('clf__subsample', 0.7, 0.9),
            colsample_bytree = trial.suggest_float('clf__colsample_bytree', 0.65, 0.95),
            min_child_samples = trial.suggest_int('clf__min_child_samples', 10, 25),
            reg_lambda       = trial.suggest_float('clf__reg_lambda', 0.01, 0.5),
            lambda_l1        = trial.suggest_float('clf__lambda_l1', 0.001, 2.0, log=True),
            lambda_l2        = trial.suggest_float('clf__lambda_l2', 0.1, 5.0, log=True),
            min_split_gain   = trial.suggest_float('clf__min_split_gain', 0.0, 0.5),
            random_state     = SEED,
            verbose          = -1,
            # device           = 'gpu'
)

    else:  # catboost
        clf = CatBoostClassifier(
            depth            = trial.suggest_int("clf__depth", 5, 15),
            learning_rate    = trial.suggest_float("clf__lr", 0.01, 0.2, log=True),
            iterations       = trial.suggest_int("clf__iters", 100, 200, step=20),
            l2_leaf_reg      = trial.suggest_float("clf__l2_leaf_reg", 4.0, 8.0),
            bagging_temperature = trial.suggest_float("clf__bagging_temperature", 0.1, 5.0),
            border_count     = trial.suggest_int("clf__border_count", 32, 255),
            random_strength  = trial.suggest_float("clf__random_strength", 0.5, 2.0),
            verbose          = False,
            random_state     = SEED,
            # task_type        = 'GPU'
        )

    # Final Pipeline
    pipe = ImbPipeline(steps=[
        ('prep', preprocessor),                                       # 1. Imputation + scaling
        ('smote', smote),                                             # 2. Oversampling
        ('to_df', ArrayToDataFrame(column_names=intermediate_names)), # 3. Convert to DataFrame
        ('feature_selection', feature_selector),                      # 6. Feature Selection
        ('ohe', ohe_adaptive),                                        # 4. One-Hot Encoding
        ('clf', clf)                                                  # 7. Classifier
    ])
    
    return pipe

# =========================================================================== #

def objective(trial, model_name, X_train, y_train, feature_config, feature_selection="None"):
    pipeline = build_pipeline(trial, model_name, X_train, 
                              feature_config, feature_selection)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        try:
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            score = float(balanced_accuracy_score(y_val, y_pred))
            scores.append(score)
        except ValueError as e:
            print(f"Error in fold ({model_name}): {e}")
            # Clear memory and continue
            gc.collect()
            continue

    # Clear memory after trial
    del pipeline
    gc.collect()
    
    if len(scores) == 0:
        return -1
    
    return np.mean(scores)