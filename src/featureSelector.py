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

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NoFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Dummy transformer that performs no selection (passes all features through).
    Useful for comparison with other feature selection methods.
    Stores feature names for reference.
    Returns original feature names in get_feature_names_out.
    """
    def __init__(self):
        self.feature_names_ = None
        self.selected_feature_names_ = None

    def fit(self, X, y=None):
        """
        Store feature names during fitting.
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.selected_feature_names_ = self.feature_names_

        print(f"\n No Feature Selection (using all {X.shape[1]} features)")
        return self
    
    def transform(self, X):
        """
        Return X without modification.
        """
        return X
    
    def get_feature_names_out(self, input_features=None):
        """
        Return original feature names.
        """
        if self.feature_names_ is not None:
            return self.feature_names_
        elif input_features is not None:
            return input_features
        else:
            return None
