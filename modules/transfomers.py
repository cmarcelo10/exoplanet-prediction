import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NeutralImputer(TransformerMixin, BaseEstimator):
    '''Combines scaling and imputing into one step and returns the original, unscaled data.
      Only applicable for tree-based models'''

    def __init__(self, scaler, imputer):
        self.scaler = scaler
        self.imputer = imputer

    def fit(self, X, y = None, **fit_params):
        ## params are unused, and only added for compatibility
        self.scaler.fit(X)
        temp = self.scaler.transform(X)
        self.imputer.fit(temp)
        return self
    
    def transform(self, X, y = None, **transform_params):
        # params added for compatibility. Unused
        X_transform = self.scaler.transform(X)
        X_transform = self.imputer.transform(X_transform)
        return self.scaler.inverse_transform(X_transform)
    
    def fit_transform(self, X, y = None, **fit_params):
        # params don't do anything
        self.fit(X, y, **fit_params)
        return self.transform(X, y, transform_params=None)
    
## This was never used in the final submission.
class SmartFunctionTransfomer(TransformerMixin, BaseEstimator):
    '''Special function transformer that allows for specific columns to be referenced
    in the transforming function. This is different from the regular FunctionTransformer
    which uses Numpy arrays. Does not define inverse transform.'''
    def __init__(self, func, feature_names_in):
        self.feature_names_in = feature_names_in
        self.func = func
        
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, y=None):
        return self.func(pd.DataFrame(X, columns=self.feature_names_in))
        
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
