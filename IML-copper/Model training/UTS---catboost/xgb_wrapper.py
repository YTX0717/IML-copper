from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd

# XGBWrapper is a wrapper for XGBoost, used for training and prediction, eliminating compatibility issues, allowing direct use of xgboost under python3.12 and the latest numpy library
class XGBWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 n_estimators=100, 
                 learning_rate=0.1, 
                 max_depth=3, 
                 subsample=1.0,
                 colsample_bytree=1.0,
                 gamma=0,
                 reg_alpha=0,
                 reg_lambda=1,
                 random_state=None,
                 nthread=-1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.nthread = nthread
        self._estimator_type = "regressor"
        self._feature_importances_ = None  # Add private attribute to store feature importance

    def fit(self, X, y):
        self.model_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            nthread=self.nthread,
            importance_type='weight'
        )
        self.model_.fit(X, y)
        self._feature_importances_ = self.model_.feature_importances_  # Set private attribute
        return self


    def plot_feature_importance(self, feature_names=None, max_num_features=20):
        """Visualize feature importance"""
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.feature_importances_))]
            
        # Sort by importance
        indices = np.argsort(self.feature_importances_)[::-1]
        
        # Draw bar chart
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.barh(range(min(len(indices), max_num_features)), 
                self.feature_importances_[indices[:max_num_features]][::-1], 
                color='b', align='center')
        plt.yticks(range(min(len(indices), max_num_features)), 
                  np.array(feature_names)[indices[:max_num_features]][::-1])
        plt.xlabel("Relative Importance")
        plt.gca().invert_yaxis()  # Display importance from high to low
        plt.show()

    def predict(self, X):
        return self.model_.predict(X)

    @property
    def feature_importances_(self):
        return self._feature_importances_  # Return private attribute

    # ... existing get_params and set_params methods ...

# SHAPXWrapper is a subclass of XGBWrapper, used to calculate SHAP importance and perform feature selection, this subclass can be used together with BORUTA_SHAPWrapper for feature selection
class SHAPXWrapper(XGBWrapper):
    def __init__(self, max_depth=None, n_estimators=100, random_state=42, **kwargs):
        super().__init__(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state, **kwargs)
        self._shap_importances_ = None  # Use private attribute to store SHAP importance

    def fit(self, X, y):
        # If X is numpy.ndarray, convert to pandas.DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

    # Split training and validation sets (e.g., 80% training, 20% validation)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
            )
        super().fit(X_train, y_train)  # Train only on training set
    
    # Calculate SHAP values on validation set
        explainer = shap.TreeExplainer(self.model_)
        shap_values = explainer.shap_values(X_val)
        self._shap_importances_ = np.abs(shap_values).mean(axis=0)
        return self
    
    # Calculate SHAP values on k-fold validation, final selection results will differ from validation set
    # def _clone_model(self):
    #     """Create a copy of the current model"""
    #     clone = SHAPXWrapper(
    #     max_depth=self.max_depth,
    #     n_estimators=self.n_estimators,
    #     random_state=self.random_state
    #     )
    # # Can copy other parameters if needed
    #     return clone
    
    # def fit(self, X, y):
    # # Initialize SHAP values storage list
    #     all_shap_values = []
    
    # # Create 5-fold cross-validation splitter
    #     kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    #     for train_index, val_index in kf.split(X):
    #     # Split training and validation sets
    #         X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    #         y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
    #     # Clone new model to prevent parameter contamination between cross-validations
    #         fold_model = self._clone_model()  # Need to implement model cloning method
    #         fold_model.fit(X_train, y_train)
        
    #     # Calculate SHAP values for this fold
    #         explainer = shap.TreeExplainer(fold_model.model_)
    #         fold_shap_values = explainer.shap_values(X_val)


    #         all_shap_values.append(fold_shap_values)
        
    #     # Calculate global average SHAP importance
    #     self._shap_importances_ = np.mean(all_shap_values, axis=0)
    #     # Must retrain final model on complete data
    #     super().fit(X, y)
    #     return self

    @property
    def feature_importances_(self):
        return self._shap_importances_  # Return SHAP importance
# Ensure SHAPXWrapper can be imported
__all__ = ['XGBWrapper', 'SHAPXWrapper']
# ... existing code ...

