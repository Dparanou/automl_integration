import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib

class XGBRegressor:
    def __init__(self, **kwargs):
        self.model = xgb.XGBRegressor(**kwargs)
    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_xgb_params(self):
        return self.model.get_xgb_params()
    
    def save_model(self, path):
        self.model.save_model(path)
    
    def evaluation_metrics(self, ytrue, ypred):
        results = {}
        results['MSE'] = mean_squared_error(ytrue, ypred)
        results['MAE'] = mean_absolute_error(ytrue, ypred)
        results['MAPE'] = mean_absolute_percentage_error(ytrue, ypred)*100
        results['RMSE'] = mean_squared_error(ytrue, ypred, squared=False)
        
        self.model.performance_metrics = results

        return self.model.performance_metrics
    
    def get_params_search_space(self):
        params_dict = {
            'booster': ['gbtree', 'gblinear', 'dart'],
            'learning_rate': np.arange(0.01, 0.5, 0.01),
            'max_depth': range(1, 25, 1),
            'min_child_weight': range(1, 10, 1),
            'gamma': np.arange(0.01, 0.5, 0.05),
            'lambda': np.arange(0.01, 0.5, 0.05),
            'alpha': np.arange(0.01, 0.5, 0.05),
            'colsample_bytree': np.arange(0.1, 1, 0.1),
            'n_estimators': range(100, 2000, 100),
            'n_jobs': -1
        }

        return params_dict 

class LGBMRegressor:
    def __init__(self, **kwargs):
        self.model = lgb.LGBMRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_lgbm_params(self):
        return self.model.get_params()
    
    def save_model(self, path):
        self.model.booster_.save_model(path)
    
    def evaluation_metrics(self, ytrue, ypred):
        results = {}
        results['MSE'] = mean_squared_error(ytrue, ypred)
        results['MAE'] = mean_absolute_error(ytrue, ypred)
        results['MAPE'] = mean_absolute_percentage_error(ytrue, ypred)*100
        results['RMSE'] = mean_squared_error(ytrue, ypred, squared=False)
        
        self.model.performance_metrics = results

        return self.model.performance_metrics

    def get_params_search_space(self):
        params_dict = {
            'boosting_type': ['gbdt', 'dart', 'rf'],
            'learning_rate': np.arange(0.01, 0.5, 0.01),
            'max_depth': range(1, 25, 1),
            'num_leaves': range(1, 100, 10),
            'min_child_weight': range(1, 10, 1),
            'reg_alpha': np.arange(0.01, 0.5, 0.05),
            'reg_lambda': np.arange(0.01, 0.5, 0.05),
            'colsample_bytree': np.arange(0.1, 1, 0.1),
            'n_estimators': range(100, 2000, 100),
            'n_jobs': -1
        }

        return params_dict
    
class LinearRegressor:
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_linear_params(self):
        return self.model.get_params()
    
    def save_model(self, path):
        joblib.dump(self.model, path)
    
    def evaluation_metrics(self, ytrue, ypred):
        results = {}
        results['MSE'] = mean_squared_error(ytrue, ypred)
        results['MAE'] = mean_absolute_error(ytrue, ypred)
        results['MAPE'] = mean_absolute_percentage_error(ytrue, ypred)*100
        results['RMSE'] = mean_squared_error(ytrue, ypred, squared=False)
        
        self.model.performance_metrics = results

        return self.model.performance_metrics
    
    def get_params_search_space(self):
        params_dict = {
            'fit_intercept': [True, False],
            'positive': [True, False],
            'n_jobs': -1
        }

        return params_dict