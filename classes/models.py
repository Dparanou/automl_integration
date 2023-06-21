import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
from skopt.space import Real, Categorical, Integer

# Inherit from the base class
class XGBRegressor:
    def __init__(self, **kwargs):
        self.model = xgb.XGBRegressor(**kwargs)
    
    def get_model(self):
        return self.model
    
    def fit(self, X, y,  **kwargs):
        # Check if kwargs is not empty
        if kwargs:
            self.model.set_params(**kwargs)
        
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_xgb_params(self):
        return self.model.get_xgb_params()

    def get_feature_names(self):
        return self.model.feature_names_in_
    
    def save_model(self, model_name, target):
        self.model.save_model(model_name + "_" + target + '.json')
    
    def load_model(self, model_path):
        self.model.load_model(fname=model_path)

    def evaluation_metrics(self, ytrue, ypred):
        results = {}
        results['MSE'] = mean_squared_error(ytrue, ypred)
        results['MAE'] = mean_absolute_error(ytrue, ypred)
        results['MAPE'] = mean_absolute_percentage_error(ytrue, ypred)*100
        results['RMSE'] = mean_squared_error(ytrue, ypred, squared=False)
        
        self.model.performance_metrics = results

        return self.model.performance_metrics
    
    def get_params_search_space(self, search_technique):
        if search_technique == 'bayesian':
            params_dict = {
                'booster': Categorical(['gbtree', 'gblinear', 'dart']),
                'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
                'max_depth': Integer(1, 25),
                'min_child_weight': Integer(1, 10),
                'gamma': Real(0.01, 0.5, prior='log-uniform'),
                'lambda': Real(0.01, 0.5, prior='log-uniform'),
                'alpha': Real(0.01, 0.5, prior='log-uniform'),
                'colsample_bytree': Real(0.1, 1, prior='log-uniform'),
                'n_estimators': Integer(100, 2000),
            }
        else:
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
            }

        return params_dict 

class LGBMRegressor:
    def __init__(self, **kwargs):
        self.model = lgb.LGBMRegressor(**kwargs)
    
    def get_model(self):
        return self.model

    def fit(self, X, y, **kwargs):
        # Check if kwargs is not empty
        if kwargs:
            self.model.set_params(**kwargs)

        # Check if multiple values have to be fitted
        if len(y.shape) > 1:
            if y.shape[1] > 1:
                self.model = MultiOutputRegressor(self.model).fit(X, y)
        else:
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_lgbm_params(self):
        return self.model.get_params()
    
    def save_model(self, model_name, target):
        joblib.dump(self.model, model_name + "_" + target + '.joblib')
    
    def evaluation_metrics(self, ytrue, ypred):
        results = {}
        results['MSE'] = mean_squared_error(ytrue, ypred)
        results['MAE'] = mean_absolute_error(ytrue, ypred)
        results['MAPE'] = mean_absolute_percentage_error(ytrue, ypred)*100
        results['RMSE'] = mean_squared_error(ytrue, ypred, squared=False)
        
        self.model.performance_metrics = results

        return self.model.performance_metrics

    def get_params_search_space(self, search_technique):
        if search_technique == 'bayesian':
            params_dict = {
                'estimator__boosting_type': Categorical(['gbdt', 'dart', 'rf']),
                'estimator__learning_rate': Real(0.01, 0.5, 'log-uniform'),
                'estimator__max_depth': Integer(1, 25),
                'estimator__num_leaves': Integer(1, 100),
                'estimator__min_child_weight': Integer(1, 10),
                'estimator__reg_alpha': Real(0.01, 0.5, 'log-uniform'),
                'estimator__reg_lambda': Real(0.01, 0.5, 'log-uniform'),
                'estimator__colsample_bytree': Real(0.1, 1, 'log-uniform'),
                'estimator__n_estimators': Integer(100, 2000),
                'estimator__bagging_freq': Integer(1, 10),
                'estimator__bagging_fraction': Real(0.1, 0.99, 'uniform')
            }
        else:
            params_dict = {
                'estimator__boosting_type': ['gbdt', 'dart', 'rf'],
                'estimator__learning_rate': np.arange(0.01, 0.5, 0.01),
                'estimator__max_depth': range(1, 25, 1),
                'estimator__num_leaves': range(1, 100, 10),
                'estimator__min_child_weight': range(1, 10, 1),
                'estimator__reg_alpha': np.arange(0.01, 0.5, 0.05),
                'estimator__reg_lambda': np.arange(0.01, 0.5, 0.05),
                'estimator__colsample_bytree': np.arange(0.1, 1, 0.1),
                'estimator__n_estimators': range(100, 2000, 100),
                'estimator__bagging_freq': range(1, 10),
                'estimator__bagging_fraction': np.arange(0.1, 0.99, 0.05)
            }
        
        return params_dict
    
class LinearRegressor:
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)
    
    def get_model(self):
        return self.model

    def fit(self, X, y, **kwargs):
        # Check if kwargs is not empty
        if kwargs:
            self.model.set_params(**kwargs)
        
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_linear_params(self):
        return self.model.get_params()
    
    def save_model(self, model_name, target):
        joblib.dump(self.model, model_name + "_" + target + '.joblib')
    
    def evaluation_metrics(self, ytrue, ypred):
        results = {}
        results['MSE'] = mean_squared_error(ytrue, ypred)
        results['MAE'] = mean_absolute_error(ytrue, ypred)
        results['MAPE'] = mean_absolute_percentage_error(ytrue, ypred)*100
        results['RMSE'] = mean_squared_error(ytrue, ypred, squared=False)
        
        self.model.performance_metrics = results

        return self.model.performance_metrics
    
    def get_params_search_space(self, search_technique):
        if search_technique == 'bayesian':
            params_dict = {
                'fit_intercept': Categorical([True, False]),
                'positive': Categorical([True, False]),
            }
        else:
            params_dict = {
                'fit_intercept': [True, False],
                'positive': [True, False],
            }

        return params_dict