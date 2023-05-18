from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from skopt import BayesSearchCV
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

models = {
    'XGBoost': xgb.XGBRegressor(),
    'LGBM': MultiOutputRegressor(lgb.LGBMRegressor()),
    'Linear': LinearRegression(),
}

class GridSearch:
    def __init__(self, data, estimator, param_grid) :
        self.tuner = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv= TimeSeriesSplit(n_splits=2).split(data),
            scoring='neg_mean_squared_error',
            refit=True,
            n_jobs=-1
        )

    def fit(self, X, y):
        self.search = self.tuner.fit(X, y)

        # Save the best model, best parameters, and best score
        self.best_model = self.search.best_estimator_
        self.best_params = self.search.best_params_
        self.best_score = self.search.best_score_
    
    def get_best_params(self):
        return self.best_params
    
    def get_best_score(self):
        return self.best_score
    
    def get_best_model(self):
        return self.best_model
    
    def save_model(self, path):
        joblib.dump(self.best_model, path)

class RandomSearch:
    def __init__(self, data, estimator_name, param_distributions) :
        self.tuner = RandomizedSearchCV(
            estimator=models[estimator_name],
            param_distributions=param_distributions,
            cv= TimeSeriesSplit(n_splits=2).split(data),
            scoring='neg_mean_squared_error',
            n_iter=30,
            n_jobs=-1,
        )
    
    def fit(self, X, y):
        self.search = self.tuner.fit(X, y)

        # Save the best model, best parameters, and best score
        self.best_model = self.search.best_estimator_
        self.best_params = self.search.best_params_
        self.best_score = self.search.best_score_
    
    def get_best_params(self):
        return self.best_params
    
    def get_best_score(self):
        return self.best_score
    
    def get_best_model(self):
        return self.best_model
    
    def save_model(self, path):
        joblib.dump(self.best_model, path)

class BayesSearch:
    def __init__(self, data, estimator, param_distributions) :
        self.tuner = BayesSearchCV(
            estimator=estimator,
            search_spaces=param_distributions,
            cv= TimeSeriesSplit(n_splits=2).split(data),
            scoring='neg_mean_squared_error',
            n_iter=30,
            n_jobs=-1,
        )
    
    def fit(self, X, y):
        self.search = self.tuner.fit(X, y)

        # Save the best model, best parameters, and best score
        self.best_model = self.search.best_estimator_
        self.best_params = self.search.best_params_
        self.best_score = self.search.best_score_
    
    def get_best_params(self):
        return self.best_params
    
    def get_best_score(self):
        return self.best_score
    
    def get_best_model(self):
        return self.best_model
    
    def save_model(self, path):
        joblib.dump(self.best_model, path)