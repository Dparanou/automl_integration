from flask import (Blueprint, request)

from flaskr.classes.models import XGBRegressor, LGBMRegressor, LinearRegressor
from flaskr.classes.search_methods import BayesSearch, GridSearch, RandomSearch
import json
import pandas as pd

bp = Blueprint('model', __name__)

models = {
    'xgboost': XGBRegressor(),
    'lgbm': LGBMRegressor(),
    'linear': LinearRegressor()
}

search_methods = {
    'bayesian': BayesSearch,
    'grid': GridSearch,
    'random': RandomSearch
}

# Initialize the model
@bp.route('/init_model', methods=('GET', 'POST'))
def init_model():
    config = [
        {
        'model_type': 'lgbm',
        'fine_tune': False,
        'fine_tune_technique': 'grid',
        'save_model': False,
        'save_model_path': 'flaskr/models/model.pkl'
    },
    {
        'model_type': 'xgboost',
        'fine_tune': False,
        'fine_tune_technique': 'bayesian',
        'save_model': False,
        'save_model_path': 'flaskr/models/model.pkl'
    },
    {
        'model_type': 'linear',
        'fine_tune': False,
        'fine_tune_technique': 'grid',
        'save_model': False,
        'save_model_path': 'flaskr/models/model.pkl'
    }
    ]

    # Read data from json
    with open('data.json') as f:
        data = json.load(f)
    
    X_train = pd.read_json(data['train_X'], orient='split')
    y_train = pd.read_json(data['train_y'], orient='split')

    X_test = pd.read_json(data['test_X'], orient='split')
    y_test = pd.read_json(data['test_y'], orient='split')


    # print(X_train.columns)
    # print(X_test.columns)
    # print(X_train.shape)
    # print(X_test.shape)

    # Get model type
    for model_config in config:
        model = models[model_config['model_type']]

        if model_config['fine_tune']:
            search_method = search_methods[config['fine_tune_technique']](X, model.get_model(), model.get_params_search_space(config['fine_tune_technique']))
            search_method.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        evaluation = model.evaluation_metrics(y_test, y_pred)
        
        print()
        print(model_config['model_type'])
        print(evaluation)

    return 'Model initialized'