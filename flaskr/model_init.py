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
    config = {
        'model_type': 'lgbm',
        'fine_tune': True,
        'fine_tune_technique': 'grid',
        'save_model': False,
        'save_model_path': 'flaskr/models/model.pkl'
    }

    # Read data from json
    with open('data.json') as f:
        data = json.load(f)
    
    data = pd.read_json(data['train'], orient='split')
    X = data.drop('target_value', axis=1)
    y = data['target_value']
    # Get model type
    model = models[config['model_type']]

    if config['fine_tune']:
        search_method = search_methods[config['fine_tune_technique']](X, model.get_model(), model.get_params_search_space(config['fine_tune_technique']))
        search_method.fit(X, y)

    
    return 'Model initialized'