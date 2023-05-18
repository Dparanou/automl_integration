from flask import (Blueprint, request)

from flaskr.classes.models import XGBRegressor, LGBMRegressor, LinearRegressor
from flaskr.classes.search_methods import RandomSearch
import json
import pandas as pd
import numpy as np

bp = Blueprint('model', __name__)

models = ['XGBoost', 'LGBM', 'Linear']

# Initialize the model
@bp.route('/init_model', methods=('GET', 'POST'))
def init_model():
    # Get config file
    with open('config.json') as json_file:
        config = json.load(json_file)

    results = {}
    for target in config['targetColumn']:
        # Read data from json
        with open('./Data/data_' + target + '.json') as f:
            data = json.load(f)
        
        X_train = pd.read_json(data['train_X'], orient='split')
        y_train = pd.read_json(data['train_y'], orient='split')

        X_train = pd.read_json(data['train_X'], orient='split')
        y_train = pd.read_json(data['train_y'], orient='split')

        X_test = pd.read_json(data['test_X'], orient='split')
        y_test = pd.read_json(data['test_y'], orient='split')

        # Get model type
        for model_name in config['algorithms'].keys():
            params = config['algorithms'][model_name]

            if model_name == 'XGBoost':
                model = XGBRegressor(**params)
            elif model_name == 'LGBM':
                model = LGBMRegressor(**params)
            elif model_name == 'Linear':
                model = LinearRegressor(**params)

            if isinstance(params[list(params.keys())[0]], list):
                # Get the search params
                search_params = {}
                for param in params.keys():
                    # check if array contain numbers and if yes then create a range
                    if isinstance(params[param][0], int) or isinstance(params[param][0], float):
                        # If model is LGBM then add the prefix estimator__
                        if model_name == 'LGBM':
                            search_params['estimator__' + param] = np.arange(params[param][0], params[param][1], params[param][2])
                        else:
                            search_params[param] = np.arange(params[param][0], params[param][1], params[param][2])
                    else:
                        if model_name == 'LGBM':
                            search_params['estimator__' + param] = params[param]
                        else:
                            search_params[param] = params[param]
                
                search_method = RandomSearch(X_train, model_name, search_params)
                search_method.fit(X_train, y_train)

                if model_name == 'LGBM':
                    # Remove the prefix estimator__ from the best params
                    best_params = {}
                    for param in search_method.get_best_params().keys():
                        best_params[param.split('__')[1]] = search_method.get_best_params()[param]
                    
                    model.fit(X_train, y_train, **best_params)
                else:
                    model.fit(X_train, y_train, **search_method.get_best_params())
            else:
                model.fit(X_train, y_train,)
            
            y_pred = model.predict(X_test)
            evaluation = model.evaluation_metrics(y_test, y_pred)
            
            results[model_name + "_" + target] = {}
            results[model_name + "_" + target]['y_pred'] = y_pred
            results[model_name + "_" + target]['evaluation'] = evaluation

    # print(results)
    return 'Models are ready'