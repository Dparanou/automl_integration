from flask import (Blueprint, request)

from flaskr.classes.models import XGBRegressor, LGBMRegressor, LinearRegressor

bp = Blueprint('model', __name__)

# Initialize the model
@bp.route('/init_model', methods=('GET', 'POST'))
def init_model():
    config = {
        'model_type': 'xgboost',
        'fine_tune': False,
        'fine_tune_technique': 'bayesian',
        'save_model': False,
        'save_model_path': 'flaskr/models/model.pkl'
    }
    if config['model_type'] == 'xgboost':
        model = XGBRegressor()
    elif config['model_type'] == 'lgbm':
        model = LGBMRegressor()
    elif config['model_type'] == 'linear':
        model = LinearRegressor()
    else:
        print('Model type not recognized')
        return 'Model type not recognized'
    
    return 'Model initialized'