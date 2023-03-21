from flask import (Blueprint, request)

from flaskr.classes.models import XGBoost

bp = Blueprint('model', __name__)

# Initialize the model
@bp.route('/init_model', methods=('GET', 'POST'))
def init_model():
    args = request.args
    model_type = args.get('model')
    if model_type == 'xgboost':
        model = XGBoost()
    return model