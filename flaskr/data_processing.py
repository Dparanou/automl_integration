from flask import (Blueprint, request)
from flaskr.db import get_db
import datetime

from flaskr.classes.data_processor import Data

bp = Blueprint('data', __name__)

# Just add 6 rows to the database
@bp.route('/init', methods=('GET', 'POST'))
def init():
    db = get_db()
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-01 00:00:00','%Y-%m-%d %H:%M:%S'), 8)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-01 12:00:00','%Y-%m-%d %H:%M:%S'), 12)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-01 17:00:00','%Y-%m-%d %H:%M:%S'), 15)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-02 00:00:00','%Y-%m-%d %H:%M:%S'), 17)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-02 12:00:00','%Y-%m-%d %H:%M:%S'), None)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-03 00:00:00','%Y-%m-%d %H:%M:%S'), 10)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-03 12:00:00','%Y-%m-%d %H:%M:%S'), 20)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-04 00:00:00','%Y-%m-%d %H:%M:%S'), -2)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-04 12:00:00','%Y-%m-%d %H:%M:%S'), 19)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-05 00:00:00','%Y-%m-%d %H:%M:%S'), 10)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-05 12:00:00','%Y-%m-%d %H:%M:%S'), 14)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-06 00:00:00','%Y-%m-%d %H:%M:%S'), 8)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-06 12:00:00','%Y-%m-%d %H:%M:%S'), 17)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-07 00:00:00','%Y-%m-%d %H:%M:%S'), 11)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-07 12:00:00','%Y-%m-%d %H:%M:%S'), 23)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-08 00:00:00','%Y-%m-%d %H:%M:%S'), 15)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-08 12:00:00','%Y-%m-%d %H:%M:%S'), 18)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, target_value) VALUES (?, ?)',
        (datetime.datetime.strptime('2020-01-09 00:00:00','%Y-%m-%d %H:%M:%S'), 12)
    )
    db.commit()
    return 'OK'

@bp.route('/check_db', methods=('GET', 'POST'))
def check_db():
    db = get_db()
    # Get the name of the table
    table_name = 'timeseries'
    # Execute the PRAGMA command to get information about the table
    result = db.execute("PRAGMA table_info({})".format(table_name)).fetchall()

    print()
    # Print the results
    for row in result:
        print("Column name:", row[1])
        print("Column type:", row[2])
    
    print()

    # Get all the data from the table
    result = db.execute("SELECT * FROM {}".format(table_name)).fetchall()

    ts = [tuple(row) for row in result]

    return ts

# http://127.0.0.1:5000/data_processing
@bp.route('/data_processing')
def get_data():
    # Get config file
    config = {"start_date": "2020-01-01", 
              "end_date": "2020-01-09", 
              "value": "target_value", 
              "clean_data": True,
              "time_interval": "12H",
              "test_perc": 0,
              "val_perc": 0,
              "are_temporal_features_enabled": True,
              "are_past_features_enabled": False,
              "are_derivative_features_enabled": True,
              "temporal_features": ["hour", "day", "month", "year", "dayofweek", "day_of_year", "week_of_year", 'is_weekend', 'is_working_hour'],
              "past_features": [{
                    # "last_12_hours": ["actual", "mean", "max", "min"],
                    "last_day": ["actual", "mean", "max", "min"],
                    # "last_week": ["actual", "mean", "max", "min"],
              }],
              "derivative_features": ["slope", "curvature"],
              "future_predictions": 3
              }
    
    start_date = datetime.datetime.strptime(config['start_date'], '%Y-%m-%d')
    # Add 1 day to the end date to include all the info about the last day
    end_date = datetime.datetime.strptime(config['end_date'], '%Y-%m-%d') + datetime.timedelta(days=1)
    value = config['value']
    clean_data = config['clean_data']

    data = Data(start_date=start_date, end_date=end_date, value=value)

    # Clean data
    if clean_data:
        data.clean_data(value, config['time_interval'])

    # Split data
    data.split_data(test_perc=config['test_perc'], val_perc=config['val_perc'])
    
    # Generate features
    data.generate_features(config)

    # Normalize data
    data.normalize_data(config)

    # One hot encoding for categorical features
    data.one_hot_encode()

    # Split data to features and target
    data.split_data_to_features_and_target(config['value'])

    # Shift target column
    data.shift_target(config['future_predictions'])

    # Printe data attributes
    # print(data.__dict__.keys())

    return 'Data ok'
