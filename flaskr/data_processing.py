from flask import (Blueprint, request)
from flaskr.db import get_db
import datetime
import json

from flaskr.classes.data_processor import Data

bp = Blueprint('data', __name__)

# Just add 6 rows to the database
@bp.route('/init', methods=('GET', 'POST'))
def init():
    db = get_db()
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-01 00:00:00','%Y-%m-%d %H:%M:%S'), 24, 113, 8)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-01 12:00:00','%Y-%m-%d %H:%M:%S'), 24, 143, 12)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-01 17:00:00','%Y-%m-%d %H:%M:%S'), 24, 113, 15)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-02 00:00:00','%Y-%m-%d %H:%M:%S'), 28, 123, 17)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-02 12:00:00','%Y-%m-%d %H:%M:%S'), 20, 113, None)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-03 00:00:00','%Y-%m-%d %H:%M:%S'), 34, 113, 10)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-03 12:00:00','%Y-%m-%d %H:%M:%S'), 24, 149, 20)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-04 00:00:00','%Y-%m-%d %H:%M:%S'), 21, 100, -2)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-04 12:00:00','%Y-%m-%d %H:%M:%S'), 18, 143, 19)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-05 00:00:00','%Y-%m-%d %H:%M:%S'), 24, 113, 10)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-05 12:00:00','%Y-%m-%d %H:%M:%S'), 38, 141, 14)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-06 00:00:00','%Y-%m-%d %H:%M:%S'), 42, 123, 8)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-06 12:00:00','%Y-%m-%d %H:%M:%S'), 24, 113, 17)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-07 00:00:00','%Y-%m-%d %H:%M:%S'), 32, 153, 11)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-07 12:00:00','%Y-%m-%d %H:%M:%S'), 20, 119, 23)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-08 00:00:00','%Y-%m-%d %H:%M:%S'), 18, 113, 15)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-08 12:00:00','%Y-%m-%d %H:%M:%S'), 14, 113, 18)
    )
    db.execute(
        'INSERT INTO timeseries (daytime, wind_speed, pitch_angle, target_value) VALUES (?, ?, ?, ?)',
        (datetime.datetime.strptime('2020-01-09 00:00:00','%Y-%m-%d %H:%M:%S'), 29, 125, 12)
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
    with open('config.json') as json_file:
        config = json.load(json_file)

    # Convert timestamp to datetime
    config['startDate'] = datetime.datetime.fromtimestamp(config['startDate'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
    config['endDate'] = datetime.datetime.fromtimestamp(config['endDate'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
    config['time_interval'] = "12H"
    # Get the columns name that needed to be fetched from the database
    all_columns = []
    config['future_predictions'] = 3

    for features in config['features']['columnFeatures']:
        # Check if the feature already exist in the columns list
        for item in features['features']:
            if item not in all_columns:
                all_columns.append(item)

    # concat the 2 list of futures and targets
    all_columns = all_columns + config['targetColumn']

    # Get the data from the database
    # data = Data(start_date=config['startDate'], end_date=config['endDate'], columns=columns)
    data = Data(start_date="2020-01-01", end_date="2020-01-09", columns=all_columns)

    # config = {"start_date": "2020-01-01", 
    #           "end_date": "2020-01-09", 
    #           "value": "target_value", 
    #           "time_interval": "12H",
    #           "test_perc": 0.3,
    #           "val_perc": 0,
    #           "are_temporal_features_enabled": True,
    #           "are_past_features_enabled": False,
    #           "are_derivative_features_enabled": True,
    #           "temporal_features": ["hour", "day", "month", "year", "dayofweek", "day_of_year", "week_of_year", 'is_weekend', 'is_working_hour'],
    #           "past_features": [{
    #                 # "last_12_hours": ["actual", "mean", "max", "min"],
    #                 "last_day": ["actual", "mean", "max", "min"],
    #                 # "last_week": ["actual", "mean", "max", "min"],
    #           }],
    #           "derivative_features": ["slope", "curvature"],
    #           "future_predictions": 3
    #           }
    for target in config['targetColumn']:
        print(target)
        data.set_data(data.get_all_data()[target].to_frame())

        # Generate the time features
        data.generate_time_features(config)

        # One hot encode the time features
        data.one_hot_encode_time_features(config)

        # Add column features to the data
        data.add_column_features(config, target)

        # Split data
        data.split_data(val_perc=config['dataSplit'][1]/100, test_perc=config['dataSplit'][2]/100)
        
        # Generate features
        data.generate_features(config, target)

        # Normalize data
        data.normalize_data(config, target)

        # Split data to features and target
        data.split_data_to_features_and_target(target)

        # Shift target column
        data.shift_target(config['future_predictions'])
        
    return 'OK'
        
    
    # Add 1 day to the end date to include all the info about the last day
    # end_date = datetime.datetime.strptime(config['end_date'], '%Y-%m-%d') + datetime.timedelta(days=1)
    # value = config['value']

    # 

    # # data.check_data_quality(target=config['value'], TIME_INTERVAL=config['time_interval'])

    # # Clean data
    # data.clean_data(value, config['time_interval'])

    # # Save data to json
    # data.export_data("data.json", "json")

    # # Print data attributes
    # # print(data.__dict__.keys())

    return 'Data ok'
