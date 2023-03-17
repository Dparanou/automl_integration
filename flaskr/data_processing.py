from flask import (Blueprint, request)
from flaskr.db import get_db
import datetime
import pandas as pd
import json

from flaskr.data_preparation.data_processor import Data, generate_temporal_features, normalize_data, one_hot_encode, separate_X_y, shifted_target

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
              "are_derivative_features_enabled": False,
              "temporal_features": ["hour", "day", "month", "year", "dayofweek", "dayofyear", "weekofyear", "quarter"],
              "past_features": [{
                    # "last_12_hours": ["actual", "mean", "max", "min"],
                    "last_day": ["actual", "mean", "max", "min"],
                    # "last_week": ["actual", "mean", "max", "min"],
              }],
              "derivative_features": ["slope", "curvature"],}
    
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

    # # Export data to json
    # data.export_data("data.json", type='json')

    return 'Data ok'

# http://127.0.0.1:5000/split?test=0.1&val=0.1
@bp.route('/split')
def split_dataset():
    args = request.args
    test_percentage = float(args.getlist('test')[0])
    validation_percentage = float(args.getlist('val')[0])

    data = Data()
    # Read json as dataframe and set it as data
    data.set_data(pd.read_json('data.json', orient='split'))


    data.split_data(test_percentage, validation_percentage)

    aggr_dict = {}

    aggr_dict['train'] = data.train.to_json(orient='split')
    aggr_dict['test'] = data.test.to_json(orient='split')
    aggr_dict['validation'] = data.val.to_json(orient='split')

    json_object = json.dumps(aggr_dict, indent = 4) 
    # Write to file
    with open("data.json", "w") as outfile: 
        outfile.write(json_object)

    return 'Split ok'

@bp.route('/features')
def feature_generation():
    train = Data()
    test = Data()
    val = Data()

    # Read json
    with open('data.json') as json_file:
        data_dict = json.load(json_file)

    # Set data
    train.set_data(pd.read_json(data_dict['train'], orient='split'))
    test.set_data(pd.read_json(data_dict['test'], orient='split'))
    val.set_data(pd.read_json(data_dict['validation'], orient='split'))

    conf = {'week_of_year': True,
            'weekday': True,
            'day': True,
            'month': True,
            'hour': True,
            'minute': True,
            'is_working_hour': True,
            'is_weekend': True
            }
    
    train.set_data(generate_temporal_features(train.get_data(), conf))
    test.set_data(generate_temporal_features(test.get_data(), conf))
    val.set_data(generate_temporal_features(val.get_data(), conf))

    aggr_dict = {}
    aggr_dict['train'] = train.get_data().to_json(orient='split')
    aggr_dict['test'] = test.get_data().to_json(orient='split')
    aggr_dict['validation'] = val.get_data().to_json(orient='split')

    json_object = json.dumps(aggr_dict, indent = 4) 
    # Write to file
    with open("data.json", "w") as outfile: 
        outfile.write(json_object)

    return 'Features ok'

@bp.route('/normalization')
def data_normalization():
    train = Data()
    test = Data()
    val = Data()

    # Read json
    with open('data.json') as json_file:
        data_dict = json.load(json_file)

    # Set data
    train.set_data(pd.read_json(data_dict['train'], orient='split'))
    test.set_data(pd.read_json(data_dict['test'], orient='split'))
    val.set_data(pd.read_json(data_dict['validation'], orient='split'))

    categorical_features = ['weekday', 'hour', 'minute']
    # Normalize data
    train_scaled, val_scaled, test_scaled, scaler, target_scaler = normalize_data(
        train.get_data(), val.get_data(), test.get_data(), 'target_value', categorical_features)
    
    train.set_data(train_scaled)
    test.set_data(test_scaled)
    val.set_data(val_scaled)

    aggr_dict = {}
    aggr_dict['train'] = train.get_data().to_json(orient='split')
    aggr_dict['test'] = test.get_data().to_json(orient='split')
    aggr_dict['validation'] = val.get_data().to_json(orient='split')

    json_object = json.dumps(aggr_dict, indent = 4) 
    # Write to file
    with open("data.json", "w") as outfile: 
        outfile.write(json_object)

    return 'Normalization ok'

# http://127.0.0.1:5000/encode_categorical?encode_values=weekday,hour,minute
@bp.route('/encode_categorical')
def encode_categorical():
    args = request.args
    encode_values = args.getlist('encode_values')[0].split(',')

    train = Data()
    test = Data()
    val = Data()

    # Read json
    with open('data.json') as json_file:
        data_dict = json.load(json_file)

    # Set data
    train.set_data(pd.read_json(data_dict['train'], orient='split'))
    test.set_data(pd.read_json(data_dict['test'], orient='split'))
    val.set_data(pd.read_json(data_dict['validation'], orient='split'))

    # Encode categorical features
    train_encoded = one_hot_encode(train.get_data(), encode_values)
    test_encoded = one_hot_encode(test.get_data(), encode_values)
    val_encoded = one_hot_encode(val.get_data(), encode_values)

    train.set_data(train_encoded)
    test.set_data(test_encoded)
    val.set_data(val_encoded)

    aggr_dict = {}
    aggr_dict['train'] = train.get_data().to_json(orient='split')
    aggr_dict['test'] = test.get_data().to_json(orient='split')
    aggr_dict['validation'] = val.get_data().to_json(orient='split')

    json_object = json.dumps(aggr_dict, indent = 4) 
    # Write to file
    with open("data.json", "w") as outfile: 
        outfile.write(json_object)

    return 'Encode ok'

# http://127.0.0.1:5000/shift?target=target_value&window=3
@bp.route('/shift')
def shift_data():
    args = request.args
    target = args.get('target')
    window = int(args.get('window'))

    train = Data()
    test = Data()
    val = Data()

    # Read json
    with open('data.json') as json_file:
        data_dict = json.load(json_file)

    # Set data
    train.set_data(pd.read_json(data_dict['train'], orient='split'))
    test.set_data(pd.read_json(data_dict['test'], orient='split'))
    val.set_data(pd.read_json(data_dict['validation'], orient='split'))

    # Shift data
    X_train, y_train_shifted = shifted_target(*separate_X_y(train.get_data(), target), window)

    # train.set_data(train_shifted)
    # test.set_data(test_shifted)
    # val.set_data(val_shifted)

    # aggr_dict = {}
    # aggr_dict['train'] = train.get_data().to_json(orient='split')
    # aggr_dict['test'] = test.get_data().to_json(orient='split')
    # aggr_dict['validation'] = val.get_data().to_json(orient='split')

    # json_object = json.dumps(aggr_dict, indent = 4) 
    # # Write to file
    # with open("data.json", "w") as outfile: 
    #     outfile.write(json_object)

    return 'Shift ok'