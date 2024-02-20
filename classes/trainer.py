import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyarrow as pa
from influxdb_client import InfluxDBClient
from matplotlib import pyplot as plt
import joblib
import pymongo
import configparser
from sklearn.preprocessing import MinMaxScaler
import pytz

from classes.data_processor import Data, generate_features_new_data
from classes.models import XGBRegressor, LGBMRegressor, LinearRegressor
from classes.search_methods import RandomSearch

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
config = configparser.ConfigParser()
config.read("/data/1/more-visual-files/more-workspace/config/settings.cfg")
models = ['XGBoost', 'LGBM', 'LinearRegression']

# Define the InfluxDB cloud parameters
# influx_cloud_url = 'http://localhost:8086/'
# influx_cloud_token = 'gJExfQxYqEI5cCRa26wSWkUUdyn9nmF-f34nlfcBGGHUEM3YzYYWlgDDkcvoewrYSKBW6QE9A9Y7bvCy0zwTPg=='
influx_cloud_url = config['DEFAULT']['influx_url']
influx_cloud_token = config['DEFAULT']['token']
bucket = config['DEFAULT']['bucket']
org = config['DEFAULT']['org']

mongoUri = config['DEFAULT']['mongo_uri_py']

class Trainer:
  def __init__(self, config_dict, target) -> None:
    self.models = {}
    self.data = None
    self.config = config_dict
    self.results = {}
    kind = config_dict['kind']
   
    field = target
    # get the features of the target value from ["features"]["columnFeatures"]
    extra_columns = next((column["features"] for column in config_dict["features"]["columnFeatures"] if column["columnName"] == target), [])

    # define the start and end date of the data that we want to get from influx in time format
    start_date = datetime.fromtimestamp(self.config['startDate'] / 1000).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date = datetime.fromtimestamp(self.config['endDate']/ 1000).strftime('%Y-%m-%dT%H:%M:%SZ')
    time_interval = config_dict['time_interval']

    client = InfluxDBClient(url=influx_cloud_url, token=influx_cloud_token, org=org)

    # create a query in influx to get the acti_power data that start from 2018-01-03 00:00:00 to 2018-01-06 00:00:00
    query = f'from(bucket: "{bucket}") \
      |> range(start: {start_date}, stop: {end_date})\
      |> filter(fn: (r) => r._measurement == "{kind}")\
      |> filter(fn:(r) => r._field == "{field}" '
    
    for extra_column in extra_columns:
      query += f'or r._field == "{extra_column}" '

    query += f')\
      |> window(every: {time_interval})\
      |> mean()'
    
    query_api = client.query_api()
    result = query_api.query(query=query, org=org)

    # add target column to the results as dictionary keys
    keys = [target] + extra_columns
    results = {key: [] for key in keys}

    for table in result:
      for record in table.records:
        results[record['_field']].append({record['_start']: record['_value']})

    # convert the results to dataframe format, where each key is a column
    self.df = pd.DataFrame({col: [list(record.values())[0] for record in results[col]] for col in results})
    # Set the datetime as the index
    self.df.set_index(pd.DatetimeIndex([list(record.keys())[0] for record in results[target]]), inplace=True)
    # call the init_data function to initialize the data

    # Shift the target column one value down - so as to predict the t+1 values - and remove the NaN value
    self.df[target] = self.df[target].shift(-1)
    self.df.dropna(inplace=True)

    # TODO: check issue with shifted predictions
    # plot the predictions
    # plt.figure(figsize=(25,8))
    # plt.plot(y_test_unscaled[:,0], label='True')
    # plt.plot([row[0] for row in self.get_results()[list(self.get_results().keys())[0]]['y_pred_test']], label='Predicted')
    # plt.xlabel('Time')
    # plt.legend()
    # plt.title('Predictions of ' + target)

    # plt.savefig('test.png')


  def start(self):
    self.init_data(self.df)

    self.train()

  def init_data(self, data_table):
    self.data = Data(data_table, self.config["time_interval"])
    self.target = data_table.columns[0]
    
    self.data.set_data(self.data.get_all_data()[self.target].to_frame())

    # Clean data
    self.data.clean_data(self.target)

    # Generate the time features
    self.data.generate_time_features(self.config)

    # One hot encode the time features
    self.data.one_hot_encode_time_features(self.config)

    # # Add column features to the data
    self.data.add_column_features(self.config, self.target)

    # Generate features
    self.data.generate_features(self.config, self.target)
    
    # Split data
    self.data.split_data(val_perc=self.config['dataSplit'][1]/100, test_perc=self.config['dataSplit'][2]/100)

    # Normalize data
    self.data.normalize_data(self.config, self.target)

    # Split data to features and target
    self.data.split_data_to_features_and_target(self.target)

    # Shift target column
    self.data.shift_target(self.config['future_predictions'])

    # Save column names of data
    self.data.columns = self.data.train_X.columns

  def train(self):
    # Get model type
    for model_name in self.config['algorithms'].keys():
        params = self.config['algorithms'][model_name]

        if model_name == 'XGBoost':
            self.models[model_name] = XGBRegressor(**params)
        elif model_name == 'LGBM':
            self.models[model_name] = LGBMRegressor(**params)
        elif model_name == 'LinearRegression':
            self.models[model_name] = LinearRegressor(**params)

        if isinstance(params[list(params.keys())[1]], list):
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
            
            search_method = RandomSearch(self.data.train_X, model_name, search_params)
            search_method.fit(self.data.train_X, self.data.train_y)

            if model_name == 'LGBM':
                # Remove the prefix estimator__ from the best params
                best_params = {}
                for param in search_method.get_best_params().keys():
                    best_params[param.split('__')[1]] = search_method.get_best_params()[param]
                
                self.models[model_name].fit(self.data.train_X, self.data.train_y, **best_params)
            else:
                self.models[model_name].fit(self.data.train_X, self.data.train_y, **search_method.get_best_params())
        else:
            self.models[model_name].fit(self.data.train_X, self.data.train_y)
        
        # y_pred_train = self.models[model_name].predict(self.data.train_X)
        y_pred_test = self.models[model_name].predict(self.data.test_X)
        evaluation = self.models[model_name].evaluation_metrics(self.data.test_y, y_pred_test)

        # Unscaled data
        # y_pred_train = self.data.target_scaler.inverse_transform(y_pred_train)[:,0]
        y_pred_test = self.data.target_scaler.inverse_transform(y_pred_test)[:,0]


        self.results[model_name + "_" + self.target] = {}
        # self.results[model_name + "_" + self.target]['predictions'] = pa.Table.from_pandas(temp_df)
        # Assing the predictions to the results as a dictionary
        # Keys are timestamps and values are the predictions
        # convert timestamps to int
        # self.results[model_name + "_" + self.target]['predictions'] = dict(zip(np.array([timestamp.timestamp() for timestamp in self.data.train_X.index.tolist()]), y_pred_train.tolist()))
        self.results[model_name + "_" + self.target]['predictions'] = dict(zip(np.array([timestamp.timestamp() for timestamp in self.data.test_X.index.tolist()]).astype(int).astype(str), y_pred_test.tolist()))
        self.results[model_name + "_" + self.target]['evaluation'] = evaluation

  def save_model(self, model_type, model_name, target):
    aggr_dict = {}
    # save the model
    model_path = self.models[model_type].save_model(model_name)

    # Add information about the model
    aggr_dict['model_type'] = model_type
    aggr_dict['model_name'] = model_name
    aggr_dict['model_path'] = model_path
    aggr_dict['target'] = target
    aggr_dict['time_interval'] = self.config['time_interval']
    aggr_dict['features'] = self.config['features']
    aggr_dict['kind'] = self.config['kind']

    # Add the scaler and convert NaN to 0
    scaler_min = self.data.scaler.min_
    # scaler_min[np.isnan(scaler_min)] = 0
    scaler_scale = self.data.scaler.scale_
    # scaler_scale[np.isnan(scaler_scale)] = 1

    # save the scaler information 
    aggr_dict['scaler'] = {}
    aggr_dict['scaler']['min'] = scaler_min.tolist()
    aggr_dict['scaler']['scale'] = scaler_scale.tolist()
    aggr_dict['target_scaler'] = {}
    aggr_dict['target_scaler']['min'] = self.data.target_scaler.min_.tolist()
    aggr_dict['target_scaler']['scale'] = self.data.target_scaler.scale_.tolist()
    aggr_dict['feature_names'] = self.data.columns.tolist()

    # After preparing the dictionary, save it to the MongoDB
    client = pymongo.MongoClient(mongoUri)
    db = client['more']
    collection = db['meta']

    # Insert the document into the collection
    result = collection.insert_one(aggr_dict)
    msg = ''
    # Check if the insertion was successful
    if result.acknowledged:
        print("Insertion successful. Inserted document ID:", result.inserted_id)
        msg = 'Model saved successfully'
    else:
        print("Insertion failed.")
        msg = 'Model saving failed'

    # Close the connection to the MongoDB database
    client.close()

    return msg

  def get_results(self):
    return self.results

def predict(timestamp, date_df, model_name, db_kind):
  '''
  Predicts the target value for the given timestamp
  timestamp: timestamp for which the prediction is made
  config_dict: config dictionary that include model_type, model_name and target
  '''
  # Load the model if exist or stop the process
  if load_model_and_config(model_name=model_name) is None:
    return None
  else:
    model, config_dict = load_model_and_config(model_name=model_name)

  # Get the feature names from the model
  features = config_dict['feature_names']

  # First, create empty target column and set timestamp as index
  date_df[config_dict['target']] = [0 for i in range(len(date_df))]

  # Get the column Features - if exits
  general_features = pd.DataFrame()
  for column_data in config_dict['features']["columnFeatures"]:
      if column_data["columnName"] == config_dict['target']:
          general_features = get_general_features(timestamp, column_data["features"], config_dict, db_kind)
          break
  
  # Get the past metrics from the influxdb based on the enabled metrics in the config file
  past_metrics = pd.DataFrame()
  if 'pastMetrics' in config_dict['features']['optionalFeatures']:
    past_metrics = get_past_values(date_df, config_dict, db_kind)

  # Generate the features for the given timestamp based on the model's features
  X = generate_features_new_data(df = date_df, 
                                 config = config_dict, 
                                 past_metrics = past_metrics,
                                 features = features)
  
  # Add the general features to the dataframe - if exists
  if not general_features.empty:
    X = pd.concat([X, general_features], axis=1)

  # Drop the target column
  X = X.drop(config_dict['target'], axis = 1)

  # Sort the columns alphabetically
  X = X.reindex(sorted(X.columns), axis=1)

  # Scale the features
  scaler = MinMaxScaler()
  target_scaler = MinMaxScaler()
  # assign the scaler data and convert to numbers
  scaler.min_ = [float(element) for element in config_dict['scaler']['min']] 
  scaler.scale_ = [float(element) for element in config_dict['scaler']['scale']]
  target_scaler.min_ = config_dict['target_scaler']['min']
  target_scaler.scale_ = config_dict['target_scaler']['scale']

  X_scaled = scaler.transform(X)

  # Predict the target value
  y_pred = model.predict(X_scaled)

  # Unscale the target value and convert to dataframe
  y_pred = target_scaler.inverse_transform(y_pred)[0]

  time_interval = config_dict['time_interval']
  next_timestamps = []
  
  # Calculate the next 5 timestamps with 30-minute intervals
  for i in range(1, len(y_pred)+1):  # i will be from 1 to 5 (inclusive)
    # Calculate the next timestamp by adding the time interval to the previous timestamp
    if time_interval[-1] == 'm':
      next_timestamp = datetime.fromtimestamp(timestamp / 1000) + timedelta(minutes=int(time_interval[:-1]) * i)
    elif time_interval[-1] == 'h':
      next_timestamp = datetime.fromtimestamp(timestamp / 1000) + timedelta(hours=int(time_interval[:-1]) * i)
    elif time_interval[-1] == 'd':
      next_timestamp = datetime.fromtimestamp(timestamp / 1000) + timedelta(days=int(time_interval[:-1]) * i)

    # Append the next timestamp as a Unix timestamp to the list
    next_timestamps.append(int(next_timestamp.timestamp()))

  # convert predictions to dictionary - keys: timestamps & values: predictions
  predictions = dict(zip(np.array([timestamp for timestamp in next_timestamps]).astype(str), y_pred.tolist()))
  # y_pred = {'timestamp': y_pred}

  # # Convert pandas to pyarrow table schema
  # y_pred = pa.Table.from_pandas(y_pred)
  # schema = y_pred.schema

  # # Serialize the schema
  # schema_serialized = schema.serialize().to_pybytes()

  return predictions

def load_model_and_config(model_name):
  '''
  Loads the model and config from the folder by getting the path from mongoDB
  model_name: name of the model - XGBoost_active_power.json

  Returns: model and config dictionary
  '''
  # Connect to the MongoDB database
  client = pymongo.MongoClient(mongoUri)
  db = client['more']
  collection = db['meta']

  # Get the model information from the MongoDB
  model_info = collection.find_one({'model_name': model_name})

  # Close the connection to the MongoDB database
  client.close()
  
  # Check that model exists in the db
  if model_info is None:
    return None

  # Otherwise get the model
  if model_info['model_type'] == 'XGBoost':
    model = XGBRegressor()
    model.load_model(model_info['model_path'])

  elif model_info['model_type'] == 'LGBM' or model_info['model_type'] == 'LinearRegression':
    model = joblib.load(model_info['model_path'])
     
  return model, model_info

def get_past_values(timestamp, config_dict, db_kind):
  '''
  Get the past values from the influxdb - only the desired timestamps
  '''
  client = InfluxDBClient(url=influx_cloud_url, token=influx_cloud_token, org=org)

  past_metrics = []
  categories = list(config_dict['features']['optionalFeatures']['pastMetrics'].keys())
  for category in categories:
    category_metrics = config_dict['features']['optionalFeatures']['pastMetrics'][category]
    if len(category_metrics) != 0:
      # create a query in influx to get the target data that start from X - 3hours to X, where X is the timestamp
      if category[4:] == 'Hour':
        start_date = timestamp.index[0] - pd.Timedelta(hours = 3)
        end_date = timestamp.index[0]
        
      elif category[4:] == 'Day':
        start_date = timestamp.index[0] - pd.Timedelta(days = 1) - pd.Timedelta(hours = 3)
        end_date = timestamp.index[0] - pd.Timedelta(days = 1)

      elif category[4:] == 'Week':
        start_date = timestamp.index[0] - pd.Timedelta(weeks = 1) - pd.Timedelta(hours = 3)
        end_date = timestamp.index[0] - pd.Timedelta(weeks = 1)

      elif category[4:] == 'Month':
        start_date = timestamp.index[0] - pd.Timedelta(months = 1) - pd.Timedelta(hours = 3)
        end_date = timestamp.index[0] - pd.Timedelta(months = 1)

      start_date = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
      # add also one time interval the end date in order to include the last value
      end_date = end_date + pd.Timedelta(minutes = 30) 
      end_date = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
      
      query = f'from(bucket: "{bucket}") \
          |> range(start: {start_date}, stop: {end_date})\
          |> filter(fn: (r) => r._measurement == "{db_kind}")\
          |> filter(fn:(r) => r._field == "{config_dict["target"]}" )\
          |> window(every: {config_dict["time_interval"]})\
          |> mean()'
      
      query_api = client.query_api()
      result = query_api.query(query=query, org=org)

      for table in result:
        for record in table.records:
           # append the past metric to the array with its timestamp
          past_metrics.append({"timestamp": record['_start'], config_dict['target']: record['_value']})
      
  past_metrics = pd.concat([pd.DataFrame(columns=['timestamp', config_dict['target']]), pd.DataFrame(past_metrics)], ignore_index=True)
  past_metrics['timestamp'] = pd.to_datetime(past_metrics['timestamp'])
  past_metrics = past_metrics.set_index('timestamp') # set the timestamp as index
  past_metrics = past_metrics.sort_index() # sort the index

  return past_metrics 

def get_general_features(timestamp, features_array, config_dict, db_kind):
    client = InfluxDBClient(url=influx_cloud_url, token=influx_cloud_token, org=org)

    start_date = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%dT%H:%M:%SZ')
    # start_date = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    # end_date = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date = datetime.fromtimestamp(timestamp/1000) + pd.Timedelta(hours = 1)
    end_date = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    query = f'from(bucket: "{bucket}") \
      |> range(start: {start_date}, stop: {end_date})\
      |> filter(fn: (r) => r._measurement == "{db_kind}")\
      |> filter(fn:(r) => '
    
    for extra_column in features_array:
      query += f'r._field == "{extra_column}" or '
    
    # Remove the last or
    query = query[:-4]
    query += ')'
    
    query_api = client.query_api()
    result = query_api.query(query=query, org=org)

    results = []
    for table in result:
      for record in table.records:
        results.append({"timestamp": record['_time'], record['_field']: record['_value']})

    df = pd.DataFrame(results)
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_convert(None)

    # Group the DataFrame to combine entries with the same timestamp into a single row
    df = df.groupby('timestamp').sum(min_count=1)

    # Keep only the first row of the results in case of multiple values
    df = df.head(1)

    # Replace the timestamp with the given from user
    df.index.values[0] = pd.to_datetime(datetime.fromtimestamp(timestamp/1000))

    return df
