import json
import pandas as pd
import numpy as np
from datetime import datetime
import pyarrow as pa
from influxdb_client import InfluxDBClient

from matplotlib import pyplot as plt

from data_processor import Data, generate_features_new_data
from models import XGBRegressor, LGBMRegressor, LinearRegressor
from search_methods import RandomSearch

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

models = ['XGBoost', 'LGBM', 'Linear']

class Trainer:
  def __init__(self, config_dict, target) -> None:
    self.model = None
    self.data = None
    self.config = config_dict
    self.results = {}


    influx_cloud_url = 'http://localhost:8086/'
    influx_cloud_token = 'gJExfQxYqEI5cCRa26wSWkUUdyn9nmF-f34nlfcBGGHUEM3YzYYWlgDDkcvoewrYSKBW6QE9A9Y7bvCy0zwTPg=='
    bucket = 'more'
    org = 'Athena'
    kind = 'bebeze'
    field = target
    # get the features of the target value from ["features"]["columnFeatures"]
    extra_columns = next((column["features"] for column in config_dict["features"]["columnFeatures"] if column["columnName"] == target), [])

    # define the start and end date of the data that we want to get from influx in time format
    start_date = datetime.fromisoformat('2018-01-02T00:00:00Z'[:-1]).isoformat() + 'Z'
    end_date = datetime.fromisoformat('2018-01-27T02:00:00Z'[:-1]).isoformat() + 'Z'
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
    
    
    # print(f'Querying from InfluxDB cloud: "{query}" ...')
    query_api = client.query_api()
    result = query_api.query(query=query, org=org)

    # add target column to the results as dictionary keys
    keys = [target] + extra_columns
    results = {key: [] for key in keys}

    for table in result:
      for record in table.records:
        results[record['_field']].append({record['_start']: record['_value']})

    # convert the results to dataframe format, where each key is a column
    df = pd.DataFrame({col: [list(record.values())[0] for record in results[col]] for col in results})
    # Set the datetime as the index
    df.set_index(pd.DatetimeIndex([list(record.keys())[0] for record in results[target]]), inplace=True)
    # print(df)
    # call the init_data function to initialize the data
    self.init_data(df)

    # exit()
    self.train()

    # get the results of the model
    self.get_results()

    # unscale the predictions
    y_test_unscaled = self.data.target_scaler.inverse_transform(self.data.test_y)
    # print(y_test_unscaled[:,0])
    # print(len(y_test_unscaled[:,0]))

    # exit()
    # plot the predictions
    plt.figure(figsize=(25,8))
    plt.plot(y_test_unscaled[:,0], label='True')
    plt.plot([row[1] for row in self.get_results()[list(self.get_results().keys())[0]]['y_pred_test']], label='Predicted')
    plt.xlabel('Time')
    plt.legend()
    plt.title('Predictions')

    plt.savefig('test.png')


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

    # print(self.data.columns)

  def train(self):
    # Get model type
    for model_name in self.config['algorithms'].keys():
        params = self.config['algorithms'][model_name]

        if model_name == 'XGBoost':
            self.model = XGBRegressor(**params)
        elif model_name == 'LGBM':
            self.model = LGBMRegressor(**params)
        elif model_name == 'Linear':
            self.model = LinearRegressor(**params)

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
                
                self.model.fit(self.data.train_X, self.data.train_y, **best_params)
            else:
                self.model.fit(self.data.train_X, self.data.train_y, **search_method.get_best_params())
        else:
            self.model.fit(self.data.train_X, self.data.train_y)
        
        y_pred_train = self.model.predict(self.data.train_X)
        y_pred_test = self.model.predict(self.data.test_X)
        evaluation = self.model.evaluation_metrics(self.data.test_y, y_pred_test)

        # Unscaled data
        y_pred_train = self.data.target_scaler.inverse_transform(y_pred_train)
        y_pred_test = self.data.target_scaler.inverse_transform(y_pred_test)

        self.results[model_name + "_" + self.target] = {}
        self.results[model_name + "_" + self.target]['y_pred_train'] = y_pred_train.tolist()
        self.results[model_name + "_" + self.target]['y_pred_test'] = y_pred_test.tolist()
        self.results[model_name + "_" + self.target]['evaluation'] = evaluation
      
  def get_results(self):
    return self.results


def predict(timestamp, past_metrics, config_dict, model, target):
  '''
  Predicts the target value for the given timestamp
  timestamp: timestamp for which the prediction is made
  past_metrics: past metrics for the given timestamp
  config_dict: config dictionary
  target: target column name
  '''
  # First, create empty target column and set timestamp as index
  timestamp = pd.DataFrame(timestamp)
  timestamp.index = pd.to_datetime(timestamp.index)
  timestamp[target] = [0 for i in range(len(timestamp))]

  # Generate the features for the given timestamp based on the model's features
  X = generate_features_new_data(df = timestamp, 
                                 config = config_dict, 
                                 past_metrics = past_metrics)

  # Infer Arrow schema from pandas
  # schema = pa.Schema.from_pandas(df)


if "__main__" == __name__:
  # Load config file
  with open("config.json", "r") as json_file:
    config_dict = json.load(json_file)
  
  # Load data
  # df = pd.read_parquet('../data/data.parquet').set_index('daytime')
  # # Convert index to datetime
  # df.index = pd.to_datetime(df.index)

  # # Create a dataframe with timeseries
  # data = pd.DataFrame()
  # dt_index = pd.date_range(
  #         start='2020-01-31', end='2020-02-01', freq='5T')
  # data['date'] = dt_index
  # data = data.set_index('date')
  # # Keep only the first 1 row
  # data = data[:1]

  # model=""
  for target in config_dict['targetColumn']:
    trainer = Trainer(config_dict, target)
