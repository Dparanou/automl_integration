import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import re

import warnings
warnings.filterwarnings('ignore')
# disable chained assignments
pd.options.mode.chained_assignment = None 

# Define the past features to be used in the generation of shifted features
past_features = {
    'prevHour': {
        'lookback': '1H',
        'window_size': 1,
    },
    'prev3Hour': {
        'lookback': '3H',
        'window_size': 3,
    },
    'prev12Hour': {
        'lookback': '12H',
        'window_size': 12,
    },
    'prevDay': {
        'lookback': '1D',
        'window_size': 24,
    },
    'prevWeek': {
        'lookback': '1W',
        'window_size': 168,
    },
    'prevMonth': {
        'lookback': '1M',
        'window_size': 720,
    },
}

time_intervals = {
    "m": "T",
    "h": "H",
    "d": "D",
    "w": "W",
    "M": "M",
}
class Data:
    def __init__(self, data: pd.DataFrame, time_interval):
        """
        Initialize the data.
        """
        self.all_data = data
        # Set the time interval
        self.time_interval = time_interval[:-1] + time_intervals[time_interval[-1]]

    def set_data(self, data):
        """
        Set the data.

        Parameters:
        data (pd.DataFrame): The data
        """
        self.data = data

    def update_set(self, selected_set, new_value):
        setattr(self, selected_set, new_value)
    
    def get_all_data(self):
        """
        Return the data.

        Returns:
        pd.DataFrame: The data
        """
        return self.all_data
    
    def get_data(self):
        """
        Return the data.

        Returns:
        pd.DataFrame: The data
        """
        return self.data

    def has_outliers(self, target):
        """
        Check if the data contains outliers and return a boolean value.

        Returns:
        bool: True if the data contains outliers, False otherwise
        """
        # Check if dataset contains outliers using IQR (interquartile range)
        q1 = self.data[target].quantile(0.25)  # 25th & 75th percentile
        q3 = self.data[target].quantile(0.75)
        IQR = q3 - q1  # Interquartile range
        outliers = self.data[target][((self.data[target] < (q1 - 1.5 * IQR)) |
                                      (self.data[target] > (q3 + 1.5 * IQR)))]  # Outliers
        # if outliers.shape[0] > 0:
        #     print('There is/are {:d} outlier/s in the dataset'.format(
        #         outliers.shape[0]))

        return outliers.shape[0] > 0

    def has_nan_values(self):
        """
        Check if the data contains NaN values and return a boolean value.

        Returns:
        bool: True if the data contains NaN values, False otherwise
        """
        return self.data.isnull().values.any()

    def has_incorrect_timestamps(self):
        """
        Check if the data contains incorrect timestamps and return a boolean value.

        Returns:
        bool: True if the data contains incorrect timestamps, False otherwise
        """
        # Check if the frequency of the records is in given interval
        incorrect_stamps = np.where(
            self.data.index.to_series().diff() != pd.Timedelta(self.time_interval))[0][1:]

        return len(incorrect_stamps) > 0

    def calc_perc_nan_values(self, target):
        """
        Calculate the percentage of NaN values in the data.

        Returns:
        float: The percentage of NaN values
        """
        return (self.data[target].isnull().sum() / self.data[target].shape[0]) * 100

    def calc_perc_incorrect_timestamps(self):
        """
        Calculate the percentage of incorrect timestamps in the data.

        Returns:
        float: The percentage of incorrect timestamps
        """
        # print(self.data.index.to_series().diff())
        incorrect_stamps = np.where(
            self.data.index.to_series().diff() != pd.Timedelta(self.time_interval))[0][1:]

        print('Incorrect stamps: ', incorrect_stamps.shape[0] / 2)
        return ((incorrect_stamps.shape[0] / 2) / self.data.shape[0]) * 100

    def check_data_quality(self, target):
        """
        Check if the data quality is good enough to be used for training.
        - Check percentage of dataset NaN values
        - Check percentage of timestamps not in the given frequency
        - Check if dataset contains outliers
        """

        max_perc_nan_values = 0.2
        max_perc_wrong_interval_values = 0.2

        if self.calc_perc_nan_values(target) > max_perc_nan_values * 100:
            print('Too many NaN values')
        if self.calc_perc_incorrect_timestamps(self.time_interval) > max_perc_wrong_interval_values * 100:
            print('Too many incorrect timestamps')
        if self.has_outliers(target):
            print('There are outliers in the dataset')
        if (self.calc_perc_nan_values(target) > max_perc_nan_values * 100
            and self.has_outliers(target)
                and self.calc_perc_incorrect_timestamps(self.time_interval) > max_perc_wrong_interval_values * 100):
            print('Data quality is not good enough')

    def remove_outliers(self, target):
        """
        Remove the outliers from the data, by replacing with NaN values
        Then, with fix nan values function, the NaN values will be replaced with the mean of the previous and next value
        """
        q1 = self.data[target].quantile(0.25)  # 25th & 75th percentile
        q3 = self.data[target].quantile(0.75)
        IQR = q3 - q1  # Interquartile range
        outliers = self.data[target][((self.data[target] < (q1 - 1.5 * IQR)) |
                                      (self.data[target] > (q3 + 1.5 * IQR)))]  # Outliers

        # Replace the outliers with NaN values
        self.data.loc[outliers.index, target] = np.nan

    def fill_nan_values(self, target):
        """
        Fill the NaN values with the mean of the previous and next value of the target column
        """
        self.data[target].interpolate(method='linear', inplace=True)

        # Fill the edge NaN values
        self.data[target].fillna(method='bfill', inplace=True)
        self.data[target].fillna(method='ffill', inplace=True)

    def fix_incorrect_timestamps(self):
        """
        Fix the frequency of the timestamps to be in given interval
        by removing the timestamps that are apart from the given interval
        """
        # Keep only the correct timestamps in given intervals
        self.data = self.data.resample(self.time_interval).first()

        # Sort the dataframe by the index
        self.data = self.data.sort_index()

        # Fill the NaN values if exist
        self.data.interpolate(method='linear', inplace=True)

    def is_within_20_percent(self, value1, value2):
        """
        Check if the difference between two values is within 20%
        """
        # Calculate the percentage difference
        perc_diff = (abs(value1 - value2) / value2) * 100
        # Check if the percentage difference is within 20%
        return abs(perc_diff) <= 20

    def plot_data(self, y, title):
        """
        Plot the data
        """
        plt.figure(figsize=(20, 10))
        plt.plot(self.data.index, self.data[y])
        plt.xlabel('Time')
        plt.ylabel(y)
        plt.title(title)
        plt.show()

    def split_data(self, val_perc, test_perc):
        """
        Split the data into train, validation and test sets
        """
        df_len = len(self.data)

        # Get the percentage of the train set
        train_perc = 1 - (test_perc + val_perc)

        n_train = int(train_perc * df_len)
        n_val = int(val_perc * df_len)
        n_test = int(test_perc * df_len)
        
        # Split the data into train, validation and test sets and assing them to the class attributes
        self.train = self.data.iloc[:n_train]
        self.val = self.data.iloc[n_train: n_train + n_val]
        self.test = self.data.iloc[n_train + n_val: n_train + n_val + n_test]
        
    def clean_data(self, target):
        """
        Clean the data by removing the outliers, filling NaN values and fixing the incorrect timestamps
        """
        self.remove_outliers(target)
        self.fill_nan_values(target)
        self.fix_incorrect_timestamps()

    def generate_time_features(self, config):
        """
        Generate the time features
        """
        # Generates features specified in the config file
        if 'temporal' in config['features']['optionalFeatures']:
            self.data = generate_temporal_features(self.data, config)

    def one_hot_encode_time_features(self, config):
        """
        One hot encode the time features
        """
        # Check if the categorical features are in the dataframe
        if 'temporal' in config['features']['optionalFeatures']:
            if set(config['features']['optionalFeatures']['temporal']).issubset(self.data.columns):
                self.data = pd.get_dummies(self.data, columns=config['features']['optionalFeatures']['temporal'])

    def add_column_features(self, config, target):
        features_list = None

        for item in config['features']['columnFeatures']:
            if item['columnName'] == target:
                features_list = item['features']
                break
        
        # Check if the features list is not empty
        if len(features_list) > 0:
            # Add the column features to the data 
            self.data = pd.concat([self.data, self.all_data[features_list]], axis=1)

    def generate_features(self, config, target):
        """
        Generate the features specified in the config file
        """
        if 'pastMetrics' in config['features']['optionalFeatures']:
            # Update the data with the generated features
            self.data = generate_metric_features(self.get_data(), config, target, self.time_interval)
        
        if 'derivatives' in config['features']['optionalFeatures']:
            self.data = generate_derivative_features(self.get_data(), config, target)

        # Sort the columns alphabetically
        self.data = self.data.reindex(sorted(self.data.columns), axis=1)


    def normalize_data(self, config, target):
        """
        Normalize the data using MinMaxScaler from sklearn
        """
        # Get the features
        features = getattr(self, 'train').columns

        # Get the numerical features (all features except the categorical and target features)
        numerical_features = list((set(features) - set(config['features']['optionalFeatures']['temporal'])))
        numerical_features.remove(target)
        # Sort the features
        numerical_features.sort()
        # Normalize the numerical features
        scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        self.train[numerical_features] = scaler.fit_transform(self.train[numerical_features])
        self.train[target] = target_scaler.fit_transform(self.train[target].values.reshape(-1, 1))

        # Normalize the validation set if it is not empty
        if not getattr(self, 'val').empty:
            self.val[numerical_features] = scaler.transform(self.val[numerical_features])
            self.val[target] = target_scaler.transform(self.val[target].values.reshape(-1, 1))
        
        # Normalize the test set if it is not empty
        if not getattr(self, 'test').empty:
            self.test[numerical_features] = scaler.transform(self.test[numerical_features])
            self.test[target] = target_scaler.transform(self.test[target].values.reshape(-1, 1))

        self.scaler = scaler
        self.target_scaler = target_scaler

    def split_data_to_features_and_target(self, target):
        """
        Split the data into features and target
        """
        for set in ['train', 'val', 'test']:
            # Check that the set is not empty
            if not getattr(self, set).empty:
                # Split the data into features and target
                setattr(self, set + '_X', getattr(self, set).drop(target, axis=1))
                setattr(self, set + '_y', getattr(self, set)[target])

    def shift_target(self, num_steps):
        """
        Shift the target by one time step
        """
        for set in ['train', 'val', 'test']:
            # Check that the set is not empty
            temp_df = pd.DataFrame()
            if not getattr(self, set).empty:
                # Shift the target
                for i in range(1, num_steps + 1):
                    if i == 1:
                        temp_df["feat_window_" + str(i)] = getattr(self, set + '_y').values
                    else:
                        temp_df["feat_window_" + str(i)] = temp_df["feat_window_" + str(i-1)].shift(-1)
                
                temp_df.dropna(axis=0, inplace=True)
                setattr(self, set + '_y', temp_df)
                setattr(self, set + '_X', getattr(self, set + '_X').drop(getattr(self, set + '_X').tail(num_steps - 1).index))

    def export_data(self, path, type):
        """
        Export the data to the given files
        """
        if type == 'json':
            aggr_dict = {}

            for set in ['train', 'val', 'test']:
                if not getattr(self, set).empty:
                    aggr_dict[set + '_X'] = getattr(self,set + '_X').to_json(orient='split')
                    aggr_dict[set + '_y'] = getattr(self,set + '_y').to_json(orient='split')
            
            # Add the scaler and convert NaN to 0
            scaler_min = self.scaler.min_
            scaler_min[np.isnan(scaler_min)] = 0
            scaler_scale = self.scaler.scale_
            scaler_scale[np.isnan(scaler_scale)] = 1

            aggr_dict['scaler'] = {}
            aggr_dict['scaler']['min'] = scaler_min.tolist()
            aggr_dict['scaler']['scale'] = scaler_scale.tolist()
            aggr_dict['target_scaler'] = {}
            aggr_dict['target_scaler']['min'] = self.target_scaler.min_.tolist()
            aggr_dict['target_scaler']['scale'] = self.target_scaler.scale_.tolist()

            json_object = json.dumps(aggr_dict, indent = 4) 
            # Write to file
            with open(path, "w") as outfile: 
                outfile.write(json_object)
        elif type == 'csv':
            self.data.to_csv(path, index=True, index_label='daytime')


def generate_temporal_features(df, conf):
    if 'week_of_year' in conf['features']['optionalFeatures']['temporal']:
        df['week_of_year'] = df.index.isocalendar().week.astype('int')
    if 'weekday' in conf['features']['optionalFeatures']['temporal']:
        df['weekday'] = df.index.weekday  # 0 monday - 6 sunday
    if 'day' in conf['features']['optionalFeatures']['temporal']:
        df['day'] = df.index.day  # 1-31 calendar day
    if 'month' in conf['features']['optionalFeatures']['temporal']:
        df['month'] = df.index.month  # 1 january - 12 december
    if 'hour' in conf['features']['optionalFeatures']['temporal']:
        df['hour'] = df.index.hour  # 0-23
    if 'minute' in conf['features']['optionalFeatures']['temporal']:
        df['minute'] = df.index.minute  # 0-59
    if 'is_working_hour' in conf['features']['optionalFeatures']['temporal']:
        # If the hour is between 8 and 20 and it is not a weekend set the value to 1
        df['is_working_hour'] = np.where((df.index.hour >= 8) & (df.index.hour <= 20) & (
            df.index.dayofweek != 5) & (df.index.dayofweek != 6), 1, 0)
    if 'is_weekend' in conf['features']['optionalFeatures']['temporal']:
        df['is_weekend'] = np.where((df.index.dayofweek == 5) | (
            df.index.dayofweek == 6), 1, 0)

    return df


def generate_metric_features(df, conf, target, time_interval):
    # Generate the features for the past metrics - example: past 7 days
    categories = list(conf['features']['optionalFeatures']['pastMetrics'].keys())

    for category in categories:
        category_metrics = conf['features']['optionalFeatures']['pastMetrics'][category]
        if len(category_metrics) != 0:
            df = generate_category_metric_features(
                df, category, category_metrics, target, time_interval, False)
    return df


def generate_category_metric_features(df, category, category_metrics, target, time_interval, is_nan_allowed: False):
    # Get lookback size
    freq = past_features[category]['lookback']
    # Get Interval length and number of intervals based on the dataset frequency
    interval_length = pd.to_timedelta(int(time_interval[:-1]), unit=time_interval[-1])
    
    # check if freq is "M" so as to convert to days as it is not supported by pandas
    if freq[-1] == "M":
        freq = "30D"
    LOOKBACK = int(pd.Timedelta(int(freq[:-1]), unit=freq[-1]) / interval_length)

    # Calculate actual load if needed
    if 'actual' in category_metrics:
        df[category + '_actual'] = df[target].shift(LOOKBACK, fill_value=np.NaN)

    # Calculate mean/min/max load if needed
    if 'mean' in category_metrics or 'min' in category_metrics or 'max' in category_metrics:
        temp_dict_loads = {}

        # Iterate over the dataframe
        for i in range(0, len(df)):
            # Find the timestamp of the previous hour/day/week/month
            if time_interval[-1] == 'T':
                prev_timestamp = df.index[i] - pd.Timedelta(minutes=LOOKBACK*int(time_interval[:-1]))

                # Find the timestamp of the previous hour 3 hours
                prev_timestamp_lookback = df.index[i] - pd.Timedelta(minutes=LOOKBACK*int(time_interval[:-1])+180)
                
            elif time_interval[-1] == 'H':
                prev_timestamp = df.index[i] - pd.Timedelta(hours=LOOKBACK)

                # Find the timestamp of the previous 3 hours
                prev_timestamp_lookback = df.index[i] - pd.Timedelta(hours=LOOKBACK + 3)
                
            elif time_interval[-1] == 'D':
                prev_timestamp = df.index[i] - pd.Timedelta(days=LOOKBACK)

                # Find the timestamp of the previous 3 days
                prev_timestamp_lookback = df.index[i] - pd.Timedelta(days=LOOKBACK + 3)
            # elif time_interval[-1] == 'W':
            #     prev_timestamp = df.index[i] - pd.Timedelta(weeks=LOOKBACK)
            # elif time_interval[-1] == 'M':
            #     prev_timestamp = df.index[i] - pd.Timedelta(months=LOOKBACK)
            

            # Get the previous loads among the lookback time
            temp_dict_loads[df.index[i]] = df.loc[(df.index >= prev_timestamp_lookback) & (
                df.index <= prev_timestamp)][[target]].values

            # If the previous loads are empty
            if len(temp_dict_loads[df.index[i]]) == 0:
                # If it is the first row or NaN values are allowed, set the value to NaN
                if i == 0 or is_nan_allowed:
                    temp_dict_loads[df.index[i]] = [[np.NaN]]
                # Else set the value to the previous value
                else:
                    temp_dict_loads[df.index[i]] = df.loc[[
                        df.index[i - 1]], [target]].values
        
        if 'mean' in category_metrics:
            df[category + '_mean'] = df.index.to_series().apply(lambda x: temp_dict_loads[x].mean() if len(temp_dict_loads[x]) > 1 else temp_dict_loads[x][0][0])
            # Round the mean value to 3 decimal places
            df[category + '_mean'] = df[category + '_mean'].round(3)
        if 'min' in category_metrics:
            df[category + '_min'] = df.index.to_series().apply(lambda x: temp_dict_loads[x].min() if len(temp_dict_loads[x]) > 1 else temp_dict_loads[x][0][0])
            df[category + '_min'] = df[category + '_min'].round(3)
        if 'max' in category_metrics:
            df[category+ '_max'] = df.index.to_series().apply(lambda x: temp_dict_loads[x].max() if len(temp_dict_loads[x]) > 1 else temp_dict_loads[x][0][0])
            df[category + '_max'] = df[category + '_max'].round(3)

    # If NaN values are not allowed
    if not is_nan_allowed:
        # Replace the NaN values with the mean value
        df.fillna(df.mean(), inplace=True)
        # # Replace the NaN values with the next value
        # df.fillna(method='bfill', inplace=True)
        # # Replace the NaN values with the previous value
        # df.fillna(method='ffill', inplace=True)
        # # Replace the NaN values with 0
        # df.fillna(0, inplace=True)
    else:
        # drop rows with NaN values
        df.dropna(inplace=True)

    return df


def generate_derivative_features(df, config, target):
    if 'slope' in config['features']['optionalFeatures']['derivatives']:
        # Calculate the first derivative (slope)
        df['slope'] = np.NaN
        for i in range(0, len(df)):
            if i < 2:
                df.at[df.index[i], 'slope'] = 0
                continue
            dt = df.index[i - 1] - df.index[i - 2]
            dy = df[target][i - 1] - df[target][i - 2]
            df.at[df.index[i], 'slope'] = dy / dt.total_seconds()
        
        # Round the slope to 2 decimal places
        df['slope'] = df['slope'].round(3)

    # slope has to be calculated before curvature
    if 'curvature' in config['features']['optionalFeatures']['derivatives']:
        # calculate the second derivative (curvature)
        df['curvature'] = np.NaN

        # Check if slope is enabled so it has already been calculated
        if 'slope' in config['features']['optionalFeatures']['derivatives']:
            for i in range(0, len(df)):
                if i < 2:
                    df.at[df.index[i], 'curvature'] = 0
                    continue
                dt = df.index[i - 1] - df.index[i - 2]
                dy = df.slope[i - 1] - df.slope[i - 2]
                df.at[df.index[i], 'curvature'] = dy / dt.total_seconds()
        else:
            # Calculate the first derivative (slope) and then the second derivative (curvature)
            slope = []
            for i in range(0, len(df)):
                if i < 2:
                    slope.append(0)
                    continue
                dt = df.index[i - 1] - df.index[i - 2]
                dy = df[target][i - 1] - df[target][i - 2]
                slope.append(dy / dt.total_seconds())

            for i in range(0, len(df)):
                if i < 2:
                    df.at[df.index[i], 'curvature'] = 0
                    continue
                dt = df.index[i - 1] - df.index[i - 2]
                dy = slope[i - 1] - slope[i - 2]
                df.at[df.index[i], 'curvature'] = dy / dt.total_seconds()

        # Round the curvature to 2 decimal places
        df['curvature'] = df['curvature'].round(3)

    return df


def generate_features_new_data(df, config, past_metrics, features):
    """
    Generate the features for the new data based on the old data
    """
    # Create the temporal features
    if 'temporal' in config['features']['optionalFeatures']:
        for encoded_feature in config['features']['optionalFeatures']['temporal']:
            # Get all the columns that include the encoded_feature string in the format encoded_feature_0, encoded_feature_1, ...
            pattern = re.compile(r'({})_(0|[1-9]|1[0-9])'.format(encoded_feature))
            encoded_feature_columns = [column for column in features if pattern.match(column)]
       
             # Add the missing columns and based on df_timeseries index update the values
            for column in encoded_feature_columns:
                # Itterate throught rows of df
                if encoded_feature == 'minute':
                    df[column] = df.index.to_series().apply(lambda x: 1 if x.minute == int(column.split('_')[1]) else 0)
                elif encoded_feature == 'hour':
                    df[column] = df.index.to_series().apply(lambda x: 1 if x.hour == int(column.split('_')[1]) else 0)
                elif encoded_feature == 'day':
                    df[column] = df.index.to_series().apply(lambda x: 1 if x.day == int(column.split('_')[1]) else 0)
                elif encoded_feature == 'month':
                    df[column] = df.index.to_series().apply(lambda x: 1 if x.month == int(column.split('_')[1]) else 0)
                elif encoded_feature == 'weekday':
                    df[column] = df.index.to_series().apply(lambda x: 1 if x.dayofweek == int(column.split('_')[1]) else 0)
                elif encoded_feature == 'week_of_year':
                    df[column] = df.index.isocalendar().week.eq(int(column.split('_')[3])).astype(int)


    # Create the past metric features if the past metrics dataframe is not empty
    if not past_metrics.empty :
        categories = list(config['features']['optionalFeatures']['pastMetrics'].keys())
        for category in categories:
            # Get all the columns that include the metric string in the format metric_0, metric_1, ...
            category_metrics = config['features']['optionalFeatures']['pastMetrics'][category]
            if len(category_metrics) != 0:
                # verify that the category exists in the features - Example features ["prevDay_actual", "prevHour_min"] and category "prevDay"
                pattern = re.compile(r'({})_(max|min|actual)'.format(category))
                category_columns = [column for column in features if pattern.match(column)]

                if len(category_columns) != 0:
                    # Save the actual values of the past metrics
                    if category == 'prevHour':
                        # Retrieve the previous hour's value
                        previous_timestamp = df.index[0] - pd.Timedelta(hours=1)
                    elif category == 'prevDay':
                        # Retrieve the previous day's value
                        previous_timestamp = df.index[0] - pd.Timedelta(days=1)
                    elif category == 'prevWeek':
                        # Retrieve the previous week's value
                        previous_timestamp = df.index[0] - pd.Timedelta(weeks=1)
                    elif category == 'prevMonth':
                        # Retrieve the previous month's value
                        previous_timestamp = df.index[0] - pd.Timedelta(days=30)

                    # convert the previous day as Daytime
                    previous_timestamp = previous_timestamp.strftime("%Y-%m-%d %H:%M:%S")

                    if "actual" in category_metrics:
                        previous_value = past_metrics.loc[previous_timestamp, config['target']]
                        df[category + '_actual'] = previous_value
                            
                        
                    if 'mean' in category_metrics or 'min' in category_metrics or 'max' in category_metrics:
                        temp_dict_loads = {}

                        for i in range(0, len(df)):
                            # Find the timestamp of the previous hour/day/week/month + 3 hours
                            # Save the actual values of the past metrics
                            if category == 'prevHour':
                                # Retrieve the previous hour's value
                                prev_timestamp_lookback = df.index[0] - pd.Timedelta(hours=3)
                            elif category == 'prevDay':
                                # Retrieve the previous day's value
                                prev_timestamp_lookback = df.index[0] - pd.Timedelta(days=1) - pd.Timedelta(hours=3)
                            elif category == 'prevWeek':
                                # Retrieve the previous week's value
                                prev_timestamp_lookback = df.index[0] - pd.Timedelta(weeks=1) - pd.Timedelta(hours=3)
                            elif category == 'prevMonth':
                                # Retrieve the previous month's value
                                prev_timestamp_lookback = df.index[0] - pd.Timedelta(days=30) - pd.Timedelta(hours=3)
                            
                            # convert the previous day as Daytime
                            prev_timestamp_lookback = prev_timestamp_lookback.strftime("%Y-%m-%d %H:%M:%S")

                            # Get the previous loads among the lookback time
                            temp_dict_loads[df.index[i]] = past_metrics.loc[(past_metrics.index >= prev_timestamp_lookback) & (
                                past_metrics.index <= previous_timestamp)][[config['target']]].values
                            
                            if 'mean' in category_metrics:
                                df[category + '_mean'] = df.index.to_series().apply(lambda x: temp_dict_loads[x].mean() if len(temp_dict_loads[x]) > 1 else temp_dict_loads[x][0][0])
                                # Round the mean value to 3 decimal places
                                df[category + '_mean'] = df[category + '_mean'].round(3)
                            if 'min' in category_metrics:
                                df[category + '_min'] = df.index.to_series().apply(lambda x: temp_dict_loads[x].min() if len(temp_dict_loads[x]) > 1 else temp_dict_loads[x][0][0])
                                df[category + '_min'] = df[category + '_min'].round(3)
                            if 'max' in category_metrics:
                                df[category+ '_max'] = df.index.to_series().apply(lambda x: temp_dict_loads[x].max() if len(temp_dict_loads[x]) > 1 else temp_dict_loads[x][0][0])
                                df[category + '_max'] = df[category + '_max'].round(3)
        
        # Create the derivative features
        # if 'derivative' in config['features']['optionalFeatures']:
        #     df = generate_derivative_features(df, config)
    
    return df

                