from flaskr.db import get_db
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json

import warnings
warnings.filterwarnings('ignore')
# disable chained assignments
pd.options.mode.chained_assignment = None 

# Define the past features to be used in the generation of shifted features
past_features = {
    'last_hour': {
        'lookback': '1H',
        'window_size': 1,
    },
    'last_3_hours': {
        'lookback': '3H',
        'window_size': 3,
    },
    'last_12_hours': {
        'lookback': '12H',
        'window_size': 12,
    },
    'last_day': {
        'lookback': '1D',
        'window_size': 24,
    },
    'last_week': {
        'lookback': '1W',
        'window_size': 168,
    },
    'last_month': {
        'lookback': '1M',
        'window_size': 720,
    },
}

class Data:
    def __init__(self, **kwargs):
        """
        Get the data from the database.
        Possible input parameters:
        start_date, end_date, values
        """
        # If the input parameters are not provided, use default values
        if 'start_date' not in kwargs:
            self.data = pd.DataFrame()
            return
           
        db = get_db()
        # Generate the SQL query string dynamically
        query_string = 'SELECT daytime, ' + kwargs['value'] + ' FROM timeseries WHERE (daytime >= ? AND daytime < ?)'

        # Execute the query and pass the query string as a parameter
        ts = db.execute(query_string, (kwargs['start_date'], kwargs['end_date'])).fetchall()

        ts = [tuple(row) for row in ts]

        # Convert the list of tuples to a pandas DataFrame
        df = pd.DataFrame(ts, columns=['daytime', kwargs['value']])
        # Set the index to the datetime column
        df.set_index('daytime', inplace=True)
        # Convert the index to datetime
        df.index = pd.to_datetime(df.index)
        # Sort the DataFrame by index
        df.sort_index(inplace=True)

        self.data = df
    
    def set_data(self, data):
        """
        Set the data.

        Parameters:
        data (pd.DataFrame): The data
        """
        self.data = data

    def update_set(self, selected_set, new_value):
        setattr(self, selected_set, new_value)
    
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

    def has_incorrect_timestamps(self, TIME_INTERVAL):
        """
        Check if the data contains incorrect timestamps and return a boolean value.

        Returns:
        bool: True if the data contains incorrect timestamps, False otherwise
        """
        # Check if the frequency of the records is in given interval
        incorrect_stamps = np.where(
            self.data.index.to_series().diff() != pd.Timedelta(TIME_INTERVAL))[0][1:]

        return len(incorrect_stamps) > 0

    def calc_perc_nan_values(self, target):
        """
        Calculate the percentage of NaN values in the data.

        Returns:
        float: The percentage of NaN values
        """
        return (self.data[target].isnull().sum() / self.data[target].shape[0]) * 100

    def calc_perc_incorrect_timestamps(self, TIME_INTERVAL):
        """
        Calculate the percentage of incorrect timestamps in the data.

        Returns:
        float: The percentage of incorrect timestamps
        """
        # print(self.data.index.to_series().diff())
        incorrect_stamps = np.where(
            self.data.index.to_series().diff() != pd.Timedelta(TIME_INTERVAL))[0][1:]

        print('Incorrect stamps: ', incorrect_stamps.shape[0] / 2)
        return ((incorrect_stamps.shape[0] / 2) / self.data.shape[0]) * 100

    def check_data_quality(self, target, TIME_INTERVAL):
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
        if self.calc_perc_incorrect_timestamps(TIME_INTERVAL) > max_perc_wrong_interval_values * 100:
            print('Too many incorrect timestamps')
        if self.has_outliers(target):
            print('There are outliers in the dataset')
        if (self.calc_perc_nan_values(target) > max_perc_nan_values * 100
            and self.has_outliers(target)
                and self.calc_perc_incorrect_timestamps(TIME_INTERVAL) > max_perc_wrong_interval_values * 100):
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

    def fix_incorrect_timestamps(self, TIME_INTERVAL):
        """
        Fix the frequency of the timestamps to be in given interval
        by removing the timestamps that are apart from the given interval
        """
        # Keep only the correct timestamps in given intervals
        self.data = self.data.resample(TIME_INTERVAL).first()

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

    def split_data(self, test_perc, val_perc):
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
        
    def clean_data(self, target, TIME_INTERVAL):
        """
        Clean the data by removing the outliers, filling NaN values and fixing the incorrect timestamps
        """
        self.remove_outliers(target)
        self.fill_nan_values(target)
        self.fix_incorrect_timestamps(TIME_INTERVAL)

    def generate_features(self, configs):
        """
        Generate the features specified in the config file
        """
        for set in ['train', 'val', 'test']:
            # Check that the set is not empty
            if not getattr(self, set).empty:
                # Generates all the features specified in the config file
                if configs["are_past_features_enabled"]:
                    self.update_set(set, generate_metric_features(getattr(self, set), configs))

                if configs["are_temporal_features_enabled"]:
                    self.update_set(set, generate_temporal_features(getattr(self, set), configs))

                if configs["are_derivative_features_enabled"]:
                    self.update_set(set, generate_derivative_features(getattr(self, set), configs))

    def normalize_data(self, configs):
        """
        Normalize the data using MinMaxScaler from sklearn
        """
        # Get the features
        features = getattr(self, 'train').columns

        # Get the numerical features (all features except the categorical and target features)
        numerical_features = list((set(features) - set(configs['temporal_features'])) - set(configs['value']))
        
        # Normalize the numerical features
        scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        self.train[numerical_features] = scaler.fit_transform(self.train[numerical_features])
        self.train[configs['value']] = target_scaler.fit_transform(self.train[configs['value']].values.reshape(-1, 1))

        # Normalize the validation set if it is not empty
        if not getattr(self, 'val').empty:
            self.val[numerical_features] = scaler.transform(self.val[numerical_features])
            self.val[configs['value']] = target_scaler.transform(self.val[configs['value']].values.reshape(-1, 1))
        
        # Normalize the test set if it is not empty
        if not getattr(self, 'test').empty:
            self.test[numerical_features] = scaler.transform(self.test[numerical_features])
            self.test[configs['value']] = target_scaler.transform(self.test[configs['value']].values.reshape(-1, 1))

        self.scaler = scaler
        self.target_scaler = target_scaler

    def one_hot_encode(self):
        """
        One hot encode the categorical features
        """
        # Get the categorical features
        categorical_features = getattr(self, 'train').columns
        # Define the features that will be one hot encoded
        # encode_features = ['hour', 'day', 'minute', 'month', 'weekday']
        encode_features = ['hour']

        # Keep the intersection of the categorical features and the features that will be one hot encoded
        categorical_features = list(set(categorical_features) & set(encode_features))
        
        # One hot encode the categorical features
        self.update_set('train', pd.get_dummies(getattr(self, 'train'), columns=categorical_features))

        # One hot encode the validation set if it is not empty
        if not getattr(self, 'val').empty:
            self.update_set('val', pd.get_dummies(getattr(self, 'val'), columns=categorical_features))
        
        # One hot encode the test set if it is not empty
        if not getattr(self, 'test').empty:
            self.update_set('test', pd.get_dummies(getattr(self, 'test'), columns=categorical_features))

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
            
            json_object = json.dumps(aggr_dict, indent = 4) 
            # Write to file
            with open(path, "w") as outfile: 
                outfile.write(json_object)
        elif type == 'csv':
            self.data.to_csv(path, index=True, index_label='daytime')


def generate_temporal_features(df, conf):
    if 'week_of_year' in conf['temporal_features']:
        df['week_of_year'] = df.index.isocalendar(
        ).week.astype('int')  # 1-52 week number
    if 'weekday' in conf['temporal_features']:
        df['weekday'] = df.index.dayofweek  # 0 monday - 6 sunday
    if 'day' in conf['temporal_features']:
        df['day'] = df.index.day  # 1-31 calendar day
    if 'month' in conf['temporal_features']:
        df['month'] = df.index.month  # 1 january - 12 december
    if 'hour' in conf['temporal_features']:
        df['hour'] = df.index.hour  # 0-23
    if 'minute' in conf['temporal_features']:
        df['minute'] = df.index.minute  # 0-59
    if 'is_working_hour' in conf['temporal_features']:
        # If the hour is between 8 and 20 and it is not a weekend set the value to 1
        df['is_working_hour'] = np.where((df.index.hour >= 8) & (df.index.hour <= 20) & (
            df.index.dayofweek != 5) & (df.index.dayofweek != 6), 1, 0)
    if 'is_weekend' in conf['temporal_features']:
        df['is_weekend'] = np.where((df.index.dayofweek == 5) | (
            df.index.dayofweek == 6), 1, 0)

    return df


def generate_metric_features(df, conf):
    categories = list(conf['past_features'][0].keys())
    is_nan_allowed = False

    for category in categories:
        category_metrics = conf['past_features'][0][category]
        df = generate_category_metric_features(
            df, category, category_metrics, conf['value'], conf['time_interval'], is_nan_allowed)
    
    return df


def generate_category_metric_features(df, category, category_metrics, target, time_interval, is_nan_allowed):
    # Get lookback and window size
    freq = past_features[category]['lookback']
    # Get Interval length and number of intervals based on the dataset frequency
    interval_length = pd.to_timedelta(int(time_interval[:-1]), unit=time_interval[-1])
    num_intervals = int(pd.Timedelta(1, unit='D') / interval_length)
    LOOKBACK = int(pd.Timedelta(int(freq[:-1]), unit=freq[-1]) / interval_length)
    WINDOW_SIZE = past_features[category]['window_size'] * num_intervals

    # Calculate actual load if needed
    if 'actual' in category_metrics:
        df[category + '_actual'] = df[target].shift(LOOKBACK, fill_value=np.NaN)

    # Calculate mean/min/max load if needed
    if 'mean' in category_metrics or 'min' in category_metrics or 'max' in category_metrics:
        temp_dict_loads = {}

        # Iterate over the dataframe
        for i in range(0, len(df)):
            # Find the timestamp of the previous hour/day/week/month
            prev_timestamp = df.index[i] - pd.Timedelta(hours=WINDOW_SIZE)
            # Find the timestamp of the previous hour/day/week/month + lookback time -> ex 2 hours
            prev_timestamp_lookback = df.index[i] - \
                pd.Timedelta(hours=WINDOW_SIZE + (num_intervals * int(time_interval[:-1])))

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
        # Replace the NaN values with the next value
        df.fillna(method='bfill', inplace=True)

    return df


def generate_derivative_features(df, conf):
    target = conf['value']
    if 'slope' in conf['derivative_features']:
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
    if 'curvature' in conf['derivative_features']:
        # calculate the second derivative (curvature)
        df['curvature'] = np.NaN

        # Check if slope is enabled so it has already been calculated
        if 'slope' in conf['derivative_features']:
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
