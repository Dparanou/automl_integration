from flaskr.db import get_db
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')
# disable chained assignments
pd.options.mode.chained_assignment = None 

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
        incorrect_stamps = np.where(
            self.data.index.to_series().diff() != pd.Timedelta(TIME_INTERVAL))[0][1:]

        return ((incorrect_stamps.shape[0] / 2) / self.data.shape[0]) * 100

    def check_data_quality(self, target, configs):
        """
        Check if the data quality is good enough to be used for training.
        - Check percentage of dataset NaN values
        - Check percentage of timestamps not in the given frequency
        - Check if dataset contains outliers
        """
        TIME_INTERVAL = configs.get_setting('time_interval')

        if self.calc_perc_nan_values(target) > configs.get_setting('max_perc_nan_values') * 100:
            print('Too many NaN values')
        if self.calc_perc_incorrect_timestamps(TIME_INTERVAL) > configs.get_setting('max_perc_wrong_interval_values') * 100:
            print('Too many incorrect timestamps')
        if self.has_outliers(target):
            print('There are outliers in the dataset')
        if (self.calc_perc_nan_values(target) > configs.get_setting('max_perc_nan_values') * 100
            and self.has_outliers(target)
                and self.calc_perc_incorrect_timestamps(TIME_INTERVAL) > configs.get_setting('max_perc_wrong_interval_values') * 100):
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

    def export_data(self, path, type):
        """
        Export the data to the given files
        """
        if type == 'json':
            # Export the data
            self.data.to_json(path, index=True, orient='split')
        elif type == 'csv':
            self.data.to_csv(path, index=True, index_label='daytime')

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

                # if configs["are_temporal_features_enabled"]:
                #     self[set] = generate_temporal_features(configs)

                # if configs["are_derivative_features_enabled"]:
                #     self[set] = generate_derivative_features(configs, target)

                # self[set] = self[set].prettify_df(configs)

def generate_features(df, target, configs):
    '''
    Takes dataframe with schema: DAYTIME, TARGET.
    Also takes path to a YAML file that has to be like features.yml
    Returns a Dataframe filled with the features specified in the config file
    Optional args: export_path -> path to output csv containing the df - Default: None
    '''
    # Add the feature columns to the dataframe
    df = add_feature_columns_to_df(df, configs)

    # Generates all the features specified in the config file
    if configs.is_category_enabled("past_metrics"):
        df = generate_metric_features(df, configs, target)

    if configs.is_category_enabled("temporal"):
        df = generate_temporal_features(df, configs)

    if configs.is_category_enabled("derivatives"):
        df = generate_derivative_features(df, configs, target)

    df = prettify_df(df, configs)

    return df


def generate_temporal_features(df, conf):
    # Check if dataframe is empty
    if df.empty:
        return df
    
    if (conf['week_of_year']):
        df['week_of_year'] = df.index.isocalendar(
        ).week.astype('int')  # 1-52 week number
    if (conf['weekday']):
        df['weekday'] = df.index.dayofweek  # 0 monday - 6 sunday
    if (conf['day']):
        df['day'] = df.index.day  # 1-31 calendar day
    if (conf['month']):
        df['month'] = df.index.month  # 1 january - 12 december
    if (conf['hour']):
        df['hour'] = df.index.hour  # 0-23
    if (conf['minute']):
        df['minute'] = df.index.minute  # 0-59
    if (conf['is_working_hour']):
        # If the hour is between 8 and 20 and it is not a weekend set the value to 1
        df['is_working_hour'] = np.where((df.index.hour >= 8) & (df.index.hour <= 20) & (
            df.index.dayofweek != 5) & (df.index.dayofweek != 6), 1, 0)
    if (conf['is_weekend']):
        df['is_weekend'] = np.where((df.index.dayofweek == 5) | (
            df.index.dayofweek == 6), 1, 0)

    # minutes_in_day = 60 * 24
    # minutes = df.index.hour * 60 + df.index.minute

    # if (conf['cyclical_hour']):
    #     df['cyclical_hour_sin'] = np.sin(
    #         2 * np.pi * minutes / minutes_in_day)
    # if (conf['cyclical_hour']):
    #     df['cyclical_hour_cos'] = np.cos(
    #         2 * np.pi * minutes / minutes_in_day)

    return df


def generate_metric_features(df, conf):
    categories = list(conf['past_features'][0].keys())
    is_nan_allowed = False

    for category in categories:
        category_metrics = conf['past_features'][0][category]
        df = generate_category_metric_features(
            df, category, category_metrics, conf['value'], conf['time_interval'], is_nan_allowed)
    
    print(df.tail(10))
    return df

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

def generate_category_metric_features(df, category, category_metrics, target, time_interval, is_nan_allowed):
    # Get lookback and window size
    freq = past_features[category]['lookback']
    # Get Interval length and number of intervals based on the dataset frequency
    interval_length = pd.to_timedelta(int(time_interval[:-1]), unit=time_interval[-1])
    num_intervals = int(pd.Timedelta(1, unit='D') / interval_length)
    LOOKBACK = int(pd.Timedelta(int(freq[:-1]), unit=freq[-1]) / interval_length)
    WINDOW_SIZE = past_features[category]['window_size'] * num_intervals

    # print('LOOKBACK ',LOOKBACK)
    # print('Interval length ',interval_length)
    # print('WINDOW_SIZE ',WINDOW_SIZE)
    # print('num_intervals ',num_intervals)
    # print()
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
        if 'min' in category_metrics:
            df[category + '_min'] = df.index.to_series().apply(lambda x: temp_dict_loads[x].min() if len(temp_dict_loads[x]) > 1 else temp_dict_loads[x][0][0])
        if 'max' in category_metrics:
            df[category+ '_max'] = df.index.to_series().apply(lambda x: temp_dict_loads[x].max() if len(temp_dict_loads[x]) > 1 else temp_dict_loads[x][0][0])

    # If NaN values are not allowed
    # if not is_nan_allowed:
    #     # Replace the NaN values with the next value
    #     df.fillna(method='bfill', inplace=True)

    return df


def generate_derivative_features(df, conf, target):
    if (conf.is_feature_enabled('slope')):
        # Calculate the first derivative (slope)
        df['slope'] = np.NaN
        for i in range(0, len(df)):
            if i < 2:
                df.at[df.index[i], 'slope'] = 0
                continue
            dt = df.index[i - 1] - df.index[i - 2]
            dy = df[target][i - 1] - df[target][i - 2]
            df.at[df.index[i], 'slope'] = dy / dt.total_seconds()

    # slope has to be calculated before curvature
    if (conf.is_feature_enabled('curvature')):
        # calculate the second derivative (curvature)
        df['curvature'] = np.NaN

        # Check if slope is enabled so it has already been calculated
        if conf.is_feature_enabled('slope'):
            for i in range(0, len(df)):
                if i < 2:
                    df.at[df.index[i], 'curvature'] = 0
                    continue
                dt = df.index[i - 1] - df.index[i - 2]
                dy = df.slope[i - 1] - df.slope[i - 2]
                df.at[df.index[i], 'curvature'] = dy / dt.total_seconds()
        else:
            # Calculate the first derivative (slope)
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

    return df


def prettify_df(df, conf):
    # round all numbers to reduce csv file size (3.333333333333 -> 3.34)
    df = df.round(2)

    # For temporal features, convert the values to int
    if conf.is_category_enabled("temporal"):
        for feature in conf.get_category_features("temporal"):
            # Verify that the feature has not to do with cos/sin/cyclical hour
            if feature not in ['hour_sin', 'hour_cos', 'cyclical_hour']:
                df[feature] = df[feature].astype('int')
    return df


def one_hot_encode(df, categorical_features):
    """
    One hot encode the categorical features
    """
    # Check if the categorical features are in the dataframe
    if set(categorical_features).issubset(df.columns):
        df = pd.get_dummies(df, columns=categorical_features)

    return df


def normalize_data(df_train, df_val, df_test, target, categorical_features = []):
    """
    Normalize the data using MinMaxScaler from sklearn
    """
    # Get the features
    features = df_train.columns

    # Get the numerical features
    numerical_features = list(set(features) - set(categorical_features))
    # Remove the target from the numerical features
    numerical_features.remove(target)

    # Normalize the numerical features
    scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Normalize the train data
    df_train[numerical_features] = scaler.fit_transform(
        df_train[numerical_features])
    df_train[target] = target_scaler.fit_transform(
        df_train[target].values.reshape(-1, 1))

    # Normalize the validation data if it exists
    if not df_val.empty:
        df_val[numerical_features] = scaler.transform(
            df_val[numerical_features])
        df_val[target] = target_scaler.transform(
            df_val[target].values.reshape(-1, 1))

    # Normalize the test data if it exists
    if not df_test.empty:
        df_test[numerical_features] = scaler.transform(
            df_test[numerical_features])
        df_test[target] = target_scaler.transform(
            df_test[target].values.reshape(-1, 1))

    return df_train, df_val, df_test, scaler, target_scaler


def separate_X_y(df, target):
    """
    Seperate the dataframe to X and y
    """
    X = df.drop(target, axis=1)
    y = df[target]

    return X, y


def shifted_target(df, df_target, window_size):
    """
    Shift the target to the future
    """
    df_shifted = pd.DataFrame()

    for i in range(0, window_size):
        if i == 0:
            df_shifted["feat_window_" + str(i)] = df_target.values
        else:
            df_shifted["feat_window_" +
                       str(i)] = df_shifted["feat_window_" + str(i - 1)].shift(-1)

    # Drop the nan values
    df_shifted = df_shifted.dropna(axis=0)
    df.drop(df.tail(window_size - 1).index, inplace=True)

    return df, df_shifted


if __name__ == "__main__":
    data = Data()

    # Get the configuration file
    config = {'time_interval': '8H', 'target': 'cores'}
    target = config['target']

    # Slpit the data into train, validation and test
    data.split_data(config.get_setting('test_perc'), config.get_setting('val_perc'))

    # Generate features of the data
    train = generate_features(
        data.train, target, config)
    val = generate_features(data.val, target, config)
    test = generate_features(
        data.test, target, config)
    
    # Normalize the data
    train_scaled, val_scaled, test_scaled, scaler, target_scaler = normalize_data(
        train, val, test, target, config.get_setting("categorical_features"))

    # One hot encode the categorical features
    train_scaled = one_hot_encode(train_scaled, config.get_setting("one_hot_encode_features"))
    val_scaled = one_hot_encode(val_scaled, config.get_setting("one_hot_encode_features"))
    test_scaled = one_hot_encode(test_scaled, config.get_setting("one_hot_encode_features"))

    # Separate the data to X and y
    X_train, y_train = separate_X_y(train_scaled, target)
    X_val, y_val = separate_X_y(val_scaled, target)
    X_test, y_test = separate_X_y(test_scaled, target)

    # Shift the target to the future
    X_train, y_train_shifted = shifted_target(X_train, y_train, config.get_setting('predicted_window'))
    X_val, y_val_shifted = shifted_target(X_val, y_val, config.get_setting('predicted_window'))
    X_test, y_test_shifted = shifted_target(X_test, y_test, config.get_setting('predicted_window'))

    # Export the data to csv files
    # export_data(X_train, 'train_scaled.csv')
    # export_data(y_train_shifted, 'y_train_shifted.csv')
