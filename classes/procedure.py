import pandas as pd
import sys
import numpy as np
import joblib
import pickle 
from timeit import default_timer as timer
from pandas import Timestamp
from sklearn.pipeline import Pipeline
from feature_engine.datetime import DatetimeFeatures

def predict(df_test, model, feats, target):
    df_x = df_test[feats]
    df_y = df_test[target]
    X = df_x.values
    y_true = df_y#.values
    y_pred = model.predict(X)
    return y_pred
    
def create_lagged_features(df):
    new_df = df #df[[x for x in df.columns if x != 'label']]
    #create lagged features
    lags = 3
    df_list = []
    #for l in np.unique(new_df.label):
    
    df_lags_list = [new_df]
    cols_past = [x for x in new_df.columns if "t+" not in x] 
    df_past = new_df[cols_past]
    for i in range(1, lags+1):
        df_temp = df_past.shift(i)
        df_temp.columns = [c+f"_lag_{i}" for c in cols_past]
        df_lags_list.append(df_temp.copy())
    df_lagged = pd.concat(df_lags_list, axis=1)
    dtfs = DatetimeFeatures(
        variables="index",
        features_to_extract=["month", "hour", "day_of_week"],
        drop_original=False
    )
    df_temp = dtfs.fit_transform(df_lagged)
    return df_temp





def grpc_forecasting(turbine_label,forecast_index):
   
    a=timer()
    print(f"Turbine Label {turbine_label}, Forecasting dates {forecast_index} till {forecast_index+pd.Timedelta(hours=48)}")
    df = pd.read_csv(f"./datas/beico_{turbine_label}.csv", index_col=0)
    df=df.set_index(['Unnamed: 0'])

    df = df[[col for col in df.columns if 'Rtr' not in col]]
    df.index = pd.to_datetime(df.index)
    if forecast_index < df.index.min() or forecast_index > df.index.max()-pd.Timedelta(hours=48):
        raise ValueError(f"forecast_index is outside the valid range.Applicable between {df.index.min()}:{df.index.max()}")
    test_index = df.loc[forecast_index:forecast_index+pd.Timedelta(hours=48)].index
    # print(test_index)
    df=create_lagged_features(df)
    
    future_steps = 48
    df_list = []
    for l in np.unique(df.label):
        df_list_inner = [df.loc[df.label==l]]
        df_future = df.loc[df.label==l, ['Grd_Prod_Pwr_min', 'MeanWindSpeedUID_10.0m', 'MeanWindSpeedUID_100.0m', 
                                        'DirectionUID_10.0m', 'DirectionUID_100.0m', "month", "hour", "day_of_week"]]
        for i in range(1, future_steps+1):
            if i != 0:
                df_temp = df_future.shift(-i)
                df_temp.columns = [c+f"_(t+{i})" for c in df_future.columns]
                df_list_inner.append(df_temp) 
        df_f = pd.concat(df_list_inner, axis=1)
        df_f = df_f.dropna()
        df_list.append(df_f)

    df_f = pd.concat(df_list, axis=0)
    df_f = df_f.dropna()
    # print(df_f.shape)

    df_test =df_f.loc[df_f.index.isin(test_index)]
    # print(df_test)
    target_features = [x for x in df_f.columns if 'Grd_Prod_Pwr_min_(t+' in x]
    fit_features = [x for x in df_f.columns if 'Grd_Prod_Pwr_min_(t+' not in x]

    models_file=f'/data/1/panos/Service_Forecasting/forcasting_model.pickle.z'
    with open(models_file, 'rb') as file:
        all_models_dict = joblib.load(file)
    
    
    y_pred_test =all_models_dict['0.5']['models'].predict(df_test[fit_features].values)
    result_temp = pd.DataFrame(y_pred_test, index=df_test.index)
    print(result_temp)
    
   
    result_for_vis=pd.DataFrame({"val":result_temp.iloc[0].values},index=result_temp.index[1:])
    result_for_vis.index=(result_for_vis.index.astype(int)//10**9).astype(str)
    
    result_dict={}
    result_dict['results'] = result_for_vis['val'].to_dict()
    print(result_dict['results'])
    b=timer()
    print(b-a)
    return result_dict
try:
    grpc_forecasting(2, pd.to_datetime("2022-06-01 1:00:00"))
except ValueError as e:
    print(e)