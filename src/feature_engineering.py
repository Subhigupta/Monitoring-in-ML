import pandas as pd

def data_prep(df):
    
    # Create time features
    df['year'] = df['datetime'].dt.year
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour
    
    df_copy = df.copy()
    
    # Drop atemp as temp and atemp are highly linearly coorrelation
    df_copy.drop("atemp", axis=1, inplace=True)
    df_copy.drop("datetime", axis=1, inplace=True)

    return df_copy




