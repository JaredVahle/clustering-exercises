import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats




def only_single_unit(df):
    filtered_df = df[df['calculatedfinishedsquarefeet'] < 10000]
    filtered_df = filtered_df[filtered_df['calculatedfinishedsquarefeet'] >=1000]
    filtered_df = filtered_df[filtered_df['bedroomcnt'] >=2 ]
    filtered_df = filtered_df[filtered_df['bedroomcnt'] <= 12]
    filtered_df = filtered_df[filtered_df['bathroomcnt'] >=2]
    filtered_df = filtered_df[filtered_df['bathroomcnt'] <8]
    return filtered_df


def train_validate_test(df, target):
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=174)
    # split train/validate into train (60%) and validate (20%)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=174)
    # splits our target off of our train, validate, test
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


def min_max_scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs 
    """
    scaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled

def handle_outliers(df, col):
    q1 = df[col].quantile(.25)
    q3 = df[col].quantile(.75)
    iqr = q3-q1 #Interquartile range
    lower_bound  = q1-1.5*iqr
    upper_bound = q3+1.5*iqr
    if lower_bound < 0:
        lower_bound = 0
    if upper_bound > df[col].max():
        upper_bound = df[col].max()
    df_out = df.loc[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df_out

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df