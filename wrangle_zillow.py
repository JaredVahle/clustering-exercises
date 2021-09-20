import os
import pandas as pd
import env
import numpy as np
from scipy import stats

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

import prepare
import acquire

def wrangle_zillow():
    df = acquire.get_zillow_data()
    acquire.summarize_df(df)
    df.hist(figsize=(24, 12), bins=20)
    plt.tight_layout()
    plt.show()
    acquire.nulls_by_col(df)
    acquire.nulls_by_row(df)
    df = prepare.only_single_unit(df)
    df = prepare.handle_missing_values(df, prop_required_column=0.65, prop_required_row=0.75)
    median_lot_size = df.lotsizesquarefeet.median()
    df["lotsizesquarefeet"].fillna(median_lot_size,inplace = True)
    df['yearbuilt'].fillna(2017, inplace = True)
    df.dropna(inplace = True)
    return df