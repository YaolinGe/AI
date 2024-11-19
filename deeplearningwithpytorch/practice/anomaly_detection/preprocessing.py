"""
Preprocessing module for the data

Author: Yaolin Ge
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocessing(filepath):
    df = pd.read_csv(filepath)
    temp = df.copy()
    temp['Time'] = pd.to_timedelta(temp['Time'])
    temp['Time'] = temp['Time'].dt.total_seconds()
    temp['Time'] = temp['Time'] - temp['Time'][0]
    t = temp['Time'].values
    value = temp['Value'].values
    scaler = MinMaxScaler()
    value = scaler.fit_transform(value.reshape(-1, 1)).flatten()
    return t, value

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)

