"""
This module aims to preprocess the incoming data into the input
format of the model.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:

    def __init__(self) -> None:
        pass

    def preprocess(self, df: pd.DataFrame) -> tuple:
        raw_columns = ["x", "y", "z", "strain0", "strain1"]
        df_initial = df
        df = df[raw_columns]

        # s0, scale the data
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df_initial.index)

        # s1, first difference
        df = df.diff()

        # s3, remove nan values
        df = df.dropna()

        # s4, prepare for the training data
        lags_LSTM = 30
        lags_classical = [5, 10]  # [6, 12, 18]
        windows = [30]  # [60]
        train_val_test_split = [0.78, 0.05, 0.17]  # must add to one otherwise will default to [0.8, 0.1, 0.1]

        train_end_idx = int(len(df) * train_val_test_split[0]) - 1
        val_end_idx = train_end_idx + int(len(df) * train_val_test_split[1]) - 1
        test_end_idx = len(df) - 1

        index_train = df.index[train_end_idx]
        index_val = df.index[val_end_idx]
        index_test = df.index[test_end_idx]

        def create_lags(df, lags_classical, columns):
            df_copy = df.copy()
            for col_id in columns:
                for lag in lags_classical:
                    df_copy[f"{col_id}_{lag}"] = df[col_id].shift(lag)
            return df_copy

        def create_moving_A_STD(df, windows, columns):
            df_copy = df.copy()
            for col_id in columns:
                for window in windows:
                    df_copy[f"MA_{col_id}_{window}"] = df[col_id].rolling(window=window).mean()
            return df_copy

        def create_sequence_LSTM(data, sequence_length):
            sequences = []
            for i in range(len(data) - sequence_length):
                sequence = data[i:i + sequence_length]
                sequences.append(sequence)
            return np.array(sequences)

        df_temp = create_sequence_LSTM(np.array(df), lags_LSTM)
        raw_df_in_cut_TRAIN_LSTM = df_temp[:train_end_idx]
        raw_df_in_cut_VAL_LSTM = df_temp[train_end_idx:val_end_idx]
        raw_df_in_cut_TEST_LSTM = df_temp[val_end_idx:]

        df_model = df.copy()
        df_model = create_lags(df_model, lags_classical, raw_columns)
        df_model = create_moving_A_STD(df_model, windows, raw_columns)
        df_model.dropna(inplace=True)
        raw_df_in_cut_TRAIN = df_model[df_model.index <= index_train]
        raw_df_in_cut_VAL = df_model[(df_model.index > index_train) & (df_model.index <= index_val)]
        raw_df_in_cut_TEST = df_model[df_model.index > index_val]

        # s7, output resulting processed dataframes
        new_raw_cols = list(raw_df_in_cut_TRAIN.columns)
        X_train = np.array(raw_df_in_cut_TRAIN[[*new_raw_cols]])
        X_val = np.array(raw_df_in_cut_VAL[[*new_raw_cols]])
        X_test = np.array(raw_df_in_cut_TEST[[*new_raw_cols]])

        X_train_LSTM = raw_df_in_cut_TRAIN_LSTM[:, :, :len(raw_columns)]
        X_val_LSTM = raw_df_in_cut_VAL_LSTM[:, :, :len(raw_columns)]
        X_test_LSTM = raw_df_in_cut_TEST_LSTM[:, :, :len(raw_columns)]

        return [X_train, X_val, X_test], [X_train_LSTM, X_val_LSTM, X_test_LSTM]

