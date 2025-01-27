from unittest import TestCase
import pandas as pd
import numpy as np
import onnxruntime as ort
import torch
from numpy import testing
from DataProcessor import DataProcessor
from playground import DP


class TestDataProcessor(TestCase):
    def setUp(self) -> None:
        self.dp = DataProcessor()
        self.dp2 = DP()
        self.df = pd.read_csv("df_disk1.csv")

    def test_process(self):
        raw_columns = ["x", "y", "z", "strain0", "strain1"]
        ground_truth_columns = ["InCut", "Anomaly"]
        df_processed = self.dp2.preprocess(self.df, raw_columns, scaling=True,
                                       firstDifferenceParameters={"numDifference": 1},
                                       PCAParameters=None, dropNA=True)

        useWeight = False  # use class weight in training for various algorithms for class inbalance
        take_splits = False  # split data frame so that lags are only taken from actual lags (rather than stitched together)
        drop_ts_less = 60  # drop smaller in cuts for training
        lags_LSTM = 30
        lags_classical = [5, 10]  # [6, 12, 18]
        windows = [30]  # [60]
        train_val_test_split = [0.78, 0.05, 0.17]  # must add to one otherwise will default to [0.8, 0.1, 0.1]

        classic_models_data, lstm_models_data, train_test_val_df, class_weight, sample_weight = self.dp2.create_features_train_val_test(
            raw_df=df_processed, raw_columns=raw_columns, useWeight=useWeight, take_splits=take_splits,
            drop_ts_less=drop_ts_less, lags_LSTM=lags_LSTM, lags_classical=lags_classical, windows=windows,
            train_val_test_split=train_val_test_split)
        X_train, y_train, X_val, y_val, X_test, y_test = classic_models_data
        X_train_LSTM, y_train_LSTM, X_val_LSTM, y_val_LSTM, X_test_LSTM, y_test_LSTM = lstm_models_data
        raw_df_in_cut_TRAIN_LSTM, raw_df_in_cut_VAL_LSTM, raw_df_in_cut_TEST = train_test_val_df

        classical_model_data, lstm_model_data = self.dp.preprocess(self.df)
        X_train_2, X_val_2, X_test_2 = classical_model_data
        X_train_LSTM_2, X_val_LSTM_2, X_test_LSTM_2 = lstm_model_data

        # assert all arrays from different data processing units to make sure they all produce the same results
        testing.assert_array_equal(X_train, X_train_2)
        testing.assert_array_equal(X_val, X_val_2)
        testing.assert_array_equal(X_test, X_test_2)

        testing.assert_array_equal(X_train_LSTM, X_train_LSTM_2)
        testing.assert_array_equal(X_val_LSTM, X_val_LSTM_2)
        testing.assert_array_equal(X_test_LSTM, X_test_LSTM_2)

    def test_model_output(self):
        classical_model_data, lstm_model_data = self.dp.preprocess(self.df)
        X_train_2, X_val_2, X_test_2 = classical_model_data
        X_train_LSTM_2, X_val_LSTM_2, X_test_LSTM_2 = lstm_model_data


        # Inference for classical model
        sess = ort.InferenceSession("GNB.onnx") # imports model to session object (enabling layers, functions, weights)
        input_name = sess.get_inputs()[0].name 
        label_name = sess.get_outputs()[0].name
        test_sample = np.array(X_test_2, dtype=np.float32) # np matrix where rows is observation and columns are
        test_predictions_GNB = sess.run([label_name], {input_name: test_sample})[0]
        print("Output: ", test_predictions_GNB)
