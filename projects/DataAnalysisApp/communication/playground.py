import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class DP:
    def __init__(self) -> None:
        pass

    def preprocess(self, data, columns_interest, scaling=False, firstDifferenceParameters=None, PCAParameters=None, dropNA=False):
        data_initial = data
        data = data[[*columns_interest]]
        if scaling:
            print("\nScaling Data")
            scaler = MinMaxScaler()

            # DEBUG, can be deleted
            # min_values = np.amin(data, axis=0)
            # max_values = np.amax(data, axis=0)
            # min_max_df = pd.DataFrame({'column': data.columns, 'min_value': min_values, 'max_value': max_values})
            # min_max_df.to_csv('min_max_values.csv', index=False)
            data.to_csv("data.csv", index=False)

            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data_initial.index)

        if firstDifferenceParameters is not None:
            print("\nTaking Difference")
            numDiff = int(firstDifferenceParameters["numDifference"])
            for i in range(numDiff):
                data = data[[*columns_interest]].diff()

        if PCAParameters is not None:
            end_dim = int(PCAParameters['end_dim'])
            if end_dim == None:  # if want to see explained varience
                pca = PCA()
                pca.fit(data[[*columns_interest]])
                features = range(pca.n_components_)
                plt.figure(figsize=(15, 5))
                plt.bar(features, pca.explained_variance_)
                plt.xlabel('PCA feature')
                plt.ylabel('Variance')
                plt.xticks(features)
                plt.title("Importance of the Principal Components based on inertia")
                plt.show()
            else:
                pca = PCA()
                pd.DataFrame(PCA.fit_transform(data[[*columns_interest]], end_dim),
                             columns=["pc" + str(i + 1) for i in range(end_dim)], index=data_initial.index)

        data["InCut"] = data_initial["InCut"]
        data["Anomaly"] = data_initial["Anomaly"]
        if dropNA:
            data = data.dropna()

        return data

    def create_features_train_val_test(self, raw_df, raw_columns, useWeight, take_splits, drop_ts_less, lags_LSTM, lags_classical,
                                       windows, train_val_test_split):
        # define time series functions
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
                    # df_copy[f"STD_{col_id}"] = df[col_id].rolling(window=window).std()
            return df_copy

        def create_sequence_LSTM(data, sequence_length):
            sequences = []
            for i in range(len(data) - sequence_length):
                sequence = data[i:i + sequence_length]
                sequences.append(sequence)
            return np.array(sequences)

        # ensure adds to 1
        if np.sum(train_val_test_split) != 1:
            train_val_test_split = [0.8, 0.1, 0.1]

        # Create splits for each of the incut periods of time - this is because we cannot stitch together two time series that dont belong next to each other - ie minute 30 cannot be treated as feature for minute 40
        if take_splits:
            split = []
            default = raw_df.InCut.values[0]
            for i in raw_df.InCut.values:
                if (default == 0) and (i == 1):
                    default = 1
                    split.append(1)
                elif (default == 1) and (i == 0):
                    default = 0
                    split.append(1)
                else:
                    split.append(0)
            raw_df['split'] = split

        # drop out of cut for datafrme
        raw_df_in_cut = raw_df
        # raw_df_in_cut = raw_df[(raw_df["InCut"] == 1)]
        raw_df_in_cut = raw_df_in_cut.drop(["InCut"], axis=1)
        # raw_df_in_cut.sort_index(inplace=True)  # ensure in order
        # raw_df_in_cut = raw_df

        # train-val-split calculations - work out IDX for val and test dataframe
        train_end_idx = int(len(raw_df_in_cut) * train_val_test_split[0]) - 1
        val_end_idx = train_end_idx + int(len(raw_df_in_cut) * train_val_test_split[1]) - 1
        test_end_idx = len(raw_df_in_cut) - 1
        index_train = raw_df_in_cut.index[train_end_idx]
        index_val = raw_df_in_cut.index[val_end_idx]
        index_test = raw_df_in_cut.index[test_end_idx]
        # print(index_train, index_val, index_test)

        # actually split into smaller dataframes
        if take_splits:
            dfs_train = []
            dfs_val = []
            dfs_test = []
            dfs_train_LSTM = []
            dfs_val_LSTM = []
            dfs_test_LSTM = []

            dfs = []
            start_idx = raw_df_in_cut.iloc[0].name
            last_idx = raw_df_in_cut.iloc[0].name
            for idx, row in raw_df_in_cut.iterrows():
                if row['split'] == 1 and start_idx != idx:
                    dfs.append(raw_df_in_cut[start_idx:last_idx])
                    start_idx = idx
                last_idx = idx
            dfs.append(raw_df_in_cut[start_idx:])  # Append the last segment

            # process each dataframe seperately (if splits) or as one
            for df in dfs:
                df = df.drop(["split"], axis=1)
                # LSTM
                sequences = create_sequence_LSTM(np.array(df), lags_LSTM)

                # classical models
                df = create_lags(df, lags_classical, raw_columns)
                df = create_moving_A_STD(df, windows, raw_columns)
                df.dropna(inplace=True)

                # assess if belongs to training or test set. If true, test
                if (df.index <= index_train).any() == True:  # train
                    if len(df) < drop_ts_less:  # drop periods in cut where less than x amount of data points (every 0.5 seconds, so 20 would be 10 seconds) - Train ONLY
                        continue
                    dfs_train.append(df)
                    dfs_train_LSTM.append(sequences) if len(sequences) > 0 else 1
                elif ((df.index > index_train).any() == True) & ((df.index <= index_val).any() == True):  # val
                    dfs_val.append(df)
                    dfs_val_LSTM.append(sequences) if len(sequences) > 0 else 1
                else:  # test
                    dfs_test.append(df)
                    dfs_test_LSTM.append(sequences) if len(sequences) > 0 else 1

            # classical models
            raw_df_in_cut_TRAIN = pd.concat(dfs_train)
            raw_df_in_cut_VAL = pd.concat(dfs_val)
            raw_df_in_cut_TEST = pd.concat(dfs_test)

            # LSTM
            dfs_train_LSTM = np.array(dfs_train_LSTM, dtype='object')
            dfs_val_LSTM = np.array(dfs_val_LSTM, dtype='object')
            dfs_test_LSTM = np.array(dfs_test_LSTM, dtype='object')

            raw_df_in_cut_TRAIN_LSTM = np.concatenate(dfs_train_LSTM, axis=0)
            raw_df_in_cut_VAL_LSTM = np.concatenate(dfs_val_LSTM, axis=0)
            raw_df_in_cut_TEST_LSTM = np.concatenate(dfs_test_LSTM, axis=0)

        else:
            df_temp = create_sequence_LSTM(np.array(raw_df_in_cut), lags_LSTM)
            # LSTM
            raw_df_in_cut_TRAIN_LSTM = df_temp[:train_end_idx]
            raw_df_in_cut_VAL_LSTM = df_temp[train_end_idx:val_end_idx]
            raw_df_in_cut_TEST_LSTM = df_temp[val_end_idx:]
            # raw_df_in_cut_TRAIN_LSTM = np.array(raw_df_in_cut)[:train_end_idx]
            # raw_df_in_cut_VAL_LSTM = np.array(raw_df_in_cut)[train_end_idx:val_end_idx]
            # raw_df_in_cut_TEST_LSTM = np.array(raw_df_in_cut)[val_end_idx:]

            # classical models
            df = raw_df_in_cut.copy()
            df = create_lags(df, lags_classical, raw_columns)
            df = create_moving_A_STD(df, windows, raw_columns)
            df.dropna(inplace=True)
            raw_df_in_cut_TRAIN = df[df.index <= index_train]
            raw_df_in_cut_VAL = df[(df.index > index_train) & (df.index <= index_val)]
            raw_df_in_cut_TEST = df[df.index > index_val]

        # classical models
        new_raw_cols = list(raw_df_in_cut_TRAIN.columns)
        new_raw_cols.remove("Anomaly")  # get new column names
        X_train, y_train = np.array(raw_df_in_cut_TRAIN[[*new_raw_cols]]), np.array(raw_df_in_cut_TRAIN["Anomaly"])
        X_val, y_val = np.array(raw_df_in_cut_VAL[[*new_raw_cols]]), np.array(raw_df_in_cut_VAL["Anomaly"])
        X_test, y_test = np.array(raw_df_in_cut_TEST[[*new_raw_cols]]), np.array(raw_df_in_cut_TEST["Anomaly"])

        # LSTM
        X_train_LSTM, y_train_LSTM = raw_df_in_cut_TRAIN_LSTM[:, :, :len(raw_columns)], raw_df_in_cut_TRAIN_LSTM[:, :,
                                                                                        len(raw_columns):][:, -1,
                                                                                        0]  # [:, -1, 0] gets last value (latest) from array as they have been given a sequence. So we want all previous sequences before our label.
        X_val_LSTM, y_val_LSTM = raw_df_in_cut_VAL_LSTM[:, :, :len(raw_columns)], raw_df_in_cut_VAL_LSTM[:, :,
                                                                                  len(raw_columns):][:, -1, 0]
        X_test_LSTM, y_test_LSTM = raw_df_in_cut_TEST_LSTM[:, :, :len(raw_columns)], raw_df_in_cut_TEST_LSTM[:, :,
                                                                                     len(raw_columns):][:, -1, 0]

        if useWeight:
            from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
            class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            sample_weight = np.array([class_weight[int(cls)] for cls in y_train])
        else:
            class_weight = None
            sample_weight = None

        print("\nTraining Cols: ", raw_df_in_cut_TRAIN.columns)

        return [X_train, y_train, X_val, y_val, X_test, y_test], [X_train_LSTM, y_train_LSTM, X_val_LSTM, y_val_LSTM,
                                                                  X_test_LSTM, y_test_LSTM], [raw_df_in_cut_TRAIN_LSTM,
                                                                                              raw_df_in_cut_VAL_LSTM,
                                                                                              raw_df_in_cut_TEST], class_weight, sample_weight

    def print_metrics(self, y_test, test_predictions):
        print("Accuracy", accuracy_score(y_test, test_predictions))
        print("Precision", precision_score(y_test, test_predictions))
        print("Recall", recall_score(y_test, test_predictions))
        print("F1", f1_score(y_test, test_predictions))
        print("\n-----------------------------------------------------------\n")


if __name__ == "__main__":
    dp = DP()
    df_disk1 = pd.read_csv("df_disk1.csv")
    raw_columns = ["x", "y", "z", "strain0", "strain1"]
    ground_truth_columns = ["InCut", "Anomaly"]
    df_processed = dp.preprocess(df_disk1, raw_columns, scaling=True, firstDifferenceParameters={"numDifference": 1}, PCAParameters=None, dropNA=True)
    df_processed = df_processed[[*raw_columns, *ground_truth_columns]]

    useWeight = False  # use class weight in training for various algorithms for class inbalance
    take_splits = True  # split data frame so that lags are only taken from actual lags (rather than stitched together)
    drop_ts_less = 60  # drop smaller in cuts for training
    lags_LSTM = 30
    lags_classical = [5, 10]  # [6, 12, 18]
    windows = [30]  # [60]
    train_val_test_split = [0.78, 0.05, 0.17]  # must add to one otherwise will default to [0.8, 0.1, 0.1]

    classic_models_data, lstm_models_data, train_test_val_df, class_weight, sample_weight = dp.create_features_train_val_test(
        raw_df=df_processed, raw_columns=raw_columns, useWeight=useWeight, take_splits=take_splits,
        drop_ts_less=drop_ts_less, lags_LSTM=lags_LSTM, lags_classical=lags_classical, windows=windows,
        train_val_test_split=train_val_test_split)
    X_train, y_train, X_val, y_val, X_test, y_test = classic_models_data
    X_train_LSTM, y_train_LSTM, X_val_LSTM, y_val_LSTM, X_test_LSTM, y_test_LSTM = lstm_models_data
    raw_df_in_cut_TRAIN_LSTM, raw_df_in_cut_VAL_LSTM, raw_df_in_cut_TEST = train_test_val_df


    threshold = 0.5
    GNB = GaussianNB()
    GNB.fit(X_train, y_train, sample_weight=sample_weight)
    y_prob = GNB.predict_proba(X_test)  # .squeeze()
    test_predictions_GNB = (y_prob[:, 1] >= threshold).astype(int)
    metrics_history = dp.print_metrics(y_test, test_predictions_GNB)


    df_processed



