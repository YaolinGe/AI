from unittest import TestCase
from AnomalyDetector import AnomalyDetector
from DataHandler import DataHandler
import pandas as pd
import numpy as np
import os


class TestMachineLearning(TestCase):

    def setUp(self) -> None:
        self.machineLearning = AnomalyDetector()
        self.dataHandler = DataHandler()
        folder_path = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin"
        files = os.listdir(folder_path)
        filenames = [file[:-4] for file in files if file.endswith('.cut')]
        filepath = os.path.join(folder_path, filenames[0])
        self.dataHandler.load_synchronized_data(filepath)
        self.dataHandler.get_cropped_data(8.0, 15.0)
        self.dataHandler.prepare_training_data(self.dataHandler.df_sync_cropped)

    def test_train_pred_model(self):
        # self.machineLearning.train_model(self.dataHandler.train_loader, self.dataHandler.val_loader, self.dataHandler.test_loader, num_epochs=30)
        self.machineLearning.predict(self.dataHandler.df_sync_cropped)

