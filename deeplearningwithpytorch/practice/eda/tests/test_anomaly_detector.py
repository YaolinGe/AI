from unittest import TestCase
from AnomalyDetector import AnomalyDetector
from DataHandler import DataHandler
import pandas as pd
import numpy as np
import os


class TestMachineLearning(TestCase):

    def setUp(self) -> None:
        self.anomaly_detector = AnomalyDetector()

    def test_train_pred_model(self):
        # self.machineLearning.train_model(self.dataHandler.train_loader, self.dataHandler.val_loader, self.dataHandler.test_loader, num_epochs=30)
        self.anomaly_detector.predict(self.dataHandler.df_sync_cropped)

