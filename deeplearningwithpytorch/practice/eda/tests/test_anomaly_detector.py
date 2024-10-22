from unittest import TestCase
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from AnomalyDetector import AnomalyDetector
from LSTMAutoEncoder.AutoEncoder import AutoEncoder
from Gen1CSVHandler import Gen1CSVHandler

class TestMachineLearning(TestCase):

    def setUp(self) -> None:
        self.filePath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin\SilentTools_00410_20211130-143236.cut"
        self.gen1CSVHandler = Gen1CSVHandler(self.filePath)
        self.model = AutoEncoder(input_size=7, hidden_sizes=[64, 32])
        self.criterion = nn.MSELoss
        self.optimizer = optim.Adam
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.anomaly_detector = AnomalyDetector(self.model, self.criterion, self.optimizer,
                                                pre_trained_model=True, device=self.device)

    def test_train_pred_model(self):
        data = self.gen1CSVHandler.df_sync
        self.anomaly_detector.update(data)
        # self.machineLearning.train_model(self.dataHandler.train_loader, self.dataHandler.val_loader, self.dataHandler.test_loader, num_epochs=30)
        # self.anomaly_detector.predict(self.dataHandler.df_sync_cropped)

