from unittest import TestCase
from model.ClassicalModelTrainer import ClassicalModelTrainer
from model.ModelTrainer import ModelTrainer
import pandas as pd
import numpy as np


class TestMachineLearning(TestCase):

    def setUp(self) -> None:
        self.file_path = "datasets/df_gulbox_merged.csv"
        self.cmt = ClassicalModelTrainer(self.file_path)
        self.mt = ModelTrainer(self.file_path)

    def test_train_pred_model(self):
        # self.cmt.run_training_pipeline(output_dir="model/models")
        self.mt.run_training_pipeline(output_dir="model/models")


