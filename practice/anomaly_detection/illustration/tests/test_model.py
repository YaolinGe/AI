from unittest import TestCase
from src.model.LSTM import LSTMModel
from src.model.LSTMAutoEncoder import LSTMAutoEncoder
import torch
import numpy as np
import matplotlib.pyplot as plt


class TestModel(TestCase):

    def setUp(self) -> None:
        self.model1 = LSTMModel(input_size=1, hidden_size=100, num_layers=2, output_size=1)
        self.model2 = LSTMModel(input_size=1, hidden_size=100, num_layers=2, output_size=5)
        self.model3 = LSTMAutoEncoder(input_size=7, hidden_size=128, num_layers=1, latent_size=16, output_size=7)

    def test_forward(self):
        x = torch.tensor(np.random.rand(32, 10, 1), dtype=torch.float32)
        y1 = self.model1(x)
        y2 = self.model2(x)
        self.assertEqual(y1.shape, torch.Size([32, 1]))
        self.assertEqual(y2.shape, torch.Size([32, 5]))
        x = torch.tensor(np.random.rand(32, 10, 7), dtype=torch.float32)
        y3 = self.model3(x)
        self.assertEqual(y3.shape, torch.Size([32, 10, 7]))

    def test_plot(self):
        x = torch.tensor(np.random.rand(32, 10, 1), dtype=torch.float32)
        y = self.model1(x)
        plt.plot(y.detach().numpy(), label='Prediction')
        plt.legend()
        plt.show()