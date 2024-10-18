from unittest import TestCase
from src.DataHandler import DataHandler
from src.Signal import Signal
import matplotlib.pyplot as plt
import numpy as np


class TestDataHandler(TestCase):

    def setUp(self) -> None:
        self.signal = Signal(frequency=.15, amplitude=1., phase=0., noise=0.1)
        self.signal.generate_signal()
        self.data_handler = DataHandler(look_back=20, look_forward=50, signal=self.signal, isAutoEncoder=True)
        self.data_handler.create_dataset()

    def test_create_dataset(self):
        plt.plot(self.data_handler.timestamp_train, self.data_handler.signal_train, label='Train')
        plt.plot(self.data_handler.timestamp_test, self.data_handler.signal_test, label='Test')
        plt.legend()
        plt.show()
    
    def test_sequence_step(self):
        steps = np.arange(0, 3, 1)
        plt.figure(figsize=(18, 4))
        ind_start = 0
        ind_end = -1
        for i in steps:
            plt.subplot(1, len(steps), i+1)
            plt.plot(self.data_handler.timestamp_train[ind_start:ind_end], self.data_handler.signal_train[ind_start:ind_end], 'k.-', label='GroundTruth', linewidth=0.5, markersize=2)
            plt.plot(self.data_handler.timestamp_train[i:i+self.data_handler.look_back], self.data_handler.X_train[i], 'rx', label='Feature', markersize=20)
            if not self.data_handler.isAutoEncoder:
                plt.plot(self.data_handler.timestamp_train[i+self.data_handler.look_back:i+self.data_handler.look_back+self.data_handler.look_forward], self.data_handler.Y_train[i], 'bx', label='Target')
            else:
                plt.plot(self.data_handler.timestamp_train[i:i+self.data_handler.look_back], self.data_handler.Y_train[i], 'bx', label='Target')
            plt.legend(loc="lower left")
        plt.show()


