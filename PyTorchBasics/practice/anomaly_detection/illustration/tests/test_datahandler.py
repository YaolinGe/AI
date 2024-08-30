from unittest import TestCase
from src.DataHandler import DataHandler
import numpy as np
import matplotlib.pyplot as plt


class TestDataHandler(TestCase):

    def setUp(self) -> None:
        # self.data_handler = DataHandler(look_back=20, look_forward=50)
        # self.data_handler.create_dataset(train_size=.5)

        self.data_handler_autoencoder = DataHandler(look_back=50)
        self.data_handler_autoencoder.create_dataset_for_autoencoder(train_size=.5)

    # def test_create_dataset(self):
    #     plt.plot(self.data_handler.timestamp_train, self.data_handler.signal_train, label='Train')
    #     plt.plot(self.data_handler.timestamp_test, self.data_handler.signal_test, label='Test')
    #     plt.legend()
    #     plt.show()
    
    # def test_sequence_step(self): 
    #     steps = np.arange(0, 3, 1)
    #     plt.figure(figsize=(18, 4))
    #     ind_start = 0
    #     ind_end = -1
    #     for i in steps:
    #         plt.subplot(1, len(steps), i+1)
    #         plt.plot(self.data_handler.timestamp_train[ind_start:ind_end], self.data_handler.signal_train[ind_start:ind_end], 'k.-', label='GroundTruth', linewidth=0.5, markersize=2)
    #         plt.plot(self.data_handler.timestamp_train[i:i+self.data_handler.look_back], self.data_handler.X_train[i], 'rx', label='Feature')
    #         plt.plot(self.data_handler.timestamp_train[i+self.data_handler.look_back:i+self.data_handler.look_back+self.data_handler.look_forward], self.data_handler.y_train[i], 'bx', label='Target')
    #         plt.legend(loc="lower left")
    #     plt.show()

    def test_create_dataset_for_autoencoder(self): 
        plt.plot(self.data_handler_autoencoder.timestamp_train, self.data_handler_autoencoder.signal_train, label='Train')
        plt.plot(self.data_handler_autoencoder.timestamp_test, self.data_handler_autoencoder.signal_test, label='Test')
        plt.legend()
        plt.show()
    
    def test_sequence_step_for_autoencoder(self): 
        steps = np.arange(0, 3, 1)
        plt.figure(figsize=(18, 4))
        ind_start = 0
        ind_end = -1
        for i in steps:
            plt.subplot(1, len(steps), i+1)
            plt.plot(self.data_handler_autoencoder.timestamp_train[ind_start:ind_end], self.data_handler_autoencoder.signal_train[ind_start:ind_end], 'k.-', label='GroundTruth', linewidth=0.5, markersize=2)
            plt.plot(self.data_handler_autoencoder.timestamp_train[i:i+self.data_handler_autoencoder.look_back], self.data_handler_autoencoder.X_train[i], 'rx', label='Feature')
            plt.legend(loc="lower left")
        print("Feature shape: ", self.data_handler_autoencoder.X_train.shape)
        plt.show()

