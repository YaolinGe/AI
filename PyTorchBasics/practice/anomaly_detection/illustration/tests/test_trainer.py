import torch
import matplotlib.pyplot as plt
from unittest import TestCase
from src.Trainer import Trainer
from src.model.LSTM import LSTMModel
from src.model.LSTMAutoEncoder import LSTMAutoEncoder
from src.DataHandler import DataHandler
import numpy as np
import os 


# class TestTrainerLSTM(TestCase):

#     def setUp(self) -> None:
#         self.look_back = 20
#         self.look_forward = 50
#         self.data_handler = DataHandler(look_back=self.look_back, look_forward=self.look_forward)
#         self.data_handler.create_dataset(train_size=.5)
#         self.data_handler.create_dataloader()
#         self.model = LSTMModel(input_size=1, hidden_size=100, num_layers=2, output_size=self.look_forward)
#         self.trainer = Trainer(model=self.model, train_loader=self.data_handler.train_loader, test_loader=self.data_handler.test_loader)
#         self.trainer.run(verbose=True, display=True)

#     # def test_run(self):
#     #     self.assertEqual(len(self.trainer.train_loss), 200)
#     #     self.assertEqual(len(self.trainer.test_loss), 200)
#     #
#     # def test_evaluate(self):
#     #     loss = self.trainer.evaluate()
#     #     self.assertIsInstance(loss, float)
#     #
#     # def test_plot(self):
#     #     self.trainer.display()

#     def test_prediction(self):
#         self.model.eval()
#         with torch.no_grad():
#             output = self.model(self.data_handler.X_test.to(self.trainer.device)).cpu().detach().numpy()

#         plt.figure(figsize=(48, 20))
#         for i in range(4):
#             plt.subplot(2, 4, i+1)
#             plt.plot(self.data_handler.timestamp_test, self.data_handler.signal_test, 'k.-', label='GroundTruth', linewidth=0.5, markersize=2)
#             plt.plot(self.data_handler.timestamp_test[i:i+self.look_back], self.data_handler.X_test[i].numpy(), 'bx', label='Input')
#             plt.plot(self.data_handler.timestamp_test[i+self.look_back:i+self.look_back+self.look_forward], output[i], 'rx', label='Prediction')
#             plt.legend(loc="lower left")

#             plt.subplot(2, 4, i+5)
#             error_feature = np.abs(self.data_handler.X_test[i].numpy().flatten() - self.data_handler.signal_test[i:i+self.look_back])
#             error = np.abs(self.data_handler.signal_test[i+self.look_back:i+self.look_back+self.look_forward] - output[i])
#             plt.plot(self.data_handler.timestamp_test[i:i+self.look_back], error_feature, 'bo', label='Error Feature')
#             plt.fill_between(self.data_handler.timestamp_test[i:i+self.look_back], error_feature, color='blue', alpha=0.3)
#             plt.plot(self.data_handler.timestamp_test[i+self.look_back:i+self.look_back+self.look_forward], error, 'go', label='Error')
#             plt.fill_between(self.data_handler.timestamp_test[i+self.look_back:i+self.look_back+self.look_forward], error, color='red', alpha=0.3)
#             plt.legend(loc="lower left")

#         plt.show()
#         plt.show()


class TestTrainerLSTMAutoEncoder(TestCase):

    def setUp(self) -> None:
        self.sequence_length = 50
        self.data_handler = DataHandler(look_back=self.sequence_length)
        self.data_handler.create_dataset_for_autoencoder(train_size=.5)
        self.data_handler.create_dataloader_for_autoencoder()
        self.model = LSTMAutoEncoder(input_size=1, hidden_size=64, num_layers=1, latent_size=16, output_size=1)
        self.trainer = Trainer(model=self.model, train_loader=self.data_handler.train_loader, test_loader=self.data_handler.test_loader)
        self.trainer.run_autoencoder(verbose=True, display=True)
    
    def test_prediction(self) -> None: 
        self.model.eval()
        with torch.no_grad(): 
            output = self.model(self.data_handler.X_test.to(self.trainer.device)).cpu().detach().numpy()
        plt.figure(figsize=(48, 20))
        for i in range(4): 
            plt.subplot(2, 4, i+1)
            plt.plot(self.data_handler.timestamp_test, self.data_handler.signal_test, 'k.-', label='GroundTruth', linewidth=0.5, markersize=2)
            plt.plot(self.data_handler.timestamp_test[i:i+self.sequence_length], self.data_handler.X_test[i].numpy(), 'bo', label='Input')
            plt.plot(self.data_handler.timestamp_test[i:i+self.sequence_length], output[i], 'rx', label='Prediction')
            plt.legend(loc="lower left")
            
            plt.subplot(2, 4, i+5)
            error_feature = np.abs(self.data_handler.X_test[i].numpy().flatten() - self.data_handler.signal_test[i:i+self.sequence_length])
            error = np.abs(self.data_handler.signal_test[i:i+self.sequence_length] - output[i].flatten())
            plt.plot(self.data_handler.timestamp_test[i:i+self.sequence_length], error_feature, 'bo', label='Error Feature')
            plt.fill_between(self.data_handler.timestamp_test[i:i+self.sequence_length], error_feature, color='blue', alpha=0.3)
            plt.plot(self.data_handler.timestamp_test[i:i+self.sequence_length], error, 'go', label='Error')
            plt.fill_between(self.data_handler.timestamp_test[i:i+self.sequence_length], error, color='red', alpha=0.3)
            plt.legend(loc="lower left")
        plt.show()

    def test_make_illustration(self) -> None:
        self.model.eval()
        folderpath = os.path.join(os.path.expanduser("~"), "Downloads")
        with torch.no_grad():
            input_sequences = DataHandler.create_sequences_for_autoencoder(self.data_handler.signal.truth, self.sequence_length)
            output = self.model(input_sequences.to(self.trainer.device)).cpu().detach().numpy()w

        plt.figure(figsize=(12, 6))
        plt.plot(self.data_handler.signal.timestamp[:len(self.data_handler.signal.truth)-self.sequence_length+1], input_sequences[:, 0, :].flatten(), 'k-', linewidth=1)
        plt.axis('off')
        plt.savefig(os.path.join(folderpath, "input.png"), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close("all")

        plt.figure(figsize=(12, 6))
        plt.plot(self.data_handler.signal.timestamp[:len(self.data_handler.signal.truth)-self.sequence_length+1], output[:, 0, :].flatten(), 'k-', linewidth=1)
        plt.axis('off')
        plt.savefig(os.path.join(folderpath, "output.png"), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close("all")
