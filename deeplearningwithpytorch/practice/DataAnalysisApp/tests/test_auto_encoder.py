"""
Unittest for the encoder module

Author: Yaolin Ge
Date: 2024-10-21
"""
from LSTMAutoEncoder.AutoEncoder import AutoEncoder
from LSTMAutoEncoder.LSTMAutoEncoder import LSTMAutoEncoder as ReferenceAutoEncoder
from unittest import TestCase
import torch

class TestAutoEncoder(TestCase):
    def setUp(self):
        self.input_size = 7
        self.hidden_sizes = [128, 16]
        self.batch_size = 32
        self.seq_length = 30

        self.autoencoder = AutoEncoder(self.input_size, self.hidden_sizes)
        self.reference_autoencoder = ReferenceAutoEncoder()

        # Set the same weights for both autoencoders
        with torch.no_grad():
            for lstm, ref_lstm in zip(self.autoencoder.encoder.lstms,
                                      [self.reference_autoencoder.encoder.lstm1, self.reference_autoencoder.encoder.lstm2]):
                lstm.weight_ih_l0.copy_(ref_lstm.weight_ih_l0)
                lstm.weight_hh_l0.copy_(ref_lstm.weight_hh_l0)
                lstm.bias_ih_l0.copy_(ref_lstm.bias_ih_l0)
                lstm.bias_hh_l0.copy_(ref_lstm.bias_hh_l0)

            for lstm, ref_lstm in zip(self.autoencoder.decoder.lstms,
                                      [self.reference_autoencoder.decoder.lstm1, self.reference_autoencoder.decoder.lstm2]):
                lstm.weight_ih_l0.copy_(ref_lstm.weight_ih_l0)
                lstm.weight_hh_l0.copy_(ref_lstm.weight_hh_l0)
                lstm.bias_ih_l0.copy_(ref_lstm.bias_ih_l0)
                lstm.bias_hh_l0.copy_(ref_lstm.bias_hh_l0)

    def test_print_out_parameters(self):
        # print out all parameter shapes from autoencoder
        for name, param in self.autoencoder.named_parameters():
            print(name, param.shape)

        for name, param in self.reference_autoencoder.named_parameters():
            print(name, param.shape)

    def test_output_shape(self):
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.autoencoder(x)
        ref_output = self.reference_autoencoder(x)
        self.assertEqual(output.shape, ref_output.shape)

    def test_output_values(self):
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.autoencoder(x)
        ref_output = self.reference_autoencoder(x)
        self.assertTrue(torch.allclose(output, ref_output, atol=1e-6))
