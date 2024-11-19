"""
Unittest for the encoder module

Author: Yaolin Ge
Date: 2024-10-21
"""
from LSTMAutoEncoder.Decoder import Decoder
from LSTMAutoEncoder.LSTMAutoEncoder import Decoder as ReferenceDecoder
from unittest import TestCase
import torch


class TestDecoder(TestCase):
    def setUp(self):
        self.hidden_sizes = [16, 128]
        self.output_size = 7
        self.batch_size = 32
        self.seq_length = 30

        self.decoder = Decoder(self.hidden_sizes, self.output_size)
        self.reference_decoder = ReferenceDecoder()

        # Set the same weights for both decoders
        with torch.no_grad():
            for lstm, ref_lstm in zip(self.decoder.lstms,
                                      [self.reference_decoder.lstm1, self.reference_decoder.lstm2]):
                lstm.weight_ih_l0.copy_(ref_lstm.weight_ih_l0)
                lstm.weight_hh_l0.copy_(ref_lstm.weight_hh_l0)
                lstm.bias_ih_l0.copy_(ref_lstm.bias_ih_l0)
                lstm.bias_hh_l0.copy_(ref_lstm.bias_hh_l0)

    def test_print_out_parameters(self):
        # print out all parameter shapes from both encoder and reference_encoder
        for name, param in self.decoder.named_parameters():
            print(name, param.shape)

        for name, param in self.reference_decoder.named_parameters():
            print(name, param.shape)

    def test_output_shape(self):
        x = torch.randn(self.batch_size, self.seq_length, self.hidden_sizes[0])
        output = self.decoder(x)
        ref_output = self.reference_decoder(x)

        self.assertEqual(output.shape, ref_output.shape)

    def test_output_values(self):
        x = torch.randn(self.batch_size, self.seq_length, self.hidden_sizes[0])
        output = self.decoder(x)
        ref_output = self.reference_decoder(x)

        self.assertTrue(torch.allclose(output, ref_output, atol=1e-6))