from unittest import TestCase
from src.Signal import Signal
import numpy as np
import matplotlib.pyplot as plt


class TestSignal(TestCase):

    def setUp(self) -> None:
        self.signal = Signal()
    
    def test_generate_sinusoidal(self):
        freq = 1.
        amplitude = 1.
        phase = 0.
        time = np.linspace(0, 10, 1000)
        truth_signal = self.signal.generate_sinusoidal(time, freq, amplitude, phase)
        expected_signal = amplitude * np.sin(2 * np.pi * freq * time + phase)
        # plt.plot(time, truth_signal, label='Truth Signal')
        # plt.plot(time, expected_signal, label='Expected Signal')
        # plt.legend()
        # plt.show()
        np.testing.assert_array_equal(truth_signal, expected_signal)

    def test_generate_signal(self):
        self.signal.generate_signal()
        plt.plot(self.signal.timestamp, self.signal.truth, 'k.-', label='Truth Signal')
        plt.legend()
        plt.show()