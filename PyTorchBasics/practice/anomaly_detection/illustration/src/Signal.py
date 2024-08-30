"""
Signal class to handle all signal related operations

Author: Yaolin Ge
Date: 2024-08-30
"""

import numpy as np 


class Signal: 
    def __init__(self) -> None: 
        self.frequency = .15
        self.amplitude = 1.
        self.phase = .0
        self.timestamp = np.arange(0, 20, .1)

    def generate_signal(self) -> None:
        self.truth = Signal.generate_sinusoidal(self.timestamp, freq=self.frequency, amplitude=self.amplitude, phase=self.phase)
        print("Total samples: ", len(self.truth))
        # self.truth = Signal.generate_sinusoidal(self.timestamp) + Signal.generate_sinusoidal(self.timestamp, freq=0.3, amplitude=0.5) + Signal.generate_sinusoidal(self.timestamp, freq=0.5, amplitude=0.3)

    @staticmethod
    def generate_sinusoidal(time: np.ndarray, freq: float=1.0, amplitude: float=1., phase: float=.0) -> np.ndarray:
        """
        Generate a sinusoidal signal with given parameters
        :param freq: frequency of the signal, [Hz]
        :param amplitude: amplitude of the signal
        :param phase: phase of the signal, [rad]
        :param time: time points to evaluate the signal, [sec]
        :return: sinusoidal signal
        """
        return amplitude * np.sin(2 * np.pi * freq * time + phase)
