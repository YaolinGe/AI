"""
Signal class to handle all signal related operations

Author: Yaolin Ge
Date: 2024-08-30
"""

import numpy as np 
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Signal: 
    def __init__(self, frequency: float=.15, amplitude: float=1., phase: float=.0, timestamp: np.ndarray=np.arange(0, 20, .1), noise_level: float=.0,) -> None:
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.noise_level = noise_level
        self.timestamp = timestamp

    def generate_signal(self) -> None:
        base_signal = Signal.generate_sinusoidal(self.timestamp, self.frequency, self.amplitude, self.phase)
        noise_level = self.noise_level * np.random.randn(len(self.timestamp))
        self.truth = base_signal + noise_level
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

    def display(self) -> plotly.graph_objs.Figure: 
        fig = go.Figure(data=go.Scatter(x=self.timestamp, y=self.truth, mode='lines'))
        fig.update_layout(title='Signal Plot', xaxis_title='Time', yaxis_title='Amplitude')
        return fig

