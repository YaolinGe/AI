"""
FFT Playground Page

A simple Streamlit app for exploring signals and their Fast Fourier Transform (FFT).

Created on 2024-12-04
Author: Yaolin Ge
Email: geyaolin@gmail.com
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class SignalGenerator:
    """Handles generation of various signal types."""
    @staticmethod
    def generate(
        signal_type: str, 
        frequency: float, 
        sampling_rate: float, 
        duration: float, 
        amplitude: float = 1.0, 
        noise_level: float = 0.0
    ) -> tuple:
        """
        Generate different types of signals with optional noise
        
        Args:
            signal_type (str): Type of signal ('Sine', 'Square', 'Sawtooth')
            frequency (float): Signal frequency
            sampling_rate (float): Number of samples per second
            duration (float): Total signal duration in seconds
            amplitude (float, optional): Signal amplitude. Defaults to 1.0.
            noise_level (float, optional): Level of random noise. Defaults to 0.0.
        
        Returns:
            tuple: Time array and signal array
        """
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        
        # Signal generation logic
        if signal_type == 'Sine':
            signal = amplitude * np.sin(2 * np.pi * frequency * t)
        elif signal_type == 'Square':
            signal = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        elif signal_type == 'Sawtooth':
            signal = amplitude * (2 * (frequency * t - np.floor(0.5 + frequency * t)))
        else:
            raise ValueError(f"Unsupported signal type: {signal_type}")
        
        # Add noise
        if noise_level > 0:
            signal += np.random.normal(0, noise_level, t.shape)
        
        return t, signal

class SignalAnalyzer:
    """Handles signal analysis and FFT computations."""
    @staticmethod
    def compute_fft(signal: np.ndarray, sampling_rate: float) -> tuple:
        """
        Perform Fast Fourier Transform on the signal
        
        Args:
            signal (np.ndarray): Input signal array
            sampling_rate (float): Sampling rate of the signal
        
        Returns:
            tuple: Frequency array and magnitude spectrum
        """
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 1 / sampling_rate)[:N//2]
        
        # Compute magnitude spectrum
        magnitude = 2.0/N * np.abs(yf[0:N//2])
        
        return xf, magnitude
    
    def compute_band_passed_signal(signal: np.ndarray, sampling_rate: float, low_freq: float, high_freq: float) -> np.ndarray:
        """
        Perform band-pass filtering on the signal
        
        Args:
            signal (np.ndarray): Input signal array
            sampling_rate (float): Sampling rate of the signal
            low_freq (float): Lower frequency limit
            high_freq (float): Upper frequency limit
        
        Returns:
            np.ndarray: Band-pass filtered signal
        """
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 1 / sampling_rate)

        # Band-pass filtering
        filter_mask = (xf >= low_freq) & (xf <= high_freq)
        filter_mask = filter_mask | (xf <= -low_freq) & (xf >= -high_freq)
        yf_filtered = yf * filter_mask

        magnitude_filtered = 2.0/N * np.abs(yf_filtered)
        signal_filtered = ifft(yf_filtered)

        return xf, magnitude_filtered, signal_filtered

class VisualizationManager:
    """Handles plotting and visualization of signals."""
    @staticmethod
    def plot_signal_and_fft(t: np.ndarray, signal: np.ndarray, 
                             xf: np.ndarray, magnitude: np.ndarray) -> Any:
        """
        Create a figure with signal and its FFT using Plotly
        
        Args:
            t (np.ndarray): Time array
            signal (np.ndarray): Signal array
            xf (np.ndarray): Frequency array
            magnitude (np.ndarray): Magnitude spectrum
        
        Returns:
            Any: Plotly figure with time and frequency domain plots
        """
        # Create subplots
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Signal in Time Domain', 'Magnitude Spectrum'))
        
        # Time domain plot
        fig.add_trace(go.Scatter(x=t, y=signal, mode='lines', name='Signal'), row=1, col=1)
        fig.update_xaxes(title_text='Time (s)', row=1, col=1)
        fig.update_yaxes(title_text='Amplitude', row=1, col=1)
        
        # Frequency domain plot
        fig.add_trace(go.Scatter(x=xf, y=magnitude, mode='lines', name='Magnitude'), row=2, col=1)
        fig.update_xaxes(title_text='Frequency (Hz)', row=2, col=1)
        fig.update_yaxes(title_text='Magnitude', row=2, col=1)
        fig.update_xaxes(range=[0, max(xf)/2], row=2, col=1)  # Limit x-axis to Nyquist frequency
        
        fig.update_layout(height=600, width=800, title_text="Signal and its FFT")
        
        return fig
    
    @staticmethod 
    def plot_band_pass_filtered_signal(t: np.ndarray, signal: np.ndarray,
                                       xf: np.ndarray, magnitude: np.ndarray,
                                       low: float, high: float) -> Any: 
        """
        Create a figure with band-pass filtered signal and its FFT using Plotly

        Args:
            t (np.ndarray): Time array
            signal (np.ndarray): Signal array
            xf (np.ndarray): Frequency array
            magnitude (np.ndarray): Magnitude spectrum
        
        Returns:
            Any: Plotly figure with time and frequency domain plots
        """
        # Create subplots
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Signal in Time Domain', 'Magnitude Spectrum'))
        
        # Time domain plot
        fig.add_trace(go.Scatter(x=t, y=signal, mode='lines', name='Signal'), row=1, col=1)
        fig.update_xaxes(title_text='Time (s)', row=1, col=1)
        fig.update_yaxes(title_text='Amplitude', row=1, col=1)

        # Frequency domain plot
        # fig.add_trace(go.Scatter(x=xf, y=magnitude, mode='lines', name='Magnitude'), row=2, col=1)
        # fig.add_shape(
        #     type="rect",
        #     x0=low, x1=high, y0=0, y1=max(magnitude),
        #     line=dict(color="RoyalBlue"),
        #     fillcolor="LightSkyBlue",
        #     opacity=0.5,
        #     row=2, col=1
        # )
        # fig.update_xaxes(title_text='Frequency (Hz)', row=2, col=1)
        # fig.update_yaxes(title_text='Magnitude', row=2, col=1)
        # fig.update_xaxes(range=[0, max(xf)/2], row=2, col=1)  # Limit x-axis to Nyquist frequency

        fig.update_layout(height=600, width=800, title_text="Signal and its FFT")

        return fig

@dataclass
class SignalParameter:
    """Data class to represent signal parameters with validation."""
    name: str
    type: str
    min_value: float
    max_value: float
    default_value: float
    step: float = 0.1
    
    def create_slider(self, key: str = None) -> Any:
        """
        Create a Streamlit slider for the parameter
        
        Args:
            key (str, optional): Unique key for the slider. Defaults to None.
        
        Returns:
            Any: Slider value
        """
        return st.sidebar.slider(
            self.name, 
            min_value=self.min_value, 
            max_value=self.max_value, 
            value=self.default_value,
            step=self.step,
            key=key
        )

class FFTPlaygroundPage:
    """Main application class for FFT Signal Playground."""
    def __init__(self):
        """Initialize the FFT Playground with default configurations."""
        self.signal_types = ['Sine', 'Square', 'Sawtooth']
        self.signal_parameters = [
            SignalParameter("Frequency (Hz)", float, 1., 100., 10.),
            SignalParameter("Sampling Rate (Hz)", float, 100., 2000., 1000.),
            SignalParameter("Duration (s)", float, 0.1, 5.0, 1.0),
            SignalParameter("Amplitude", float, 0.1, 5.0, 1.0),
            SignalParameter("Noise Level", float, 0.0, 1.0, 0.0, 0.01)
        ]
    
    def render(self):
        """
        Render the Streamlit application interface
        Handles UI creation, signal generation, and visualization
        """
        st.title('FFT Signal Playground')
        
        # Sidebar for signal parameters
        st.sidebar.header('Signal Parameters')
        
        # Signal type selection
        signal_type = st.sidebar.selectbox(
            'Signal Type', 
            self.signal_types, 
            index=0
        )
        
        # Collect parameter values via sliders
        param_values = {
            param.name: param.create_slider() 
            for param in self.signal_parameters
        }
        
        # Generate signal
        t, signal = SignalGenerator.generate(
            signal_type, 
            param_values["Frequency (Hz)"], 
            param_values["Sampling Rate (Hz)"], 
            param_values["Duration (s)"], 
            param_values["Amplitude"], 
            param_values["Noise Level"]
        )
        
        # Perform FFT
        xf, magnitude = SignalAnalyzer.compute_fft(
            signal, 
            param_values["Sampling Rate (Hz)"]
        )
        
        # Plot signal and FFT
        fig = VisualizationManager.plot_signal_and_fft(t, signal, xf, magnitude)
        st.plotly_chart(fig)

        # Band-pass filtering
        st.sidebar.header('Band-pass Filtering')
        low_freq = st.sidebar.slider('Low Frequency (Hz)', 0.0, param_values["Sampling Rate (Hz)"]/2, 1.0, 0.1)
        high_freq = st.sidebar.slider('High Frequency (Hz)', 0.0, param_values["Sampling Rate (Hz)"]/2, param_values["Sampling Rate (Hz)"]/4, 0.1)
        xf, signal_filtered, magnitude_filtered = SignalAnalyzer.compute_band_passed_signal(signal, param_values["Sampling Rate (Hz)"], low_freq, high_freq)
        fig = VisualizationManager.plot_band_pass_filtered_signal(t, signal_filtered, xf, magnitude_filtered, low_freq, high_freq)
        st.plotly_chart(fig)


if __name__ == "__main__":
    fftPage = FFTPlaygroundPage()
    fftPage.render()