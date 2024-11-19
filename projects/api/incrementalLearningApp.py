"""
This flask API will handle the model construction and training and using the model for prediction.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2024-10-15
"""

from flask import Flask, send_file, render_template, request, Response, jsonify, redirect, url_for
from flask_cors import CORS
import torch 
import logging 
import os
import base64
import asyncio
import time
import netron
import numpy as np
from LSTMAutoEncoder import LSTMAutoEncoder


class IncrementalLearningApp:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.initialize_app()

    def initialize_app(self):
        self.app.add_url_rule("/", "index", self.index, methods=['GET', 'POST'])

    def index(self):
        if request.method == 'POST':
            form_data = request.form
            layers = []
            for key in form_data.keys():
                if key.startswith('input_dim_'):
                    layer_num = key.split('_')[-1]
                    layers.append({
                        'input_dim': form_data[f'input_dim_{layer_num}'],
                        'hidden_dim': form_data[f'hidden_dim_{layer_num}'],
                        'output_dim': form_data[f'output_dim_{layer_num}']
                    })
            print(layers)
            # Process layers data here
            return render_template('index.html', form_data=form_data)
        return render_template('index.html')


    def run(self, debug: bool = False):
        self.app.run(host='0.0.0.0', port=1111, debug=debug)


if __name__ == '__main__':
    incremental_learning_app = IncrementalLearningApp()
    incremental_learning_app.run(debug=True)
