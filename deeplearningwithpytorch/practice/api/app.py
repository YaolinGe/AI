from flask import Flask, send_file, render_template, request, Response, jsonify, redirect, url_for
from flask_cors import CORS
from model.LSTMAutoEncoder import LSTMAutoEncoder
import torch 
import logging 
import os
import base64
import asyncio
import time
import netron


class IncrementalLearningApp:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.initialize_app()
        self.model_path = 'test.onnx'
        self.host_address = 'http://127.0.0.3:1234/'

    def initialize_app(self):
        self.app.add_url_rule("/", "index", self.index, methods=['GET', 'POST'])

    def index(self):
        if request.method == 'POST':
              print(request.form)
              self.seq_len = int(request.form['seq_len'])
              self.input_dim = int(request.form['input_dim'])
              self.hidden_dim1 = int(request.form['hidden_dim1'])
              self.hidden_dim2 = int(request.form['hidden_dim2'])
              self.output_dim = int(request.form['output_dim'])
              self.model = LSTMAutoEncoder(self.input_dim, self.hidden_dim1, self.hidden_dim2, self.output_dim)
              self.model.export(self.model_path)
              netron.serve(file=self.model_path, address=('127.0.0.3', 1234), browse=False, verbosity=0)
              return render_template('index.html', iframe_url=self.host_address)
        return render_template('index.html', iframe_url=self.host_address)

    def run(self, debug: bool = False):
        self.app.run(host='0.0.0.0', port=8123, debug=debug)


if __name__ == '__main__':
    incremental_learning_app = IncrementalLearningApp()
    incremental_learning_app.run(debug=True)
