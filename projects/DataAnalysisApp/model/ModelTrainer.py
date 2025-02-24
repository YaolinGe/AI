"""
This script defines a machine learning pipeline for training and evaluating an LSTM model on time series data.
The main functionalities include:
1. Loading and preprocessing time series data from a CSV file.
2. Creating sequences from the time series data for LSTM input.
3. Splitting the data into training, validation, and test sets.
4. Defining and training an LSTM model using PyTorch.
5. Evaluating the trained model on the test set and calculating performance metrics.
6. Exporting the trained model and configuration to an ONNX file for deployment.
Classes:
    TimeSeriesDataset: A custom PyTorch Dataset for handling time series data.
    LSTM: A PyTorch LSTM model for binary classification.
    ModelTrainer: A class that encapsulates the entire training pipeline, including data preprocessing, model training, evaluation, and export.
Functions:
    load_config: Loads the configuration for the training pipeline from a JSON file.
    preprocess_data: Preprocesses the input data by scaling features.
    create_sequences: Creates sequences and targets from the preprocessed data.
    split_data: Splits the sequences and targets into training, validation, and test sets.
    train_lstm: Trains the LSTM model on the training data and validates it on the validation data.
    evaluate_model: Evaluates the trained model on the test data and calculates performance metrics.
    export_model: Exports the trained model and configuration to an ONNX file.
    run_training_pipeline: Runs the entire training pipeline and saves the trained model and metrics.
Usage:
    Run the script from the command line with the required arguments:
    python ModelTrainer.py --data_path <path_to_csv> [--config_path <path_to_config>] [--output_dir <output_directory>]
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import onnx
import onnxruntime
import json
import os
from datetime import datetime


class Encoder(nn.Module):
    def __init__(self, input_size: int=7, hidden_size: int=128, latent_size: int=16,
                 num_layers: int=1, batch_first: bool=True):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=latent_size,
            num_layers=num_layers,
            batch_first=batch_first
        )

    def forward(self, x):
        x, (hidden, cell) = self.lstm1(x)
        x, (hidden, cell) = self.lstm2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size: int=16, hidden_size: int=128, output_size: int=7,
                 num_layers: int=1, batch_first: bool=True):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=output_size,
            num_layers=num_layers,
            batch_first=batch_first
        )

    def forward(self, x):
        x, (hidden, cell) = self.lstm1(x)
        x, (hidden, cell) = self.lstm2(x)
        return x


class LSTMAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ModelTrainer:
    def __init__(self, data_path, config_path=None):
        self.data_path = data_path
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_config(self, config_path):
        default_config = {
            'raw_columns': ['x2g', 'y2g', 'z2g', 'x50g', 'y50g', 'strain0', 'strain1'],
            'ignore_columns': ['timestamp', 'load', 'deflection', 'surfacefinish', 'vibration'],
            'target_columns': ['Anomaly'],
            'sequence_length': 30,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 100,
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config

    def preprocess_data(self):
        # Load and preprocess data
        df = pd.read_csv(self.data_path)
        
        # Scale features
        scaler = MinMaxScaler()
        df[self.config['raw_columns']] = scaler.fit_transform(df[self.config['raw_columns']])

        # Calculate first difference
        df[self.config['raw_columns']] = df[self.config['raw_columns']].diff()
        df = df.dropna()

        return df

    def create_sequences(self, data):
        sequences = []
        
        for i in range(len(data) - self.config['sequence_length']):
            seq = data[self.config['raw_columns']].values[i:i + self.config['sequence_length']]
            sequences.append(seq)
            
        return np.array(sequences)

    def split_data(self, sequences):
        n = len(sequences)
        train_end = int(n * self.config['train_split'])
        val_end = train_end + int(n * self.config['val_split'])
        
        train_seq = sequences[:train_end]
        val_seq = sequences[train_end:val_end]
        test_seq = sequences[val_end:]
        
        return train_seq, val_seq, test_seq

    def train_autoencoder(self, train_data, val_data):
        train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32))
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])

        model = LSTMAutoEncoder().to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        
        best_val_loss = float('inf')
        best_model = None
        
        for epoch in range(self.config['num_epochs']):
            model.train()
            train_loss = 0
            for batch_X in train_loader:
                batch_X = batch_X[0].to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_X)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X in val_loader:
                    batch_X = batch_X[0].to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_X)
                    val_loss += loss.item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}')
        
        model.load_state_dict(best_model)
        return model

    def evaluate_model(self, model, test_data, threshold=0.5):
        model.eval()
        test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'])
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X in test_loader:
                batch_X = batch_X[0].to(self.device)
                outputs = model(batch_X)
                loss = torch.mean((outputs - batch_X) ** 2, dim=(1, 2))
                preds = (loss > threshold).int()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_X.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds),
            'recall': recall_score(all_targets, all_preds),
            'f1': f1_score(all_targets, all_preds)
        }
        
        return metrics

    def export_model(self, model, output_dir):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export PyTorch model
        dummy_input = torch.randn(1, self.config['sequence_length'], len(self.config['raw_columns'])).to(self.device)
        torch.onnx.export(
            model,
            dummy_input,
            f"{output_dir}/lstm.onnx",
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # Save configuration
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=4)

    def run_training_pipeline(self, output_dir='models'):
        # Create timestamp for model version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}/{timestamp}"
        
        # Preprocess data
        df = self.preprocess_data()
        
        # Create sequences
        sequences = self.create_sequences(df)
        
        # Split data
        train_data, val_data, test_data = self.split_data(sequences)
        
        # Train model
        model = self.train_autoencoder(train_data, val_data)
        
        # Evaluate model
        metrics = self.evaluate_model(model, test_data)
        
        # Export model and metrics
        self.export_model(model, output_dir)
        with open(f"{output_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return model, metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to input CSV file')
    parser.add_argument('--config_path', help='Path to configuration JSON file')
    parser.add_argument('--output_dir', default='models', help='Output directory for models')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(args.data_path, args.config_path)
    model, metrics = trainer.run_training_pipeline(args.output_dir)
    print("Training complete. Final metrics:", metrics) 