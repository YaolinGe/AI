"""
This script defines a machine learning pipeline for training and evaluating classical ML models on time series data.
The main functionalities include:
1. Loading and preprocessing time series data from a CSV file
2. Creating time series features (lags, moving averages)
3. Training multiple classical models (XGBoost, Random Forest, etc.)
4. Evaluating models and selecting the best performing one
5. Exporting models and metrics for deployment

The pipeline supports:
- Feature engineering: lag features, moving averages, standard deviations
- Multiple classical models: XGBoost, Random Forest, Gradient Boosting, etc.
- Model evaluation with various metrics
- Model export to ONNX format
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
import json
import os
from datetime import datetime
import onnxmltools
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


class ClassicalModelTrainer:
    def __init__(self, data_path, config_path=None):
        self.data_path = data_path
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path):
        default_config = {
            'raw_columns': ['x2g', 'y2g', 'z2g', 'x50g', 'y50g', 'strain0', 'strain1'],
            'ignore_columns': ['timestamp', 'load', 'deflection', 'surfacefinish', 'vibration'],
            'target_columns': ['Anomaly'],
            'lag_features': [5, 10, 15],  # Number of lag steps
            'ma_windows': [30, 60],  # Window sizes for moving averages
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'models': {
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 3,
                    'learning_rate': 0.1
                },
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2
                },
                'gradient_boosting': {
                    'n_estimators': 100,
                    'max_depth': 3,
                    'learning_rate': 0.1
                },
                'gaussian_nb': {}
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config

    def preprocess_data(self):
        """Load and preprocess the data"""
        df = pd.read_csv(self.data_path)
        
        # Scale features
        scaler = MinMaxScaler()
        df[self.config['raw_columns']] = scaler.fit_transform(df[self.config['raw_columns']])

        # first difference
        df[self.config['raw_columns']] = df[self.config['raw_columns']].diff().dropna(inplace=True)

        return df
    
    def create_features(self, df):
        """Create time series features including lags and moving averages"""
        df_features = df.copy()
        
        # Create lag features
        for col in self.config['raw_columns']:
            for lag in self.config['lag_features']:
                df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)
        
        # Create moving average features
        for col in self.config['raw_columns']:
            for window in self.config['ma_windows']:
                df_features[f'{col}_ma_{window}'] = df_features[col].rolling(window=window).mean()
                # df_features[f'{col}_std_{window}'] = df_features[col].rolling(window=window).std()
        
        # Drop rows with NaN values created by lag/rolling features
        df_features.dropna(inplace=True)
        df_features = df_features.drop(columns=self.config['ignore_columns'])
        return df_features
    
    def split_data(self, df):
        """Split data into train, validation and test sets"""
        n = len(df)
        train_end = int(n * self.config['train_split'])
        val_end = train_end + int(n * self.config['val_split'])
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        feature_cols = [col for col in df.columns if col not in self.config['target_columns']]
        
        X_train = train_df[feature_cols]
        y_train = train_df[self.config['target_columns'][0]]
        X_val = val_df[feature_cols]
        y_val = val_df[self.config['target_columns'][0]]
        X_test = test_df[feature_cols]
        y_test = test_df[self.config['target_columns'][0]]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train_models(self, train_data, val_data, verbose=False):
        """Train multiple classical models and return the best one"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        models = {
            'xgboost': xgb.XGBClassifier(**self.config['models']['xgboost']),
            # 'random_forest': RandomForestClassifier(**self.config['models']['random_forest']),
            # 'gradient_boosting': GradientBoostingClassifier(**self.config['models']['gradient_boosting']),
            'gaussian_nb': GaussianNB(**self.config['models']['gaussian_nb']),
        }
        
        best_model = None
        best_score = -1
        best_model_name = None
        model_metrics = {}
        
        for name, model in models.items():
            if verbose:
                print(f"Training {name} model...")
                
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred)
            }
            model_metrics[name] = metrics
            
            if verbose:
                print(f"Metrics for {name}: {metrics}")
            
            # Track best model based on F1 score
            if metrics['f1'] > best_score:
                best_score = metrics['f1']
                best_model = model
                best_model_name = name
        
        return best_model, best_model_name, model_metrics
    
    def evaluate_model(self, model, test_data):
        """Evaluate the model on test data"""
        X_test, y_test = test_data
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return metrics
    
    def export_model(self, model, model_name, feature_names, output_dir):
        """Export model to ONNX format"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create initial type for ONNX conversion
        initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
        
        # Convert to ONNX
        if model_name == 'xgboost':
            onx = onnxmltools.convert_xgboost(model, initial_types=initial_type)
        else:
            onx = convert_sklearn(model, initial_types=initial_type)
        
        # Save model
        with open(f"{output_dir}/{model_name}.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        
        # Save feature names and configuration
        model_config = {
            'feature_names': feature_names,
            'model_config': self.config,
            'model_type': model_name
        }
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(model_config, f, indent=4)
    
    def run_training_pipeline(self, output_dir='models'):
        """Run the complete training pipeline"""
        # Create timestamp for model version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}/{timestamp}"
        
        # Preprocess data
        df = self.preprocess_data()
        
        # Create features
        df_features = self.create_features(df)
        
        # Split data
        train_data, val_data, test_data = self.split_data(df_features)
        
        # Train models and select best
        best_model, best_model_name, all_metrics = self.train_models(train_data, val_data, verbose=True)
        
        # Evaluate on test set
        test_metrics = self.evaluate_model(best_model, test_data)
        
        # Export model and metrics
        feature_names = [col for col in df_features.columns 
                        if col not in self.config['target_columns']]
        self.export_model(best_model, best_model_name, feature_names, output_dir)
        
        # Save metrics
        metrics = {
            'validation_metrics': all_metrics,
            'test_metrics': test_metrics,
            'best_model': best_model_name
        }
        with open(f"{output_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return best_model, metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to input CSV file')
    parser.add_argument('--config_path', help='Path to configuration JSON file')
    parser.add_argument('--output_dir', default='models', help='Output directory for models')
    
    args = parser.parse_args()
    
    trainer = ClassicalModelTrainer(args.data_path, args.config_path)
    model, metrics = trainer.run_training_pipeline(args.output_dir)
    print("Training complete. Final metrics:", metrics) 