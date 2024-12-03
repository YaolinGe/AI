"""
Supervised Learning Page to train and evaluate models.

Created on 2024-11-28
Author: Yaolin Ge
Email: geyaolin@gmail.com
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.Logger import Logger


class SupervisedLearningPage:
    def __init__(self):
        self.logger = Logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initialized Supervised Learning Page PLayground, using: {self.device}")
        self.reset_parameters()

    def reset_parameters(self):
        # Dataset parameters
        self.dataset_name = "Iris"
        self.task_type = "Classification"
        
        # Model hyperparameters
        self.learning_rate = 0.01
        self.epochs = 100
        self.batch_size = 32
        self.optimizer_name = "Adam"
        self.loss_function_name = "CrossEntropy"
        self.regularization = False
        
        # Data and model attributes
        self.X = None
        self.y = None
        self.model = None
        self.loss_history = []
        self.logger.info("Parameters reset to default values: Dataset: Iris; Task: Classification; Learning Rate: 0.01; Epochs: 100; Batch Size: 32; Optimizer: Adam; Loss Function: CrossEntropy; Regularization: False")

    def load_dataset(self):
        """Load and preprocess datasets."""
        datasets = {
            "Iris": load_iris(return_X_y=True),
            "Breast Cancer": load_breast_cancer(return_X_y=True),
            "Blobs Cluster": make_blobs(n_samples=300, centers=3, random_state=42)
        }
        
        self.X, self.y = datasets[self.dataset_name]
        
        # Standardize features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
        # Convert to PyTorch tensors
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)

    def create_model(self):
        """Create flexible neural network based on task and dataset."""
        input_dim = self.X.shape[1]
        
        if self.task_type == "Classification":
            output_dim = len(np.unique(self.y))
            model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
        elif self.task_type == "Regression":
            output_dim = 1
            model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
        elif self.task_type == "Clustering":
            output_dim = 3  # Cluster centroids
            model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
        
        return model.to(self.device)

    def select_loss_function(self):
        """Select loss function based on task."""
        loss_functions = {
            "CrossEntropy": nn.CrossEntropyLoss(),
            "MSE": nn.MSELoss(),
            "L1": nn.L1Loss()
        }
        return loss_functions[self.loss_function_name]

    def select_optimizer(self, model):
        """Select optimizer with optional L2 regularization."""
        optimizers = {
            "SGD": optim.SGD,
            "Adam": optim.Adam,
            "RMSprop": optim.RMSprop
        }
        
        optimizer_cls = optimizers[self.optimizer_name]
        
        if self.regularization:
            return optimizer_cls(model.parameters(), 
                                 lr=self.learning_rate, 
                                 weight_decay=0.001)
        else:
            return optimizer_cls(model.parameters(), 
                                 lr=self.learning_rate)

    def train_model(self):
        """Generic training loop for different tasks."""
        self.load_dataset()
        self.model = self.create_model()
        
        loss_fn = self.select_loss_function()
        optimizer = self.select_optimizer(self.model)
        
        self.loss_history = []
        
        for epoch in range(self.epochs):
            # Forward pass
            outputs = self.model(self.X)
            loss = loss_fn(outputs, self.y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.loss_history.append(loss.item())
    
    def plot_loss(self):
        """Create interactive loss plot."""
        return go.Figure(
            data=[go.Scatter(
                x=list(range(len(self.loss_history))), 
                y=self.loss_history, 
                mode='lines',
                name='Loss'
            )],
            layout=go.Layout(
                title='Training Loss',
                xaxis={'title': 'Epochs'},
                yaxis={'title': 'Loss'}
            )
        )

    def render_sidebar(self):
        """Render Streamlit sidebar with controls."""
        st.sidebar.title("ML Playground Controls")
        
        # Task selection
        self.task_type = st.sidebar.selectbox(
            "Select Task", 
            ["Classification", "Regression", "Clustering"]
        )
        
        # Dataset selection
        self.dataset_name = st.sidebar.selectbox(
            "Select Dataset", 
            ["Iris", "Breast Cancer", "Blobs Cluster"]
        )
        
        # Hyperparameter controls
        self.learning_rate = st.sidebar.slider(
            "Learning Rate", 0.0001, 0.1, 0.01, step=0.001
        )
        
        self.epochs = st.sidebar.slider(
            "Epochs", 10, 500, 100
        )
        
        # Optimizer and Loss Function
        self.optimizer_name = st.sidebar.selectbox(
            "Optimizer", ["SGD", "Adam", "RMSprop"]
        )
        
        self.loss_function_name = st.sidebar.selectbox(
            "Loss Function", 
            ["CrossEntropy", "MSE", "L1"]
        )
        
        self.regularization = st.sidebar.checkbox(
            "L2 Regularization", False
        )

    def render(self):
        """Main method to run Streamlit app."""
        self.render_sidebar()
        
        # Train button
        if st.button("Train Model"):
            with st.spinner("Training..."):
                self.train_model()
            
            st.subheader("Training Results")
            st.plotly_chart(self.plot_loss())

# def main():
#     st.title("ML Playground: Multi-Task Learning")
#     app = MLPlayground()
#     app.run()

# if __name__ == "__main__":
#     main()