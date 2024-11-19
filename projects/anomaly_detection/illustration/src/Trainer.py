"""
Trainer class for training the model

Author: Yaolin Ge
Date: 2024-08-30
"""
import matplotlib.pyplot as plt
import torch 
import time
import torch.nn as nn
import torch.optim as optim


class Trainer:
    
    def __init__(self, model, train_loader, test_loader) -> None: 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.num_epochs = 150
        self.train_loss = []
        self.test_loss = []

    def run(self, verbose: bool=False, display: bool=False) -> None: 
        for epoch in range(self.num_epochs): 
            start_time = time.time()
            self.model.train()
            train_loss = 0
            for X, y in self.train_loader: 
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            self.train_loss.append(train_loss / len(self.train_loader))

            test_loss = self.evaluate()
            self.test_loss.append(test_loss)
            if epoch % 10 == 0:
                if verbose:
                    elapsed_time = time.time() - start_time
                    elapsed_time = time.time() - start_time
                    elapsed_minutes = elapsed_time / 60
                    print(f"Epoch {epoch}: Train loss: {train_loss / len(self.train_loader)}, Test loss: {test_loss}, Elapsed time: {elapsed_time} seconds, Elapsed time: {elapsed_minutes} minutes, Remaining time: {elapsed_minutes * (self.num_epochs - epoch - 1):.2f} minutes, {elapsed_minutes * (self.num_epochs - epoch - 1) * 60:.2f} seconds")
                else: 
                    print(f"Epoch {epoch}: Train loss: {train_loss / len(self.train_loader)}, Test loss: {test_loss}")
        if display:
            self.display()

    def evaluate(self) -> float:
        self.model.eval()
        test_loss = 0
        with torch.no_grad(): 
            for X, y in self.test_loader: 
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                test_loss += loss.item()
        return test_loss / len(self.test_loader)
    
    def run_autoencoder(self, verbose: bool=False, display: bool=False) -> None:
        for epoch in range(self.num_epochs): 
            start_time = time.time()
            self.model.train()
            train_loss = 0
            for X in self.train_loader: 
                X = X[0].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, X)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            self.train_loss.append(train_loss / len(self.train_loader))

            test_loss = self.evaluate_autoencoder()
            self.test_loss.append(test_loss)
            if epoch % 10 == 0:
                if verbose:
                    elapsed_time = time.time() - start_time
                    elapsed_time = time.time() - start_time
                    elapsed_minutes = elapsed_time / 60
                    print(f"Epoch {epoch}: Train loss: {train_loss / len(self.train_loader)}, Test loss: {test_loss}, Elapsed time: {elapsed_time} seconds, Elapsed time: {elapsed_minutes} minutes, Remaining time: {elapsed_minutes * (self.num_epochs - epoch - 1):.2f} minutes, {elapsed_minutes * (self.num_epochs - epoch - 1) * 60:.2f} seconds")
                else: 
                    print(f"Epoch {epoch}: Train loss: {train_loss / len(self.train_loader)}, Test loss: {test_loss}")
        if display:
            self.display()

    def evaluate_autoencoder(self) -> float:
        self.model.eval()
        test_loss = 0
        with torch.no_grad(): 
            for X in self.test_loader: 
                X = X[0].to(self.device)
                output = self.model(X)
                loss = self.criterion(output, X)
                test_loss += loss.item()
        return test_loss / len(self.test_loader)

    def display(self) -> None:
        plt.plot(self.train_loss, label='Train')
        plt.plot(self.test_loss, label='Test')
        plt.legend()
        plt.show()