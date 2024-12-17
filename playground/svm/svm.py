import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from tqdm import tqdm

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel='linear', gamma=1):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.gamma = gamma
        self.w = None
        self.b = None
        self.X = None
        self.y = None
        self.alphas = None
        self.history = []

    def kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == 'poly':
            return (1 + np.dot(x1, x2)) ** 2

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X = X
        self.y = y
        self.alphas = np.zeros(n_samples)
        self.b = 0

        # Compute kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel_function(X[i], X[j])

        for _ in range(self.n_iters):
            for i in range(n_samples):
                # Calculate decision function
                decision = np.sum(self.alphas * y * K[i])
                
                # Update alphas
                if y[i] * decision < 1:
                    self.alphas[i] += self.lr * (y[i] - self.lambda_param * self.alphas[i])
                    self.alphas[i] = max(0, min(self.alphas[i], 1))  # Clip alpha
                else:
                    self.alphas[i] -= self.lr * self.lambda_param * self.alphas[i]
                
                # Update bias
                self.b = np.mean(y - np.sum(self.alphas * y * K, axis=1))

            # Store weights for visualization (only for linear kernel)
            if self.kernel == 'linear':
                self.w = np.sum(self.alphas.reshape(-1,1) * y.reshape(-1,1) * X, axis=0)
                self.history.append((self.w, self.b))
    
    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        
        for i in range(n_samples):
            s = 0
            for alpha, sv_y, sv in zip(self.alphas, self.y, self.X):
                s += alpha * sv_y * self.kernel_function(X[i], sv)
            y_pred[i] = s - self.b
            
        return np.sign(y_pred)


# Generate a toy dataset
X, y = make_blobs(n_samples=500, centers=2, cluster_std=0.6)
y = np.where(y == 0, -1, 1)

# Hyperparameters
learning_rate = 0.01
lambda_param = 0.01
n_iters = 100


# Train the SVM
svm = SVM(learning_rate=learning_rate, lambda_param=lambda_param, n_iters=n_iters)
svm.fit(X, y)

for frame in tqdm(range(len(svm.history))):
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Decision boundary plot
    w, b = svm.history[frame]
    x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x1 = -w[0] / w[1] * x0 - b / w[1]
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30)
    ax1.plot(x0, x1, 'k-')
    ax1.set_title("Decision Boundary")

    # Cost plot
    costs = [np.linalg.norm(w) for w, b in svm.history]
    ax2.plot(costs)
    ax2.set_title("Cost over Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cost")

    fig.savefig("fig/P_{:03d}.png".format(frame))
    plt.close(fig)




