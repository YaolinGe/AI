# import numpy as np 
# import matplotlib.pyplot as plt 


# class SVM:
#     def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
#         self.lr = learning_rate
#         self.lambda_param = lambda_param  # Regularization parameter
#         self.n_iters = n_iters
#         self.w = None
#         self.b = None

#     def fit(self, X, y):
#         n_samples, n_features = X.shape
        
#         # Initialize weights and bias
#         self.w = np.zeros(n_features)
#         self.b = 0

#         # Convert labels into +1 and -1
#         y_ = np.where(y <= 0, -1, 1)

#         for _ in range(self.n_iters):
#             for idx, x_i in enumerate(X):
#                 # Check if the current sample satisfies the margin condition
#                 condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
#                 if condition:
#                     # Gradient for correct classification
#                     self.w -= self.lr * (2 * self.lambda_param * self.w)
#                 else:
#                     # Gradient for incorrect classification
#                     self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
#                     self.b -= self.lr * y_[idx]

#     def predict(self, X):
#         linear_output = np.dot(X, self.w) + self.b
#         return np.sign(linear_output)

# def plot_decision_boundary(X, y, model): 
#     x0 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
#     x1 = - (model.w[0] * x0 + model.b) / model.w[1]

#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", alpha=.8)
#     plt.plot(x0, x1, color="k", lw=2)
#     plt.xlabel("F1")
#     plt.ylabel("F2")
#     plt.title("SVM decision boundary")
#     plt.show()


# if __name__ == "__main__": 
#     from sklearn.datasets import make_blobs, make_moons, make_circles
#     # X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.05)
#     # X, y = make_moons(n_samples=100, noise=0.1)
#     # X, y = make_circles(n_samples=100, noise=0.1, factor=0.1)
#     y = np.where(y == 0, -1, 1)
#     from sklearn.model_selection import train_test_split 
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#     clf = SVM(learning_rate=.001, lambda_param=0.01, n_iters=1000)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     accuracy = np.mean(y_pred == y_test)
#     print(f"Accuracy: {accuracy * 100:.2f}%")

#     plot_decision_boundary(X, y, clf)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the SVM class
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.history = []  # Store weights for visualization

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
            # Save weights and bias at each iteration
            self.history.append((self.w.copy(), self.b))

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# Generate a toy dataset
np.random.seed(42)
X = np.vstack((np.random.randn(50, 2) - [2, 2], np.random.randn(50, 2) + [2, 2]))
y = np.hstack((np.ones(50), -np.ones(50)))

# Train the SVM
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=200)
svm.fit(X, y)

# Visualization function
def plot_decision_boundary(w, b, ax):
    x0 = np.linspace(-4, 4, 100)
    x1 = -(w[0] * x0 + b) / w[1]
    ax.plot(x0, x1, 'k-')

# Animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
line, = ax.plot([], [], 'k-', lw=2)

def update(frame):
    # ax.collections.clear()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30)
    w, b = svm.history[frame]
    x0 = np.linspace(-4, 4, 100)
    x1 = -(w[0] * x0 + b) / w[1]
    line.set_data(x0, x1)
    return line,

ani = FuncAnimation(fig, update, frames=len(svm.history), interval=50, blit=True)
plt.show()
