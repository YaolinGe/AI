import numpy as np 
from scipy import stats
import streamlit as st
import plotly.express as px


class KNN:
    """
    K-Nearest Neighbours Classifier
    
    Parameters:
    -----------
    k : int, default=3
        Number of neighbors to use for classification
    metric : str, default='euclidean'
        Distance metric to use. Supported: 'euclidean', 'manhattan', 'minkowski'
    p : int, optional
        Power parameter for Minkowski distance (only used when metric='minkowski')
    """
    def __init__(self, k=3, metric='euclidean', p=2):
        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None
    
    def _euclidean(self, v1, v2):
        """Compute Euclidean distance between two vectors"""
        return np.sqrt(np.sum((v1 - v2)**2))
    
    def _manhattan(self, v1, v2):
        """Compute Manhattan distance between two vectors"""
        return np.sum(np.abs(v1 - v2))
    
    def _minkowski(self, v1, v2, p=None):
        """Compute Minkowski distance between two vectors"""
        p = self.p if p is None else p
        return np.sum(np.abs(v1 - v2)**p)**(1/p)
    
    def _compute_distance(self, v1, v2):
        """Compute distance based on the specified metric"""
        if self.metric == 'euclidean':
            return self._euclidean(v1, v2)
        elif self.metric == 'manhattan':
            return self._manhattan(v1, v2)
        elif self.metric == 'minkowski':
            return self._minkowski(v1, v2)
        else:
            raise ValueError('Supported metrics are euclidean, manhattan, and minkowski')
    
    def fit(self, X_train, y_train):
        """
        Store the training data
        
        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
        """
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        return self
    
    def _get_neighbours(self, test_row):
        """
        Find k nearest neighbours for a test row
        
        Parameters:
        -----------
        test_row : array-like
            A single test sample
        
        Returns:
        --------
        list of indices of k nearest neighbours
        """
        # Compute distances to all training points
        distances = [
            (self._compute_distance(train_row, test_row), i) 
            for i, train_row in enumerate(self.X_train)
        ]
        
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        
        # Return indices of k nearest neighbours
        return [dist_index for _, dist_index in distances[:self.k]]
    
    def predict(self, X_test):
        """
        Predict classes for test samples
        
        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples
        
        Returns:
        --------
        array of predicted classes
        """
        # Ensure X_test is a numpy array
        X_test = np.asarray(X_test)
        
        # Handle single sample case
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        # Predictions for each test sample
        preds = []
        for test_row in X_test:
            # Get indices of k nearest neighbours
            neighbour_indices = self._get_neighbours(test_row)
            
            # Get classes of neighbours
            neighbour_classes = self.y_train[neighbour_indices]
            
            # Predict by majority vote
            pred = stats.mode(neighbour_classes, keepdims=False)[0]
            preds.append(pred)
        
        return np.array(preds)


def main():
    """Streamlit app for KNN demonstration"""
    st.title("K-Nearest Neighbours Demonstration")
    
    # Sidebar for user input
    k = st.sidebar.slider("Number of Neighbours (k)", min_value=1, max_value=20, value=3)
    metric = st.sidebar.selectbox("Distance Metric", options=['euclidean', 'manhattan', 'minkowski'])
    p = st.sidebar.slider("Minkowski p value", min_value=1, max_value=5, value=2) if metric == 'minkowski' else 2
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data
    X_train = np.random.rand(50, 2)
    y_train = np.random.choice([0, 1], size=50)
    X_test = np.random.rand(1, 2)
    
    # Fit the KNN model
    knn = KNN(k=k, metric=metric, p=p)
    knn.fit(X_train, y_train)
    
    # Predict
    y_pred = knn.predict(X_test)
    
    # Get neighbour indices
    neighbours_indices = knn._get_neighbours(X_test[0])
    
    # Plot the results
    fig = px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=y_train.astype(str), 
                     title="Training Data Visualization",
                     labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Class'})
    
    # Add test point
    fig.add_scatter(x=X_test[:, 0], y=X_test[:, 1], mode='markers', 
                    marker=dict(color='red', size=10), name='Test Data')
    
    # Add lines to nearest neighbors
    for neighbour_index in neighbours_indices:
        fig.add_scatter(x=[X_test[0, 0], X_train[neighbour_index, 0]], 
                        y=[X_test[0, 1], X_train[neighbour_index, 1]], 
                        mode='lines', 
                        line=dict(color='blue', width=1), 
                        showlegend=False)
    
    st.plotly_chart(fig)
    
    # Display predictions and details
    st.write("Test Point:", X_test)
    st.write("Predicted Label:", y_pred[0])
    st.write(f"Neighbours (k={k}, metric={metric}):", 
             [X_train[index] for index in neighbours_indices])
    
    # Display indices of neighbours for verification
    st.write("Neighbour Indices:", neighbours_indices)


if __name__ == "__main__":
    main()