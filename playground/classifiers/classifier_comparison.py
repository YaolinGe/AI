import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Set Streamlit layout
st.set_page_config(layout="wide", page_title="Classifier Visualization")
st.title("📊 Classifier Performance Visualization")
st.sidebar.title("Settings")

# Available datasets
datasets_dict = {
    "Moons": datasets.make_moons(n_samples=300, noise=0.2, random_state=42),
    "Circles": datasets.make_circles(n_samples=300, noise=0.2, factor=0.5, random_state=42),
    "Blobs": datasets.make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, 
                                          n_clusters_per_class=1, random_state=42)
}

# Dataset selection
dataset_name = st.sidebar.selectbox("Select Dataset", list(datasets_dict.keys()))
X, y = datasets_dict[dataset_name]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifier selection
classifiers = {
    "Nearest Neighbors": KNeighborsClassifier,
    "SVM (Linear)": lambda **kwargs: SVC(kernel='linear', **kwargs),
    "SVM (RBF)": lambda **kwargs: SVC(kernel='rbf', **kwargs),
    "Gaussian Process": GaussianProcessClassifier,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Neural Network": MLPClassifier,
    "AdaBoost": AdaBoostClassifier,
    "Naive Bayes": GaussianNB,
    "QDA": QuadraticDiscriminantAnalysis
}
classifier_name = st.sidebar.selectbox("Select Classifier", list(classifiers.keys()))

# Parameter controls
params = {}
if classifier_name == "Nearest Neighbors":
    params["n_neighbors"] = st.sidebar.slider("n_neighbors", 1, 15, 5)
elif "SVM" in classifier_name:
    params["C"] = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
elif classifier_name in ["Decision Tree", "Random Forest"]:
    params["max_depth"] = st.sidebar.slider("max_depth", 1, 20, 5)
    if classifier_name == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 200, 100)
elif classifier_name == "Neural Network":
    params["hidden_layer_sizes"] = (st.sidebar.slider("Hidden Layers", 10, 200, 100),)
elif classifier_name == "AdaBoost":
    params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 200, 50)

# Train classifier
clf = classifiers[classifier_name](**params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Plot decision boundary
def plot_decision_boundary(X, y, model):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='RdBu')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Decision Boundary of {classifier_name}")
    st.pyplot(plt)

# Display metrics
st.sidebar.subheader("Performance Metrics")
st.sidebar.write(f"**Accuracy:** {accuracy:.4f}")
st.sidebar.write(f"**Precision:** {report['weighted avg']['precision']:.4f}")
st.sidebar.write(f"**Recall:** {report['weighted avg']['recall']:.4f}")
st.sidebar.write(f"**F1-score:** {report['weighted avg']['f1-score']:.4f}")

# Show confusion matrix
st.sidebar.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.sidebar.pyplot(fig)

# Render decision boundary
plot_decision_boundary(X, y, clf)
