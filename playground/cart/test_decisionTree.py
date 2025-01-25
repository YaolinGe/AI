from unittest import TestCase
from decisionTree import DecisionTree
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestTestDecisionTree(TestCase):

    def setUp(self) -> None:
        self.tree = DecisionTree(max_depth=4, min_samples_leaf=1)

    def print_metrics(self, Y_test, Y_pred):
        print("*"*56)
        print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")
        print(f"Precision: {precision_score(Y_test, Y_pred, average='macro')}")
        print(f"Recall: {recall_score(Y_test, Y_pred, average='macro')}")
        print(f"F1: {f1_score(Y_test, Y_pred, average='macro')}")
        print("*"*56)

    def test_iris_dataset(self) -> None:
        iris = load_iris()
        X = np.array(iris.data)
        Y = np.array(iris.target)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        self.tree.train(X_train, Y_train)
        self.print_metrics(Y_test, self.tree.predict(X_test))
        self.print_metrics(Y_train, self.tree.predict(X_train))

    def test_random_dataset(self) -> None:
        X = np.random.rand(100, 4)
        Y = np.random.randint(0, 3, 100)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        self.tree.train(X_train, Y_train)
        self.print_metrics(Y_test, self.tree.predict(X_test))
        self.print_metrics(Y_train, self.tree.predict(X_train))

