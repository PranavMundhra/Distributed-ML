"""A small deterministic demonstration model; no network or model files."""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def train_model() -> LogisticRegression:
    iris = load_iris()
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(iris.data, iris.target)
    return model
