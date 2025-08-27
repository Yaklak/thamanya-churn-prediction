import pytest
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from src.utils.metrics import evaluate_all


class DummyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, proba=0.8):
        self.proba = proba

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.full((len(X), 2), [1 - self.proba, self.proba])

    def predict(self, X):
        return np.full(len(X), int(self.proba > 0.5))


@pytest.fixture
def dummy_model():
    return DummyClassifier()


@pytest.fixture
def sample_data():
    X = pd.DataFrame(np.random.rand(100, 4), columns=list("ABCD"))
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y


def test_evaluate_all(dummy_model, sample_data):
    """Test the evaluate_all function."""
    X_te, y_te = sample_data
    dummy_model.fit(X_te, y_te)

    metrics = evaluate_all(
        dummy_model, X_te, y_te, ["roc_auc", "average_precision", "f1"]
    )

    assert "roc_auc" in metrics
    assert "average_precision" in metrics
    assert "f1" in metrics

    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["average_precision"] <= 1
    assert 0 <= metrics["f1"] <= 1
