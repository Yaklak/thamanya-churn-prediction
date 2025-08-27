from src.features.preprocess import preprocess
from src.features.build_features import build_user_features
import pandas as pd


def test_preprocess(raw_events_df):
    """Test the preprocess function."""
    # First, build features
    features_df = build_user_features(raw_events_df, inactivity_days=30, target="churn")

    # Then, preprocess
    X, y, transformer = preprocess(features_df, target="churn")

    # Check shapes
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == 2  # 2 users

    # Check y
    assert isinstance(y, pd.Series)
    assert pd.api.types.is_integer_dtype(y)

    # Check X
    assert isinstance(X, pd.DataFrame)
    assert "userId" not in X.columns  # check that id columns are dropped

    # Check transformer
    assert transformer is not None
    assert hasattr(transformer, "transformers")
