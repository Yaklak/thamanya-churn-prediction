# Model preprocessing that works for LogisticRegression, DecisionTree, RandomForest
# - Encodes categoricals with OneHotEncoder(handle_unknown="ignore")
# - Imputes numerics with median and categoricals with most_frequent
# - Optionally scales numeric features (harmless for trees, helpful for LR)

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.io_utils import save_csv

LEAKAGE_COLS = [
    # IDs / timestamps that should not be fed to the model
    "userId",
    "sessionId",
    "registration",
    "first_ts",
    "last_ts",
    "ts",
]


def _drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with a single unique value (including NaN).

    This is robust to tiny samples and mirrors the exploration in the notebook.
    """
    nunique = df.nunique(dropna=False)
    return df.loc[:, nunique > 1]


def preprocess(
    df: pd.DataFrame,
    target: str,
    drop_cols: Iterable[str] | None = None,
    scale_numeric: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Prepare features for classical ML models.

    Parameters
    ----------
    df : pd.DataFrame
        Input data at the *user-level* after feature engineering.
    target : str
        Column name of the binary target.
    drop_cols : list-like, optional
        Extra columns to drop if present (e.g., raw text or debug fields).
    scale_numeric : bool, default True
        Whether to standardize numeric features. This benefits LogisticRegression
        and is neutral for tree models (DecisionTree/RandomForest).

    Returns
    -------
    X : pd.DataFrame
        Feature dataframe (pre-encoding) to fit a ColumnTransformer on.
    y : pd.Series
        Target vector as integers {0,1}.
    fe : ColumnTransformer
        Transformer that imputes/encodes and can be plugged into a Pipeline.
    """

    # 1) Basic hygiene
    df = _drop_constant_columns(df.copy())

    # Drop obvious leakage / identifier columns if present
    to_drop = set(LEAKAGE_COLS)
    if drop_cols:
        to_drop.update(drop_cols)
    df = df.drop(columns=list(to_drop & set(df.columns)), errors="ignore")

    # 2) Split target / features
    y = df[target].astype(int)
    X = df.drop(columns=[target], errors="ignore")

    # 3) Column typing
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    # 4) Build preprocessing for each type
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        # with_mean=False keeps compatibility with sparse stacks if needed later
        num_steps.append(("scale", StandardScaler(with_mean=False)))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    fe = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    save_csv(X, "customer_churn_model_features", "data/processed")

    return X, y, fe
