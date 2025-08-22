import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def build(df: pd.DataFrame, target: str, drop_cols: list):
    y = df[target].astype(int)
    X = df.drop(columns=[target] + drop_cols, errors="ignore")
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    fe = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        remainder="passthrough",
    )
    return X, y, fe
