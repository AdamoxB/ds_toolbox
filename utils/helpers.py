import pandas as pd

def split_features_target(df: pd.DataFrame, target_col: str):
    """Splits df into X (features) and y (target)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
