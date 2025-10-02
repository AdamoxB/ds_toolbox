import pandas as pd


def impute_missing(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """Impute missing values per column."""
    df_imputed = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype.kind in ("i", "f"):
                val = (
                    df[col].median() if method == "median" else
                    df[col].mean()   if method == "mean"   else
                    df[col].mode()[0]
                )
            else:
                val = df[col].mode()[0]
            df_imputed[col].fillna(val, inplace=True)
    return df_imputed


def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any numeric column is outside 1.5 × IQR."""
    df_clean = df.copy()
    for col in df.select_dtypes(include=["number"]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
        df_clean = df_clean[mask]
    return df_clean.reset_index(drop=True)
