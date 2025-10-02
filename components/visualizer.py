import plotly.express as px
import pandas as pd


def plot_histogram(df: pd.DataFrame, column: str):
    """Histogram of a numeric or categorical column."""
    return px.histogram(df, x=column, nbins=30)


def plot_boxplot(df: pd.DataFrame, column: str):
    """Box‑plot for a numeric column."""
    return px.box(df, y=column)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Scatter plot.
    - If the same column is chosen for both axes, no trend line is added
      (Narwhals cannot handle duplicate column names in a single selection).
    - If both columns are numeric, add an OLS trend line; otherwise omit it.
    """
    if x_col == y_col:
        return px.scatter(df, x=x_col, y=y_col)

    if (
        pd.api.types.is_numeric_dtype(df[x_col])
        and pd.api.types.is_numeric_dtype(df[y_col])
    ):
        return px.scatter(df, x=x_col, y=y_col, trendline="ols")
    else:
        return px.scatter(df, x=x_col, y=y_col)


def plot_heatmap(df: pd.DataFrame):
    """Correlation heat‑map of numeric columns."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        # Not enough numeric columns for correlation
        return px.imshow([[0]], labels={"x": "", "y": ""})
    corr = numeric_df.corr()
    return px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
