import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_regression(X: pd.DataFrame, y: pd.Series):
    """Train linear regression; return model, predictions, test labels and metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds) ** 0.5,
        "R2": r2_score(y_test, preds),
    }
    return lr, preds, y_test, metrics
