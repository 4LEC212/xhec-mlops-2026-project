"""Prediction and evaluation tasks of the training flow."""

import numpy as np
import pandas as pd
from prefect import task
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline


@task(name="Predict")
def predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predict the number of rings of abalones.

    Args:
        model: The fitted pipeline.
        X: Features of the abalones.

    Returns:
        The predicted number of rings.
    """
    return model.predict(X)


@task(name="Evaluate predictions")
def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the regression metrics of the predictions.

    Args:
        y_true: Actual number of rings.
        y_pred: Predicted number of rings.

    Returns:
        The RMSE, MAE and R² of the predictions.
    """
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
