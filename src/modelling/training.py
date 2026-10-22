"""Model training task of the training flow."""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.modelling.config import MODEL_PARAMS, RANDOM_STATE
from src.modelling.preprocessing import build_preprocessor


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the preprocessing and the gradient boosting model on the training set.

    The encoder and the model are fitted and saved together in one pipeline, so the API applies
    exactly the same preprocessing as during training.

    Args:
        X_train: Features of the training set.
        y_train: Number of rings of the training set.

    Returns:
        The fitted pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", GradientBoostingRegressor(random_state=RANDOM_STATE, **MODEL_PARAMS)),
        ]
    )
    return pipeline.fit(X_train, y_train)
