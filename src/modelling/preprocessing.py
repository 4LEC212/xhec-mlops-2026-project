"""Data loading and preprocessing tasks of the training flow."""

from pathlib import Path

import pandas as pd
from prefect import task
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.modelling.config import (
    CATEGORICAL_FEATURES,
    COLUMNS_MAPPING,
    MAX_HEIGHT,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


@task(name="Load data")
def load_data(path: Path) -> pd.DataFrame:
    """Read the abalone CSV file and rename its columns to snake_case.

    Args:
        path: Path to the abalone CSV file downloaded from Kaggle.

    Returns:
        The abalone dataset with snake_case column names.
    """
    return pd.read_csv(path).rename(columns=COLUMNS_MAPPING)


@task(name="Clean data")
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the abalones with an impossible height (0 or above MAX_HEIGHT).

    Args:
        df: The abalone dataset.

    Returns:
        The dataset without the measurement errors.
    """
    has_valid_height = (df["height"] > 0) & (df["height"] <= MAX_HEIGHT)
    return df[has_valid_height].reset_index(drop=True)


@task(name="Split data")
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into train and test sets of features and target.

    Args:
        df: The cleaned abalone dataset.

    Returns:
        The train features, test features, train target and test target.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES],
        df[TARGET],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing step: one-hot encoding of the sex and scaling of the measurements.

    Returns:
        The preprocessor, not fitted yet.
    """
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numerical", StandardScaler(), NUMERICAL_FEATURES),
        ]
    )
