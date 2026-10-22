import pandas as pd
import pytest

from src.modelling.config import (
    CATEGORICAL_FEATURES,
    COLUMNS_MAPPING,
    NUMERICAL_FEATURES,
    TARGET,
    TEST_SIZE,
)
from src.modelling.preprocessing import build_preprocessor, clean_data, load_data, split_data


def test_load_data_renames_columns_to_snake_case(raw_abalone_csv):
    df = load_data.fn(raw_abalone_csv)

    assert list(df.columns) == list(COLUMNS_MAPPING.values())


def test_clean_data_removes_impossible_heights():
    df = pd.DataFrame({"height": [0.0, 0.1, 0.25, 1.13]})

    cleaned_df = clean_data.fn(df)

    assert cleaned_df["height"].tolist() == [0.1, 0.25]


def test_split_data_separates_features_and_target(raw_abalone_csv):
    df = load_data.fn(raw_abalone_csv)

    X_train, X_test, y_train, y_test = split_data.fn(df)

    assert list(X_train.columns) == CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    assert TARGET not in X_test.columns
    assert len(X_test) == pytest.approx(len(df) * TEST_SIZE, abs=1)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_build_preprocessor_one_hot_encodes_sex(raw_abalone_csv):
    df = load_data.fn(raw_abalone_csv)

    features = build_preprocessor().fit_transform(df)

    n_sexes = df["sex"].nunique()
    assert features.shape == (len(df), n_sexes + len(NUMERICAL_FEATURES))
