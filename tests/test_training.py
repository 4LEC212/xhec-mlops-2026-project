import pickle

import numpy as np
import pandas as pd
import pytest

from src.modelling.predicting import evaluate_predictions, predict
from src.modelling.preprocessing import load_data, split_data
from src.modelling.training import train_model
from src.modelling.utils import pickle_object


def test_train_model_returns_a_fitted_pipeline(raw_abalone_csv):
    X_train, X_test, y_train, _ = split_data.fn(load_data.fn(raw_abalone_csv))

    model = train_model.fn(X_train, y_train)
    predictions = predict.fn(model, X_test)

    assert predictions.shape == (len(X_test),)
    assert np.isfinite(predictions).all()


def test_evaluate_predictions():
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])

    metrics = evaluate_predictions.fn(y_true, y_pred)

    assert metrics["rmse"] == pytest.approx((4 / 3) ** 0.5)
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["r2"] == pytest.approx(-1)


def test_pickle_object_creates_missing_folders(tmp_path):
    path = tmp_path / "local_objects" / "object.pkl"

    pickle_object.fn({"answer": 42}, path)

    with path.open("rb") as file:
        assert pickle.load(file) == {"answer": 42}
