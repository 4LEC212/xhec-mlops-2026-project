import pickle

import pytest
from prefect.testing.utilities import prefect_test_harness

from src.modelling.main import train_model_workflow


@pytest.fixture(autouse=True, scope="module")
def prefect_test_server():
    """Run the flows of this module against a temporary Prefect server and database."""
    with prefect_test_harness():
        yield


def test_train_model_workflow_saves_the_model(raw_abalone_csv, tmp_path):
    model_path = tmp_path / "model.pkl"

    metrics = train_model_workflow(raw_abalone_csv, model_path)

    assert set(metrics) == {"rmse", "mae", "r2"}
    with model_path.open("rb") as file:
        model = pickle.load(file)
    assert hasattr(model, "predict")
