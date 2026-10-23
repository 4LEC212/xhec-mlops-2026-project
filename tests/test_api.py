import pytest
from fastapi.testclient import TestClient

from src.modelling.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.web_service.lib.models import InputData
from src.web_service.main import app

VALID_ABALONE = {
    "sex": "M",
    "length": 0.455,
    "diameter": 0.365,
    "height": 0.095,
    "whole_weight": 0.514,
    "shucked_weight": 0.2245,
    "viscera_weight": 0.101,
    "shell_weight": 0.15,
}


@pytest.fixture(scope="module")
def client():
    """Test client of the API, with the model loaded at startup."""
    with TestClient(app) as test_client:
        yield test_client


def test_home(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"health_check": "App up and running!"}


def test_predict_returns_rings_and_age(client):
    response = client.post("/predict", json=VALID_ABALONE)

    assert response.status_code == 201
    prediction = response.json()
    assert prediction["rings"] > 0
    assert prediction["age"] == pytest.approx(prediction["rings"] + 1.5)


@pytest.mark.parametrize(
    ("field", "invalid_value"), [("sex", "X"), ("length", -0.1), ("height", 0)]
)
def test_predict_rejects_invalid_measurements(client, field, invalid_value):
    response = client.post("/predict", json={**VALID_ABALONE, field: invalid_value})

    assert response.status_code == 422


def test_input_fields_match_the_training_features():
    assert set(InputData.model_fields) == set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES)
