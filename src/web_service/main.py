"""FastAPI application predicting the age of an abalone from its physical measurements."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from src.web_service.lib.inference import run_inference
from src.web_service.lib.models import InputData, PredictionOut
from src.web_service.utils import load_object

MODEL_PATH = Path(__file__).parent / "local_objects" / "model.pkl"
# The age of an abalone in years is its number of rings + 1.5
AGE_OFFSET = 1.5


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once, when the application starts."""
    app.state.model = load_object(MODEL_PATH)
    yield


app = FastAPI(
    title="Abalone age prediction",
    description=(
        "Predict the age of an abalone from its physical measurements, "
        "instead of counting the rings of its shell under a microscope."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home() -> dict:
    """Check that the API is up and running."""
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: InputData) -> PredictionOut:
    """Predict the number of rings and the age of an abalone."""
    rings = run_inference(payload, app.state.model)
    return PredictionOut(rings=rings, age=rings + AGE_OFFSET)
