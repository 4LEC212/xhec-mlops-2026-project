"""Inference logic of the web service."""

import pandas as pd
from sklearn.pipeline import Pipeline

from src.web_service.lib.models import InputData


def run_inference(input_data: InputData, model: Pipeline) -> float:
    """Predict the number of rings of an abalone.

    Args:
        input_data: Physical measurements of the abalone.
        model: The fitted pipeline (encoder and model) saved by the training flow.

    Returns:
        The predicted number of rings.
    """
    features = pd.DataFrame([input_data.model_dump()])
    return float(model.predict(features)[0])
