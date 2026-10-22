"""Training pipeline: read the data, preprocess it, train a model and save it for the API."""

import argparse
import logging
from pathlib import Path

from src.modelling.config import DATA_PATH, MODEL_PATH
from src.modelling.predicting import evaluate_predictions, predict
from src.modelling.preprocessing import clean_data, load_data, split_data
from src.modelling.training import train_model
from src.modelling.utils import pickle_object

logger = logging.getLogger(__name__)


def train_model_workflow(
    trainset_path: Path = DATA_PATH, model_path: Path = MODEL_PATH
) -> dict[str, float]:
    """Train a model using the data at the given path and save the model (pickle).

    Args:
        trainset_path: Path to the abalone CSV file.
        model_path: Where to save the fitted pipeline (encoder and model).

    Returns:
        The metrics of the model on the test set.
    """
    # Read and preprocess data
    df = clean_data(load_data(trainset_path))
    X_train, X_test, y_train, y_test = split_data(df)

    # Train and evaluate the model (the encoder is fitted inside the pipeline)
    model = train_model(X_train, y_train)
    metrics = evaluate_predictions(y_test, predict(model, X_test))
    logger.info("Test metrics: %s", {name: round(value, 3) for name, value in metrics.items()})

    # Pickle the model in the `src/web_service/local_objects` folder
    pickle_object(model, model_path)
    logger.info("Model saved to %s", model_path)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=Path, help="Path to the training set")
    args = parser.parse_args()
    train_model_workflow(args.trainset_path)
