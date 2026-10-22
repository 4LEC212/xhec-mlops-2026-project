"""Training flow: read the data, preprocess it, train a model and save it for the web service."""

import argparse
from pathlib import Path

from prefect import flow, get_run_logger
from prefect.artifacts import create_markdown_artifact

from src.modelling.config import DATA_PATH, MODEL_PATH
from src.modelling.predicting import evaluate_predictions, predict
from src.modelling.preprocessing import clean_data, load_data, split_data
from src.modelling.training import train_model
from src.modelling.utils import pickle_object


@flow(name="train-abalone-model")
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
    logger = get_run_logger()

    # Read and preprocess data
    df = clean_data(load_data(trainset_path))
    X_train, X_test, y_train, y_test = split_data(df)

    # Train and evaluate the model (the encoder is fitted inside the pipeline)
    model = train_model(X_train, y_train)
    metrics = evaluate_predictions(y_test, predict(model, X_test))
    logger.info("Test metrics: %s", {name: round(value, 3) for name, value in metrics.items()})
    metrics_rows = "\n".join(f"| {name} | {value:.3f} |" for name, value in metrics.items())
    create_markdown_artifact(
        key="abalone-model-metrics",
        markdown=f"| Metric | Value |\n|---|---|\n{metrics_rows}",
        description="Metrics of the abalone model on the test set",
    )

    # Pickle the model in the `src/web_service/local_objects` folder
    pickle_object(model, model_path)
    logger.info("Model saved to %s", model_path)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=Path, help="Path to the training set")
    args = parser.parse_args()
    train_model_workflow(args.trainset_path)
