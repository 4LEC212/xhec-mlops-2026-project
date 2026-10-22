"""Configuration of the training flow: paths, columns and model hyperparameters."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "abalone.csv"
MODEL_PATH = PROJECT_ROOT / "src" / "web_service" / "local_objects" / "model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns of the Kaggle CSV file -> snake_case names used in the code and by the API
COLUMNS_MAPPING = {
    "Sex": "sex",
    "Length": "length",
    "Diameter": "diameter",
    "Height": "height",
    "Whole weight": "whole_weight",
    "Shucked weight": "shucked_weight",
    "Viscera weight": "viscera_weight",
    "Shell weight": "shell_weight",
    "Rings": "rings",
}
CATEGORICAL_FEATURES = ["sex"]
NUMERICAL_FEATURES = [
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
TARGET = "rings"

# Heights of 0 or above this value are measurement errors (see notebooks/eda.ipynb)
MAX_HEIGHT = 0.4

# Gradient boosting hyperparameters selected in notebooks/modelling.ipynb
MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
}
