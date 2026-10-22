<div align="center">

# MLOps Project: Abalone Age Prediction

[![CI](https://github.com/4LEC212/xhec-mlops-2026-project/actions/workflows/ci.yaml/badge.svg)](https://github.com/4LEC212/xhec-mlops-2026-project/actions/workflows/ci.yaml)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
</div>

## 🎯 Project Overview

Determining the age of an abalone (a sea snail) traditionally requires cutting its shell, staining it and counting
its rings under a microscope. This project predicts the age **from physical measurements that are quick to take**,
and turns the model into a production-ready system:

- 📓 **Exploration and experiments**: EDA and model comparison notebooks, experiments tracked with MLflow
- 🤖 **Automated training pipeline**: Prefect flow (load → clean → split → train → evaluate → save) and a Prefect
  deployment retraining the model every week
- 🌐 **Prediction API**: FastAPI app with Pydantic input validation, containerized with Docker

## 👥 Team

| Name | GitHub |
|---|---|
| Alec Seugnet | [@4LEC212](https://github.com/4LEC212) |
| Simone Capizzi | [@FRATM2002](https://github.com/FRATM2002) |
| Korouhanba Khuman Laikhuram | [@laikhuramkorouhanba](https://github.com/laikhuramkorouhanba) |
| Tom Weinberg | [@tomwbrg](https://github.com/tomwbrg) |

## 📊 Dataset

[Kaggle - Abalone dataset](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) (originally from the UCI
Machine Learning Repository): 4,177 abalones described by 8 features. The continuous measurements were divided by
200 by the dataset authors (lengths in mm, weights in grams).

| Feature | Description |
|---|---|
| `Sex` | M (male), F (female) or I (infant) |
| `Length` | Longest shell measurement |
| `Diameter` | Shell diameter, perpendicular to the length |
| `Height` | Height, with the meat in the shell |
| `Whole weight` | Weight of the whole abalone |
| `Shucked weight` | Weight of the meat |
| `Viscera weight` | Gut weight, after bleeding |
| `Shell weight` | Weight of the shell, after drying |
| **`Rings`** | **Target**: number of rings. The age in years is `Rings + 1.5` |

## 🚀 Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.11 for you if needed)

### Installation

```bash
# Create the virtual environment (.venv) and install the dependencies
uv sync
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install the pre-commit hooks (ruff linting and formatting, file checks, lockfile check)
uv run pre-commit install
```

Dependencies are pinned in `pyproject.toml` and locked in `uv.lock`. Add a new one with
`uv add <package>==<version>` (`--dev` for development tools, `--group app` for the API).

### Get the data

Download the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) and put
`abalone.csv` in the `data/` folder, or use the Kaggle CLI (requires a
[Kaggle API token](https://www.kaggle.com/docs/api)):

```bash
uvx kaggle datasets download -d rodolfomendes/abalone-dataset -p data --unzip
```

## 📓 1. Exploration and Experiments

The notebooks can be run with VS Code / Jupyter using the `.venv` kernel, or with
`uv run --with jupyter jupyter lab`.

- [`notebooks/eda.ipynb`](notebooks/eda.ipynb): data quality, target distribution, effect of the sex, correlations
  and anomalies (4 impossible heights removed).
- [`notebooks/modelling.ipynb`](notebooks/modelling.ipynb): comparison of 6 models (5-fold cross-validation and
  test set), every run tracked with MLflow. To browse the experiments:

  ```bash
  cd notebooks
  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db  # then open http://127.0.0.1:5000
  ```

## 📈 Model Performance

Test set (20% of the data, 835 abalones):

| Model | CV RMSE | Test RMSE | Test MAE | Test R² | Size |
|---|---|---|---|---|---|
| Mean baseline | 3.16 | 3.46 | 2.54 | 0.00 | 2 KB |
| Linear regression | 2.18 | 2.23 | 1.58 | 0.59 | 2 KB |
| Random forest | 2.13 | 2.20 | 1.52 | 0.60 | 14 MB |
| **Gradient boosting (selected)** | 2.17 | **2.19** | 1.52 | **0.60** | 0.4 MB |

The gradient boosting model was selected: its accuracy matches the random forest's (the difference is within the
cross-validation noise), and it is about 35 times lighter to version and ship in the Docker image. The predictions
are off by about 1.5 years on average; old abalones are underestimated.

## ✅ Code Quality

```bash
uv run pre-commit run --all-files      # ruff linting and formatting, file checks, lockfile check
```

The CI ([`.github/workflows/ci.yaml`](.github/workflows/ci.yaml)) runs the pre-commit hooks on every push and pull request.
