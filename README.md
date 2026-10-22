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

## 📁 Repository Structure

```
├── .github/workflows/ci.yaml     # CI: pre-commit hooks (ruff, ...) and tests
├── bin/run_services.sh           # Starts the Prefect server and the API (Docker entrypoint)
├── data/                         # Put abalone.csv here (not versioned)
├── notebooks/
│   ├── eda.ipynb                 # Exploratory data analysis
│   └── modelling.ipynb           # Model comparison, tracked with MLflow
├── src/
│   ├── modelling/                # Training pipeline (Prefect tasks and flow)
│   │   ├── config.py             # Paths, columns and model hyperparameters
│   │   ├── preprocessing.py      # Load, clean and split the data, preprocessing step
│   │   ├── training.py           # Train the model pipeline
│   │   ├── predicting.py         # Predict and evaluate
│   │   ├── utils.py              # Pickle the model
│   │   ├── main.py               # Training flow
│   │   └── deploy.py             # Weekly retraining deployment
│   └── web_service/              # Prediction API
│       ├── main.py               # FastAPI app
│       ├── lib/models.py         # Pydantic input and output models
│       ├── lib/inference.py      # Inference logic
│       ├── utils.py              # Load the model
│       └── local_objects/        # Trained model (model.pkl)
├── tests/                        # Unit tests of the pipeline, the flow and the API
├── Dockerfile.app                # Docker image of the API and the Prefect server
├── Makefile                      # Shortcuts for the main commands
└── pyproject.toml / uv.lock      # Dependencies, managed with uv
```

## 🚀 Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.11 for you if needed)
- [Docker](https://docs.docker.com/get-docker/), to run the API in a container

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

### Shortcuts

The [`Makefile`](Makefile) wraps the main commands of this README:

| Command | What it does |
|---|---|
| `make install` | Install the environment and the pre-commit hooks |
| `make lint` / `make test` | Run the pre-commit hooks / the tests |
| `make mlflow-ui` | Open the MLflow UI of the notebooks experiments |
| `make train` | Train the model (Prefect flow) |
| `make prefect-server` / `make deploy` | Start the Prefect server / serve the weekly retraining deployment |
| `make api` | Run the API locally |
| `make docker-build` / `make docker-run` | Build / run the Docker image of the API |

## 📓 1. Exploration and Experiments

The notebooks can be run with VS Code / Jupyter using the `.venv` kernel, or with
`uv run --with jupyter jupyter lab`.

- [`notebooks/eda.ipynb`](notebooks/eda.ipynb): data quality, target distribution, effect of the sex, correlations
  and anomalies (4 impossible heights removed).
- [`notebooks/modelling.ipynb`](notebooks/modelling.ipynb): comparison of 6 models (5-fold cross-validation and
  test set), every run tracked with MLflow. To browse the experiments:

  ```bash
  cd notebooks
  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001  # then open http://127.0.0.1:5001
  ```

## 🤖 2. Training Pipeline (Prefect)

The training pipeline is a Prefect flow: it loads and cleans the data, trains the model, logs its test metrics and
saves the fitted pipeline (one-hot encoder + model) to `src/web_service/local_objects/model.pkl`. Training is
reproducible (fixed random seeds).

### Train the model once

```bash
uv run python -m src.modelling.main data/abalone.csv
```

### Visualize the flow runs in the Prefect UI

```bash
# Terminal 1: start the Prefect server, then open the UI at http://127.0.0.1:4200
uv run prefect server start

# Terminal 2: send the flow runs to this server, then run the training flow
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
uv run python -m src.modelling.main data/abalone.csv
```

Each run shows its tasks, logs and a `abalone-model-metrics` artifact with the test metrics.

### Retrain the model regularly (Prefect deployment)

```bash
# Terminal 2 (with the Prefect server still running): serve the deployment
uv run python -m src.modelling.deploy
```

This creates the `abalone-model-retraining` deployment, scheduled **every Monday at 02:00**, and keeps a process
polling for its scheduled runs. To trigger a run right away (or use "Quick run" in the UI):

```bash
uv run prefect deployment run 'train-abalone-model/abalone-model-retraining'
```

> Run these commands from the root of the repository.

## 🌐 3. Prediction API

### Run the API locally

```bash
uv run uvicorn src.web_service.main:app --port 8001
```

The interactive documentation is available at http://localhost:8001/docs.

### Run the API with Docker

The container runs the API and a Prefect server (see [`bin/run_services.sh`](bin/run_services.sh)). Stop any local
Prefect server first, as port 4200 is used by the container.

```bash
docker build -t abalone-api -f Dockerfile.app .
docker run -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api
```

- API documentation: http://localhost:8000/docs
- Prefect UI: http://localhost:4200

### Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sex": "F", "length": 0.53, "diameter": 0.42, "height": 0.135, "whole_weight": 0.677,
       "shucked_weight": 0.2565, "viscera_weight": 0.1415, "shell_weight": 0.21}'
```

```json
{"rings": 10.771656369915297, "age": 12.271656369915297}
```

Invalid inputs (unknown sex, missing field, measurement ≤ 0) are rejected with a `422` error describing the
problem.

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

## ✅ Code Quality and Tests

```bash
uv run pytest                          # unit tests (pipeline, Prefect flow, API) on synthetic data
uv run pre-commit run --all-files      # ruff linting and formatting, file checks, lockfile check
```

The CI ([`.github/workflows/ci.yaml`](.github/workflows/ci.yaml)) runs both on every push and pull request.

## 🎯 Deliverables Checklist

- [x] Prefect workflows for model training: [`src/modelling/main.py`](src/modelling/main.py)
- [x] Separate modules for training and inference: [`src/modelling/`](src/modelling/),
  [`src/web_service/lib/inference.py`](src/web_service/lib/inference.py)
- [x] Reproducible model and encoder generation: one scikit-learn pipeline with fixed seeds
- [x] Prefect deployment for regular retraining: [`src/modelling/deploy.py`](src/modelling/deploy.py)
- [x] REST API for predictions: [`src/web_service/main.py`](src/web_service/main.py)
- [x] Pydantic input validation: [`src/web_service/lib/models.py`](src/web_service/lib/models.py)
- [x] Docker containerization: [`Dockerfile.app`](Dockerfile.app)
- [x] README with team info and setup and run instructions
