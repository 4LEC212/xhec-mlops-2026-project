PATH_DATA ?= data/abalone.csv

.PHONY: install lint test mlflow-ui train prefect-server deploy api docker-build docker-run

install:
	uv sync
	uv run pre-commit install

lint:
	uv run pre-commit run --all-files

test:
	uv run pytest

mlflow-ui:
	cd notebooks && uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

train:
	uv run python -m src.modelling.main $(PATH_DATA)

prefect-server:
	uv run prefect server start

deploy:
	PREFECT_API_URL=http://127.0.0.1:4200/api uv run python -m src.modelling.deploy

api:
	uv run uvicorn src.web_service.main:app --port 8001

docker-build:
	docker build -t abalone-api -f Dockerfile.app .

docker-run:
	docker run --rm -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api
