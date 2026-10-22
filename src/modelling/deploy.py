"""Prefect deployment that retrains the abalone model every week."""

from src.modelling.main import train_model_workflow

if __name__ == "__main__":
    train_model_workflow.serve(
        name="abalone-model-retraining",
        cron="0 2 * * 1",  # every Monday at 02:00
        tags=["abalone", "retraining"],
        description="Weekly retraining of the abalone age model on the latest data.",
    )
