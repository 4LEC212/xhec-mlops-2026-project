"""Utility tasks of the training flow."""

import pickle
from pathlib import Path
from typing import Any

from prefect import task


@task(name="Pickle object")
def pickle_object(obj: Any, path: Path) -> None:
    """Save an object to disk with pickle, creating the parent folders if needed.

    Args:
        obj: The object to save, e.g. the fitted pipeline.
        path: Where to save the pickle file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)
