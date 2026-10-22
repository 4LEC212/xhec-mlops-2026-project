"""Utility functions of the web service."""

import pickle
from pathlib import Path
from typing import Any


def load_object(path: Path) -> Any:
    """Load a pickled object from disk.

    Args:
        path: Path to the pickle file.

    Returns:
        The unpickled object.
    """
    with path.open("rb") as file:
        return pickle.load(file)
