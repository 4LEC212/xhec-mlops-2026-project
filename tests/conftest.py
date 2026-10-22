from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_abalone_df() -> pd.DataFrame:
    """Synthetic abalones with the columns of the Kaggle CSV file."""
    rng = np.random.default_rng(seed=0)
    n_abalones = 100
    length = rng.uniform(0.1, 0.8, n_abalones)
    whole_weight = 2 * length**3 + rng.uniform(0, 0.05, n_abalones)
    return pd.DataFrame(
        {
            "Sex": rng.choice(["M", "F", "I"], n_abalones),
            "Length": length,
            "Diameter": 0.8 * length,
            "Height": 0.25 * length,
            "Whole weight": whole_weight,
            "Shucked weight": 0.45 * whole_weight,
            "Viscera weight": 0.2 * whole_weight,
            "Shell weight": 0.3 * whole_weight,
            "Rings": np.round(5 + 15 * length + rng.normal(0, 1, n_abalones)).astype(int),
        }
    )


@pytest.fixture
def raw_abalone_csv(tmp_path: Path, raw_abalone_df: pd.DataFrame) -> Path:
    """Path to a CSV file containing the synthetic abalones."""
    path = tmp_path / "abalone.csv"
    raw_abalone_df.to_csv(path, index=False)
    return path
