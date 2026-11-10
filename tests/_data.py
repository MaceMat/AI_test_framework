"""Dataset loading helper for the test modules."""

from pathlib import Path

from aitest import load_dataset

DATASETS = Path(__file__).resolve().parent.parent / "datasets"


def dataset(name: str):
    return load_dataset(DATASETS / f"{name}.yaml")
