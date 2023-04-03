"""microcirculation - Python utilities for analysis of microcirculation data."""

from pathlib import Path

__author__ = "Matthias Koenig"
__version__ = "0.0.1"

data_dir = Path(__file__).parent.parent.parent / "data"
resources_dir = Path(__file__).parent / "resources"
results_dir = Path(__file__).parent.parent.parent / "results"
