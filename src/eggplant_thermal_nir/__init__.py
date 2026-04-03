"""Eggplant thermal NIR analysis package."""

from eggplant_thermal_nir._version import __version__
from eggplant_thermal_nir.config import AnalysisConfig
from eggplant_thermal_nir.pipeline import generate_artifacts, run_analysis

__all__ = ["AnalysisConfig", "__version__", "generate_artifacts", "run_analysis"]

