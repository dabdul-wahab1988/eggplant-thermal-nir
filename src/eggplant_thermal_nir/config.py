"""Configuration objects for the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple


DEFAULT_ARCHIVES = (
    "20220630_EGGPLANT_ODILIA",
    "20220706_eggplant_Odilia2",
    "20220707_EGGPLANT_ODILIA3",
)


@dataclass
class AnalysisConfig:
    """Runtime configuration for the manuscript analysis package."""

    project_root: Path = field(default_factory=lambda: Path(".").resolve())
    workbook_name: str = "RESULT FOR PROJECT.xlsx"
    spectral_archive_names: Tuple[str, ...] = DEFAULT_ARCHIVES
    output_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    random_state: int = 42
    n_splits: int = 3
    pca_components: int = 3
    primary_preprocess: str = "sg1_snv"
    classification_preprocesses: Tuple[str, ...] = ("raw", "snv", "sg1_snv")
    classification_models: Tuple[str, ...] = ("plsda", "svm", "rf")
    classification_targets: Tuple[str, ...] = ("species", "process", "species_process")

    @property
    def workbook_path(self) -> Path:
        return self.project_root / self.workbook_name

    @property
    def spectral_dirs(self) -> Tuple[Path, ...]:
        return tuple(self.project_root / name for name in self.spectral_archive_names)

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir or (self.project_root / "manuscript" / "artifacts")

    @property
    def resolved_logs_dir(self) -> Path:
        return self.logs_dir or (self.project_root / "logs")

    @property
    def resolved_cache_dir(self) -> Path:
        return self.resolved_output_dir.parent / "cache"

    @property
    def resolved_cache_path(self) -> Path:
        return self.resolved_cache_dir / "analysis_results.pkl"

    def ensure_directories(self) -> None:
        self.resolved_output_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_logs_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    def replace_targets(self, targets: Sequence[str]) -> "AnalysisConfig":
        return AnalysisConfig(
            project_root=self.project_root,
            workbook_name=self.workbook_name,
            spectral_archive_names=self.spectral_archive_names,
            output_dir=self.output_dir,
            logs_dir=self.logs_dir,
            random_state=self.random_state,
            n_splits=self.n_splits,
            pca_components=self.pca_components,
            primary_preprocess=self.primary_preprocess,
            classification_preprocesses=self.classification_preprocesses,
            classification_models=self.classification_models,
            classification_targets=tuple(targets),
        )
