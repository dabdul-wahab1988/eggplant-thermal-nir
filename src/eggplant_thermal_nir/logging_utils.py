"""Structured logging helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunLogger:
    """Small JSONL logger used by the analysis pipeline."""

    log_dir: Path
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = self.log_dir / f"run_{self.run_id}.jsonl"
        self.statistics_log_path = self.log_dir / f"statistics_{self.run_id}.jsonl"

    def log_event(
        self,
        stage: str,
        status: str,
        message: str,
        duration_ms: Optional[float] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        payload = {
            "run_id": self.run_id,
            "timestamp": _now_iso(),
            "stage": stage,
            "status": status,
            "message": message,
            "duration_ms": duration_ms,
        }
        if extra:
            payload.update(extra)
        with self.run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def log_statistic(
        self,
        analysis_name: str,
        metric_name: str,
        metric_value: float,
        grouping: str,
        sample_size: int,
        notes: str = "",
    ) -> None:
        payload = {
            "analysis_name": analysis_name,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "grouping": grouping,
            "sample_size": sample_size,
            "notes": notes,
        }
        with self.statistics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")


class StageTimer:
    """Context manager for timing pipeline stages."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000.0, 3)

