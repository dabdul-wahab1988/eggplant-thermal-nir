"""Command-line interface for the analysis package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

from eggplant_thermal_nir.config import AnalysisConfig
from eggplant_thermal_nir.pipeline import (
    generate_artifacts,
    load_analysis_results,
    render_main_figures,
    run_analysis,
    save_analysis_results,
)


def run_interactive() -> AnalysisConfig:
    """Prompt for basic runtime options when running in an interactive terminal."""

    if not sys.stdin.isatty():
        return AnalysisConfig()

    project_root_text = input("Project root [.] : ").strip() or "."
    output_dir_text = input("Artifact output directory [manuscript/artifacts] : ").strip()
    targets_text = input("Classification targets [species,process,species_process] : ").strip()

    config = AnalysisConfig(project_root=Path(project_root_text).resolve())
    if output_dir_text:
        config.output_dir = Path(output_dir_text).resolve()
    if targets_text:
        targets = tuple(item.strip() for item in targets_text.split(",") if item.strip())
        config = config.replace_targets(targets)
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eggplant-analysis-pipeline")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--targets", default="species,process,species_process")
    parser.add_argument("--cache-file", default=None)

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("run")
    subparsers.add_parser("summaries")
    subparsers.add_parser("classify")
    subparsers.add_parser("artifacts")
    figures_parser = subparsers.add_parser("figures")
    figures_parser.add_argument(
        "--figure-labels",
        default="Figure 1,Figure 2,Figure 3,Figure 4,Figure 5",
    )
    figures_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute analysis results instead of loading the cached bundle.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    config = AnalysisConfig(project_root=Path(args.project_root).resolve())
    if args.output_dir:
        config.output_dir = Path(args.output_dir).resolve()
    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    config = config.replace_targets(targets)
    return config


def _cache_path_from_args(args: argparse.Namespace, config: AnalysisConfig) -> Path:
    if args.cache_file:
        return Path(args.cache_file).resolve()
    return config.resolved_cache_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""

    parser = _parser()
    args = parser.parse_args(argv)

    if args.command is None and argv is None:
        config = run_interactive()
    else:
        config = _config_from_args(args)

    if args.command == "summaries":
        config = config.replace_targets(tuple())
    elif args.command == "classify":
        pass
    elif args.command == "figures":
        pass
    elif args.command in ("run", "artifacts", None):
        pass

    cache_path = _cache_path_from_args(args, config)
    if args.command == "figures":
        if cache_path.exists() and not args.refresh_cache:
            results = load_analysis_results(cache_path)
        else:
            results = run_analysis(config)
            save_analysis_results(results, cache_path)
        requested_figures = tuple(
            item.strip() for item in args.figure_labels.split(",") if item.strip()
        )
        artifacts = render_main_figures(results, config.resolved_output_dir, requested_figures)
        artifacts["analysis_cache"] = str(cache_path)
    else:
        results = run_analysis(config)
        save_analysis_results(results, cache_path)
        artifacts = generate_artifacts(results, config.resolved_output_dir)
        artifacts["analysis_cache"] = str(cache_path)
    for label, path in artifacts.items():
        print("{0}: {1}".format(label, path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
