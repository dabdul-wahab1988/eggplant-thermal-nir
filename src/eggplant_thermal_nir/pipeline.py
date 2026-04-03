"""End-to-end orchestration and artifact generation."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import pandas as pd

from eggplant_thermal_nir.chemometrics import (
    build_supported_preprocessing_table,
    compute_plsda_vip,
    compute_grouped_roc_curves,
    evaluate_leakage_baseline,
    evaluate_grouped_classifiers,
    run_pca,
    run_grouped_permutation_test,
    run_hyperparameter_grid_search,
    summarize_factorial_spectral_effects,
)
from eggplant_thermal_nir.config import AnalysisConfig
from eggplant_thermal_nir.data import (
    build_sample_metadata,
    combine_nutrient_tables,
    load_nutrient_workbook,
    load_spectral_archives,
    pivot_nutrients_wide,
)
from eggplant_thermal_nir.integration import (
    aggregate_spectra_by_sample,
    compute_band_correlation_matrix,
    compute_wavelength_correlation_table,
    fit_correlation_integration,
    summarize_spectral_regions,
)
from eggplant_thermal_nir.logging_utils import RunLogger, StageTimer
from eggplant_thermal_nir.plotting import (
    plot_figure_1_framework,
    plot_figure_2_nutrient_resilience,
    plot_figure_3_spectral_structure,
    plot_figure_4_classification_rigor,
    plot_figure_5_integration_story,
    plot_supplementary_figure_s1_raw_spectra,
    plot_supplementary_figure_s2_preprocessing_comparison,
    plot_supplementary_figure_s3_outlier_detection,
    plot_supplementary_figure_s4_full_nutrient_interactions,
    plot_supplementary_figure_s5_individual_nutrient_boxplots,
    plot_supplementary_figure_s6_additional_pca,
    plot_supplementary_figure_s7_hca_dendrogram,
    plot_supplementary_figure_s8_asca_details,
    plot_supplementary_figure_s9_hyperparameter_tuning,
    plot_supplementary_figure_s10_confusion_matrices,
    plot_supplementary_figure_s11_roc_curves,
    plot_supplementary_figure_s12_loading_correlation_matrices,
)
from eggplant_thermal_nir.spectra import preprocess_spectra
from eggplant_thermal_nir.statistics import (
    compute_nutrient_summary,
    compute_retention_changes,
    compute_thermal_resilience_index,
    format_nutrient_composition_table,
    fit_factorial_models,
    summarize_best_classification_results,
    summarize_experimental_design,
    summarize_factorial_results,
    summarize_resilience_contributions,
)


def _summarize_spectral_variance(factorial_spectral: pd.DataFrame) -> pd.DataFrame:
    frame = factorial_spectral.copy()
    frame["effect"] = frame["effect"].replace(
        {
            "C(species)": "species",
            "C(process)": "process",
            "C(species):C(process)": "interaction",
            "Residual": "residual",
        }
    )
    grouped = frame.groupby("effect", as_index=False)["sum_sq"].sum()
    total = float(grouped["sum_sq"].sum()) or 1.0
    grouped["percent_variance"] = (grouped["sum_sq"] / total) * 100.0
    return grouped.sort_values("percent_variance", ascending=False).reset_index(drop=True)


SUPPLEMENTARY_PREPROCESS_METHODS = ("raw", "snv", "msc", "first_derivative", "sg1_snv", "sg2_snv")
MAIN_FIGURE_LABELS = ("Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5")


def _default_hyperparameter_grid(model_name: str) -> Dict[str, tuple]:
    if model_name == "plsda":
        return {"n_components": (2, 3, 4, 5, 6)}
    if model_name == "svm":
        return {"C": (0.5, 1.0, 2.0, 4.0), "gamma": ("scale", "auto")}
    if model_name == "rf":
        return {
            "n_estimators": (100, 300, 600),
            "max_depth": (None, 8, 16),
            "min_samples_leaf": (1, 2, 4),
        }
    raise ValueError("Unsupported model for tuning: {0}".format(model_name))


def _best_grouped_model_specs(classification_summary: pd.DataFrame) -> pd.DataFrame:
    if classification_summary.empty:
        return pd.DataFrame(columns=["target", "split_strategy", "preprocessing", "model"])
    grouped = classification_summary.loc[classification_summary["split_strategy"] == "grouped"].copy()
    if grouped.empty:
        return pd.DataFrame(columns=["target", "split_strategy", "preprocessing", "model"])
    return (
        grouped.sort_values(["target", "balanced_accuracy", "macro_f1"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .first()
        .reset_index(drop=True)
    )


def _format_tuning_configuration(model_name: str, row: pd.Series) -> str:
    if model_name == "plsda":
        return "n={0}".format(int(row["n_components"]))
    if model_name == "svm":
        return "C={0}, g={1}".format(row["C"], row["gamma"])
    if model_name == "rf":
        depth = "None" if pd.isna(row["max_depth"]) else int(row["max_depth"])
        return "trees={0}, depth={1}, leaf={2}".format(
            int(row["n_estimators"]),
            depth,
            int(row["min_samples_leaf"]),
        )
    return model_name


def _build_hyperparameter_grid_table(best_model_specs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in best_model_specs.itertuples(index=False):
        param_grid = _default_hyperparameter_grid(str(row.model))
        configuration_count = 1
        for values in param_grid.values():
            configuration_count *= len(values)
        for parameter_name, values in param_grid.items():
            rows.append(
                {
                    "target": row.target,
                    "preprocessing": row.preprocessing,
                    "model": row.model,
                    "parameter": parameter_name,
                    "candidate_values": ", ".join(str(value) for value in values),
                    "n_candidate_values": len(values),
                    "n_configurations": configuration_count,
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "model", "parameter"]).reset_index(drop=True)


def _build_cleaned_dataset(chemistry_wide: pd.DataFrame, sample_metadata: pd.DataFrame) -> pd.DataFrame:
    if chemistry_wide.empty:
        return sample_metadata.copy()
    cleaned = sample_metadata.merge(
        chemistry_wide.reset_index(),
        on="analysis_sample_id",
        how="left",
    )
    nutrient_columns = chemistry_wide.columns.tolist()
    cleaned["n_missing_nutrients"] = cleaned[nutrient_columns].isna().sum(axis=1)
    return cleaned.sort_values(["species_code", "process", "replicate"]).reset_index(drop=True)


def _summarize_asca_variance_table(factorial_spectral: pd.DataFrame) -> pd.DataFrame:
    frame = factorial_spectral.copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "component",
                "effect",
                "df",
                "sum_sq",
                "F",
                "p_value",
                "percent_within_component",
                "overall_percent_variance",
            ]
        )
    frame["effect"] = frame["effect"].replace(
        {
            "C(species)": "species",
            "C(process)": "process",
            "C(species):C(process)": "interaction",
            "Residual": "residual",
        }
    )
    component_total = frame.groupby("component")["sum_sq"].transform("sum").replace(0, 1.0)
    overall_total = float(frame["sum_sq"].sum()) or 1.0
    frame["p_value"] = frame["PR(>F)"]
    frame["percent_within_component"] = (frame["sum_sq"] / component_total) * 100.0
    frame["overall_percent_variance"] = (frame["sum_sq"] / overall_total) * 100.0
    return frame[
        [
            "component",
            "effect",
            "df",
            "sum_sq",
            "F",
            "p_value",
            "percent_within_component",
            "overall_percent_variance",
        ]
    ].sort_values(["component", "effect"]).reset_index(drop=True)


def save_analysis_results(results: Mapping[str, object], cache_path: Path) -> Path:
    """Persist analysis results for later figure-only rerenders."""

    target = Path(cache_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(results)
    serializable.pop("logger", None)
    with target.open("wb") as handle:
        pickle.dump(serializable, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def load_analysis_results(cache_path: Path) -> Dict[str, object]:
    """Load cached analysis results produced by :func:`save_analysis_results`."""

    source = Path(cache_path).resolve()
    with source.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Cached analysis payload must be a dictionary.")
    return loaded


def _build_main_figure_paths(output_root: Path) -> Dict[str, Path]:
    figures_dir = Path(output_root).resolve() / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "Figure 1": figures_dir / "figure_1_study_design_framework.png",
        "Figure 2": figures_dir / "figure_2_nutrient_responses_resilience.png",
        "Figure 3": figures_dir / "figure_3_spectral_structure_variance.png",
        "Figure 4": figures_dir / "figure_4_classification_validation.png",
        "Figure 5": figures_dir / "figure_5_nutrient_spectrum_integration.png",
    }


def _resolve_best_confusion(results: Mapping[str, object]) -> pd.DataFrame:
    classification_summary = results["classification_summary"]
    grouped_summary = (
        classification_summary.loc[classification_summary["split_strategy"] == "grouped"]
        if not classification_summary.empty
        else classification_summary
    )
    if grouped_summary.empty:
        return pd.DataFrame([[0.0]], index=["not_run"], columns=["not_run"])
    preferred = grouped_summary.loc[grouped_summary["target"] == "species_process"]
    ranking = preferred if not preferred.empty else grouped_summary
    best_key = ranking.sort_values("balanced_accuracy", ascending=False).iloc[0]["result_key"]
    return results["confusion_tables"][best_key]


def render_main_figures(
    results: Mapping[str, object],
    output_dir: Path,
    figure_labels: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Render selected main figures without rerunning the analysis workflow."""

    requested = tuple(figure_labels or MAIN_FIGURE_LABELS)
    figure_paths = _build_main_figure_paths(Path(output_dir).resolve())

    if "Figure 1" in requested:
        plot_figure_1_framework(results["sample_metadata"], results["spectra"], figure_paths["Figure 1"])
    if "Figure 2" in requested:
        plot_figure_2_nutrient_resilience(
            results["nutrient_summary"],
            results["retention"],
            results["resilience"],
            results["factorial_table"],
            results["resilience_contributions"],
            figure_paths["Figure 2"],
        )
    if "Figure 3" in requested:
        plot_figure_3_spectral_structure(
            results["spectra"],
            results["preprocessed_spectra"],
            results["spectral_meta"],
            results["pca_scores"],
            results["pca_loadings"],
            results["spectral_variance"],
            figure_paths["Figure 3"],
        )
    if "Figure 4" in requested:
        plot_figure_4_classification_rigor(
            results["classification_summary"],
            results["leakage_summary"],
            _resolve_best_confusion(results),
            results["permutation_results"],
            results["vip_scores"],
            figure_paths["Figure 4"],
        )
    if "Figure 5" in requested:
        plot_figure_5_integration_story(
            results["association"],
            results["key_wavelengths"],
            results["nutrient_loadings"],
            results["region_summary"],
            results["band_correlation"],
            figure_paths["Figure 5"],
        )

    return {label: str(figure_paths[label]) for label in requested if label in figure_paths}


def run_analysis(config: Optional[AnalysisConfig] = None) -> Dict[str, object]:
    """Run the full nutrient, NIR, and integration analysis workflow."""

    config = config or AnalysisConfig()
    config.ensure_directories()
    logger = RunLogger(config.resolved_logs_dir)

    timer = StageTimer()
    nutrient_tables = load_nutrient_workbook(config.workbook_path)
    chemistry = combine_nutrient_tables(nutrient_tables)
    spectral_meta, spectra = load_spectral_archives(config.project_root)
    sample_metadata = build_sample_metadata(nutrient_tables, spectral_meta)
    logger.log_event("load_data", "ok", "Loaded workbook and spectral archives", timer.elapsed_ms)

    timer = StageTimer()
    nutrient_summary = compute_nutrient_summary(chemistry)
    factorial_models = fit_factorial_models(chemistry)
    retention = compute_retention_changes(chemistry)
    resilience = compute_thermal_resilience_index(retention)
    logger.log_event("nutrient_statistics", "ok", "Computed nutrient summaries and factorial models", timer.elapsed_ms)

    timer = StageTimer()
    preprocessed = preprocess_spectra(spectra, config.primary_preprocess)
    pca_scores, pca_loadings = run_pca(preprocessed, n_components=config.pca_components)
    factorial_spectral = summarize_factorial_spectral_effects(spectral_meta, pca_scores)
    spectral_variance = _summarize_spectral_variance(factorial_spectral)
    logger.log_event("spectral_exploration", "ok", "Computed preprocessing, PCA, and factorial spectral summary", timer.elapsed_ms)

    classification_tables = {}
    classification_fold_tables = {}
    confusion_tables = {}
    combined_classification_frames = []
    combined_classification_fold_frames = []
    for target in config.classification_targets:
        timer = StageTimer()
        summary, confusions, fold_metrics = evaluate_grouped_classifiers(
            spectra,
            spectral_meta,
            target=target,
            random_state=config.random_state,
            n_splits=config.n_splits,
            preprocess_methods=config.classification_preprocesses,
            models=config.classification_models,
        )
        classification_tables[target] = summary
        classification_fold_tables[target] = fold_metrics
        confusion_tables.update(confusions)
        combined_classification_frames.append(summary)
        combined_classification_fold_frames.append(fold_metrics)
        logger.log_event("classification", "ok", "Completed grouped classification for {0}".format(target), timer.elapsed_ms)
        if not summary.empty:
            best_row = summary.sort_values("balanced_accuracy", ascending=False).iloc[0]
            logger.log_statistic(
                "classification",
                "best_balanced_accuracy",
                float(best_row["balanced_accuracy"]),
                grouping=target,
                sample_size=int(best_row["n_groups"]),
                notes="{0}/{1}".format(best_row["preprocessing"], best_row["model"]),
            )

    classification_summary = (
        pd.concat(combined_classification_frames, ignore_index=True)
        if combined_classification_frames
        else pd.DataFrame(columns=["target", "split_strategy", "preprocessing", "model", "accuracy", "balanced_accuracy", "macro_f1", "result_key"])
    )
    classification_fold_metrics = (
        pd.concat(combined_classification_fold_frames, ignore_index=True)
        if combined_classification_fold_frames
        else pd.DataFrame(columns=["target", "split_strategy", "preprocessing", "model", "fold", "n_test", "accuracy", "balanced_accuracy", "macro_f1"])
    )
    leakage_summary = evaluate_leakage_baseline(
        spectra,
        spectral_meta,
        classification_summary,
        random_state=config.random_state,
        n_splits=config.n_splits,
    )
    best_model_specs = _best_grouped_model_specs(classification_summary)
    preprocessing_showcase = {
        method: preprocess_spectra(spectra, method)
        for method in SUPPLEMENTARY_PREPROCESS_METHODS
    }
    preprocessing_table = build_supported_preprocessing_table()
    preprocessing_table["used_in_primary_analysis"] = preprocessing_table["method"].eq(config.primary_preprocess)
    preprocessing_table["used_in_classification"] = preprocessing_table["method"].isin(config.classification_preprocesses)
    preprocessing_table["used_in_supplementary_showcase"] = preprocessing_table["method"].isin(SUPPLEMENTARY_PREPROCESS_METHODS)
    hyperparameter_grid_table = _build_hyperparameter_grid_table(best_model_specs)

    if classification_summary.empty:
        vip_scores = pd.DataFrame(columns=["wavelength_nm", "vip_score", "target", "preprocessing"])
        permutation_results = pd.DataFrame(columns=["iteration", "kind", "balanced_accuracy"])
        hyperparameter_search_results = pd.DataFrame()
        roc_curves = pd.DataFrame()
    else:
        grouped_summary = classification_summary.loc[classification_summary["split_strategy"] == "grouped"]
        preferred = grouped_summary.loc[grouped_summary["target"] == "species_process"]
        focus_row = (
            preferred.sort_values("balanced_accuracy", ascending=False).iloc[0]
            if not preferred.empty
            else grouped_summary.sort_values("balanced_accuracy", ascending=False).iloc[0]
        )
        vip_scores = compute_plsda_vip(
            spectra=spectra,
            meta=spectral_meta,
            target=str(focus_row["target"]),
            preprocess_name=str(focus_row["preprocessing"]),
        )
        permutation_results = run_grouped_permutation_test(
            spectra=spectra,
            meta=spectral_meta,
            target=str(focus_row["target"]),
            preprocess_name=str(focus_row["preprocessing"]),
            model_name=str(focus_row["model"]),
            random_state=config.random_state,
            n_splits=config.n_splits,
        )
        tuning_frames = []
        roc_frames = []
        for row in best_model_specs.itertuples(index=False):
            param_grid = _default_hyperparameter_grid(str(row.model))
            tuning_df = run_hyperparameter_grid_search(
                spectra=spectra,
                meta=spectral_meta,
                target=str(row.target),
                preprocess_name=str(row.preprocessing),
                model_name=str(row.model),
                param_grid=param_grid,
                random_state=config.random_state,
                n_splits=config.n_splits,
            )
            if not tuning_df.empty:
                tuning_df["configuration_label"] = tuning_df.apply(
                    lambda series: _format_tuning_configuration(str(row.model), series),
                    axis=1,
                )
                tuning_frames.append(tuning_df)
            roc_df = compute_grouped_roc_curves(
                spectra=spectra,
                meta=spectral_meta,
                target=str(row.target),
                preprocess_name=str(row.preprocessing),
                model_name=str(row.model),
                random_state=config.random_state,
                n_splits=config.n_splits,
            )
            if not roc_df.empty:
                roc_frames.append(roc_df)
        hyperparameter_search_results = (
            pd.concat(tuning_frames, ignore_index=True)
            if tuning_frames
            else pd.DataFrame()
        )
        roc_curves = pd.concat(roc_frames, ignore_index=True) if roc_frames else pd.DataFrame()

    timer = StageTimer()
    chemistry_wide = pivot_nutrients_wide(chemistry)
    aggregated_spectra = aggregate_spectra_by_sample(spectra, spectral_meta)
    association, key_wavelengths, nutrient_loadings = fit_correlation_integration(chemistry_wide, aggregated_spectra)
    region_summary = summarize_spectral_regions(key_wavelengths)
    band_correlation = compute_band_correlation_matrix(chemistry_wide, aggregated_spectra)
    wavelength_statistics = compute_wavelength_correlation_table(
        chemistry_wide,
        aggregated_spectra,
        association_df=association,
        vip_scores=vip_scores,
    )
    logger.log_event("integration", "ok", "Computed aggregated spectra and nutrient-spectrum integration", timer.elapsed_ms)

    design_table = summarize_experimental_design(sample_metadata)
    nutrient_composition_table = format_nutrient_composition_table(nutrient_summary)
    factorial_table = summarize_factorial_results(factorial_models)
    resilience_contributions = summarize_resilience_contributions(retention)
    classification_table = summarize_best_classification_results(classification_summary, leakage_summary)
    cleaned_dataset = _build_cleaned_dataset(chemistry_wide, sample_metadata)
    asca_variance_table = _summarize_asca_variance_table(factorial_spectral)

    return {
        "config": config,
        "logger": logger,
        "nutrient_tables": nutrient_tables,
        "chemistry": chemistry,
        "spectral_meta": spectral_meta,
        "spectra": spectra,
        "sample_metadata": sample_metadata,
        "nutrient_summary": nutrient_summary,
        "factorial_models": factorial_models,
        "retention": retention,
        "resilience": resilience,
        "preprocessed_spectra": preprocessed,
        "pca_scores": pca_scores,
        "pca_loadings": pca_loadings,
        "factorial_spectral": factorial_spectral,
        "spectral_variance": spectral_variance,
        "classification_tables": classification_tables,
        "classification_fold_tables": classification_fold_tables,
        "classification_summary": classification_summary,
        "classification_fold_metrics": classification_fold_metrics,
        "classification_table": classification_table,
        "confusion_tables": confusion_tables,
        "leakage_summary": leakage_summary,
        "vip_scores": vip_scores,
        "permutation_results": permutation_results,
        "preprocessing_showcase": preprocessing_showcase,
        "preprocessing_table": preprocessing_table,
        "best_model_specs": best_model_specs,
        "hyperparameter_grid_table": hyperparameter_grid_table,
        "hyperparameter_search_results": hyperparameter_search_results,
        "roc_curves": roc_curves,
        "chemistry_wide": chemistry_wide,
        "aggregated_spectra": aggregated_spectra,
        "association": association,
        "key_wavelengths": key_wavelengths,
        "nutrient_loadings": nutrient_loadings,
        "region_summary": region_summary,
        "band_correlation": band_correlation,
        "wavelength_statistics": wavelength_statistics,
        "design_table": design_table,
        "nutrient_composition_table": nutrient_composition_table,
        "factorial_table": factorial_table,
        "resilience_contributions": resilience_contributions,
        "cleaned_dataset": cleaned_dataset,
        "asca_variance_table": asca_variance_table,
        "run_log_path": logger.run_log_path,
        "statistics_log_path": logger.statistics_log_path,
    }


def _write_manifest(output_root: Path) -> Path:
    manifest_path = output_root.parent / "artifact_manifest.yaml"
    try:
        relative_output = output_root.relative_to(output_root.parent.parent)
    except ValueError:
        relative_output = output_root.name
    output_prefix = str(relative_output).replace("\\", "/")
    entries = [
        ("Table 1", "Experimental design and dataset structure", "{0}/tables/table_1_experimental_design_dataset_structure.csv".format(output_prefix)),
        ("Table 2", "Nutrient composition by species and treatment", "{0}/tables/table_2_nutrient_composition.csv".format(output_prefix)),
        ("Table 3", "Factorial model results", "{0}/tables/table_3_factorial_model_results.csv".format(output_prefix)),
        ("Table 4", "Classification performance summary", "{0}/tables/table_4_classification_performance_summary.csv".format(output_prefix)),
        ("Table 5", "Key spectral regions and interpretation", "{0}/tables/table_5_key_spectral_regions.csv".format(output_prefix)),
        ("Figure 1", "Study design and integrated analytical framework", "{0}/figures/figure_1_study_design_framework.png".format(output_prefix)),
        ("Figure 2", "Nutrient responses and thermal resilience", "{0}/figures/figure_2_nutrient_responses_resilience.png".format(output_prefix)),
        ("Figure 3", "Spectral structure and variance decomposition", "{0}/figures/figure_3_spectral_structure_variance.png".format(output_prefix)),
        ("Figure 4", "Classification performance and validation rigor", "{0}/figures/figure_4_classification_validation.png".format(output_prefix)),
        ("Figure 5", "Nutrient-spectrum integration", "{0}/figures/figure_5_nutrient_spectrum_integration.png".format(output_prefix)),
        ("Supplementary Table ST1", "Raw nutrient dataset", "{0}/supplementary/tables/supplementary_table_st1_raw_nutrient_dataset.csv".format(output_prefix)),
        ("Supplementary Table ST2", "Cleaned matched dataset", "{0}/supplementary/tables/supplementary_table_st2_cleaned_matched_dataset.csv".format(output_prefix)),
        ("Supplementary Table ST3", "Preprocessing pipelines tested", "{0}/supplementary/tables/supplementary_table_st3_preprocessing_pipelines_tested.csv".format(output_prefix)),
        ("Supplementary Table ST4", "Hyperparameter grids", "{0}/supplementary/tables/supplementary_table_st4_hyperparameter_grids.csv".format(output_prefix)),
        ("Supplementary Table ST5", "Full classification metrics across folds", "{0}/supplementary/tables/supplementary_table_st5_full_classification_metrics_all_folds.csv".format(output_prefix)),
        ("Supplementary Table ST6", "ASCA-style variance table", "{0}/supplementary/tables/supplementary_table_st6_asca_variance_tables.csv".format(output_prefix)),
        ("Supplementary Table ST7", "Multiblock model coefficients", "{0}/supplementary/tables/supplementary_table_st7_multiblock_model_coefficients.csv".format(output_prefix)),
        ("Supplementary Table ST8", "Wavelength-by-wavelength statistics", "{0}/supplementary/tables/supplementary_table_st8_wavelength_by_wavelength_statistics.csv".format(output_prefix)),
        ("Supplementary Figure S1", "Raw spectra for all samples", "{0}/supplementary/figures/supplementary_figure_s1_raw_spectra_all_samples.png".format(output_prefix)),
        ("Supplementary Figure S2", "Preprocessing comparison", "{0}/supplementary/figures/supplementary_figure_s2_preprocessing_comparison.png".format(output_prefix)),
        ("Supplementary Figure S3", "Outlier detection plots", "{0}/supplementary/figures/supplementary_figure_s3_outlier_detection.png".format(output_prefix)),
        ("Supplementary Figure S4", "Full nutrient interaction plots", "{0}/supplementary/figures/supplementary_figure_s4_full_nutrient_interactions.png".format(output_prefix)),
        ("Supplementary Figure S5", "Individual nutrient boxplots", "{0}/supplementary/figures/supplementary_figure_s5_individual_nutrient_boxplots.png".format(output_prefix)),
        ("Supplementary Figure S6", "Additional PCA plots", "{0}/supplementary/figures/supplementary_figure_s6_additional_pca_plots.png".format(output_prefix)),
        ("Supplementary Figure S7", "Full HCA dendrogram", "{0}/supplementary/figures/supplementary_figure_s7_hca_full_dendrogram.png".format(output_prefix)),
        ("Supplementary Figure S8", "ASCA-style decomposition details", "{0}/supplementary/figures/supplementary_figure_s8_asca_decomposition_details.png".format(output_prefix)),
        ("Supplementary Figure S9", "Hyperparameter tuning curves", "{0}/supplementary/figures/supplementary_figure_s9_hyperparameter_tuning.png".format(output_prefix)),
        ("Supplementary Figure S10", "Full confusion matrices", "{0}/supplementary/figures/supplementary_figure_s10_full_confusion_matrices.png".format(output_prefix)),
        ("Supplementary Figure S11", "Additional ROC curves", "{0}/supplementary/figures/supplementary_figure_s11_additional_roc_curves.png".format(output_prefix)),
        ("Supplementary Figure S12", "Full loading and correlation matrices", "{0}/supplementary/figures/supplementary_figure_s12_loading_correlation_matrices.png".format(output_prefix)),
    ]
    lines = ["artifacts:"]
    for label, title, output in entries:
        lines.append("  {0}:".format(label))
        lines.append("    title: {0}".format(title))
        lines.append("    generator: eggplant_thermal_nir.pipeline.generate_artifacts")
        lines.append("    output:")
        lines.append("      - {0}".format(output))
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def generate_artifacts(results: Mapping[str, object], output_dir: Path) -> Dict[str, str]:
    """Generate manuscript-ready tables and figures from analysis results."""

    output_root = Path(output_dir).resolve()
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    supplementary_root = output_root / "supplementary"
    supplementary_tables_dir = supplementary_root / "tables"
    supplementary_figures_dir = supplementary_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    supplementary_tables_dir.mkdir(parents=True, exist_ok=True)
    supplementary_figures_dir.mkdir(parents=True, exist_ok=True)

    design_table = results["design_table"]
    nutrient_composition_table = results["nutrient_composition_table"]
    factorial_table = results["factorial_table"]
    classification_table = results["classification_table"]
    region_summary = results["region_summary"]

    main_table_paths = {
        "Table 1": tables_dir / "table_1_experimental_design_dataset_structure.csv",
        "Table 2": tables_dir / "table_2_nutrient_composition.csv",
        "Table 3": tables_dir / "table_3_factorial_model_results.csv",
        "Table 4": tables_dir / "table_4_classification_performance_summary.csv",
        "Table 5": tables_dir / "table_5_key_spectral_regions.csv",
    }
    design_table.to_csv(main_table_paths["Table 1"], index=False)
    nutrient_composition_table.to_csv(main_table_paths["Table 2"], index=False)
    factorial_table.to_csv(main_table_paths["Table 3"], index=False)
    classification_table.to_csv(main_table_paths["Table 4"], index=False)
    region_summary.to_csv(main_table_paths["Table 5"], index=False)

    supplementary_table_paths = {
        "Supplementary Table ST1": supplementary_tables_dir / "supplementary_table_st1_raw_nutrient_dataset.csv",
        "Supplementary Table ST2": supplementary_tables_dir / "supplementary_table_st2_cleaned_matched_dataset.csv",
        "Supplementary Table ST3": supplementary_tables_dir / "supplementary_table_st3_preprocessing_pipelines_tested.csv",
        "Supplementary Table ST4": supplementary_tables_dir / "supplementary_table_st4_hyperparameter_grids.csv",
        "Supplementary Table ST5": supplementary_tables_dir / "supplementary_table_st5_full_classification_metrics_all_folds.csv",
        "Supplementary Table ST6": supplementary_tables_dir / "supplementary_table_st6_asca_variance_tables.csv",
        "Supplementary Table ST7": supplementary_tables_dir / "supplementary_table_st7_multiblock_model_coefficients.csv",
        "Supplementary Table ST8": supplementary_tables_dir / "supplementary_table_st8_wavelength_by_wavelength_statistics.csv",
    }
    results["chemistry"].to_csv(supplementary_table_paths["Supplementary Table ST1"], index=False)
    results["cleaned_dataset"].to_csv(supplementary_table_paths["Supplementary Table ST2"], index=False)
    results["preprocessing_table"].to_csv(supplementary_table_paths["Supplementary Table ST3"], index=False)
    results["hyperparameter_grid_table"].to_csv(supplementary_table_paths["Supplementary Table ST4"], index=False)
    results["classification_fold_metrics"].to_csv(supplementary_table_paths["Supplementary Table ST5"], index=False)
    results["asca_variance_table"].to_csv(supplementary_table_paths["Supplementary Table ST6"], index=False)
    results["nutrient_loadings"].to_csv(supplementary_table_paths["Supplementary Table ST7"], index=False)
    results["wavelength_statistics"].to_csv(supplementary_table_paths["Supplementary Table ST8"], index=False)

    main_figure_paths = _build_main_figure_paths(output_root)
    supplementary_figure_paths = {
        "Supplementary Figure S1": supplementary_figures_dir / "supplementary_figure_s1_raw_spectra_all_samples.png",
        "Supplementary Figure S2": supplementary_figures_dir / "supplementary_figure_s2_preprocessing_comparison.png",
        "Supplementary Figure S3": supplementary_figures_dir / "supplementary_figure_s3_outlier_detection.png",
        "Supplementary Figure S4": supplementary_figures_dir / "supplementary_figure_s4_full_nutrient_interactions.png",
        "Supplementary Figure S5": supplementary_figures_dir / "supplementary_figure_s5_individual_nutrient_boxplots.png",
        "Supplementary Figure S6": supplementary_figures_dir / "supplementary_figure_s6_additional_pca_plots.png",
        "Supplementary Figure S7": supplementary_figures_dir / "supplementary_figure_s7_hca_full_dendrogram.png",
        "Supplementary Figure S8": supplementary_figures_dir / "supplementary_figure_s8_asca_decomposition_details.png",
        "Supplementary Figure S9": supplementary_figures_dir / "supplementary_figure_s9_hyperparameter_tuning.png",
        "Supplementary Figure S10": supplementary_figures_dir / "supplementary_figure_s10_full_confusion_matrices.png",
        "Supplementary Figure S11": supplementary_figures_dir / "supplementary_figure_s11_additional_roc_curves.png",
        "Supplementary Figure S12": supplementary_figures_dir / "supplementary_figure_s12_loading_correlation_matrices.png",
    }

    render_main_figures(results, output_root)

    plot_supplementary_figure_s1_raw_spectra(
        results["spectra"],
        results["spectral_meta"],
        supplementary_figure_paths["Supplementary Figure S1"],
    )
    plot_supplementary_figure_s2_preprocessing_comparison(
        results["preprocessing_showcase"],
        results["spectral_meta"],
        supplementary_figure_paths["Supplementary Figure S2"],
    )
    plot_supplementary_figure_s3_outlier_detection(
        results["spectral_meta"],
        results["pca_scores"],
        supplementary_figure_paths["Supplementary Figure S3"],
    )
    plot_supplementary_figure_s4_full_nutrient_interactions(
        results["nutrient_summary"],
        supplementary_figure_paths["Supplementary Figure S4"],
    )
    plot_supplementary_figure_s5_individual_nutrient_boxplots(
        results["chemistry"],
        supplementary_figure_paths["Supplementary Figure S5"],
    )
    plot_supplementary_figure_s6_additional_pca(
        results["spectral_meta"],
        results["pca_scores"],
        supplementary_figure_paths["Supplementary Figure S6"],
    )
    plot_supplementary_figure_s7_hca_dendrogram(
        results["preprocessed_spectra"],
        results["spectral_meta"],
        supplementary_figure_paths["Supplementary Figure S7"],
    )
    plot_supplementary_figure_s8_asca_details(
        results["asca_variance_table"],
        supplementary_figure_paths["Supplementary Figure S8"],
    )
    plot_supplementary_figure_s9_hyperparameter_tuning(
        results["hyperparameter_search_results"],
        supplementary_figure_paths["Supplementary Figure S9"],
    )
    plot_supplementary_figure_s10_confusion_matrices(
        results["classification_summary"],
        results["confusion_tables"],
        supplementary_figure_paths["Supplementary Figure S10"],
    )
    plot_supplementary_figure_s11_roc_curves(
        results["roc_curves"],
        supplementary_figure_paths["Supplementary Figure S11"],
    )
    plot_supplementary_figure_s12_loading_correlation_matrices(
        results["wavelength_statistics"],
        results["nutrient_loadings"],
        results["association"],
        supplementary_figure_paths["Supplementary Figure S12"],
    )

    manifest_path = _write_manifest(output_root)

    artifact_paths = {
        label: str(path)
        for label, path in {
            **main_table_paths,
            **supplementary_table_paths,
            **main_figure_paths,
            **supplementary_figure_paths,
        }.items()
    }
    artifact_paths["artifact_manifest"] = str(manifest_path)
    return artifact_paths
