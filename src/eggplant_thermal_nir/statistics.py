"""Nutrient summary and factorial statistics."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


PROCESS_ORDER = ["raw", "boiled", "blanched"]
EFFECT_LABELS = {
    "C(species)": "species",
    "C(process)": "process",
    "C(species):C(process)": "interaction",
    "Residual": "residual",
}
NUTRIENT_LABELS = {
    "ash": "Ash (%)",
    "carb": "Carbohydrate (%)",
    "fat": "Fat (%)",
    "fibre": "Fibre (%)",
    "moisture": "Moisture (%)",
    "protein": "Protein (%)",
    "vitamin_c": "Vitamin C (mg/100ml)",
}


def compute_nutrient_summary(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize nutrient values by species and processing state."""

    summary = (
        tidy_df.groupby(["nutrient", "species", "process"])
        .agg(
            mean_value=("value", "mean"),
            sd_value=("value", "std"),
            min_value=("value", "min"),
            max_value=("value", "max"),
            n=("value", "count"),
            unit=("unit", "first"),
        )
        .reset_index()
    )
    summary["process"] = pd.Categorical(summary["process"], categories=PROCESS_ORDER, ordered=True)
    return summary.sort_values(["nutrient", "species", "process"]).reset_index(drop=True)


def fit_factorial_models(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """Fit nutrient-specific factorial models for species, process, and interaction effects."""

    results: List[pd.DataFrame] = []
    for nutrient, nutrient_df in tidy_df.groupby("nutrient"):
        model = smf.ols("value ~ C(species) * C(process)", data=nutrient_df).fit()
        anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "effect"})
        total_sum_sq = anova["sum_sq"].sum()
        anova["eta_sq"] = anova["sum_sq"] / total_sum_sq if total_sum_sq else np.nan
        anova["nutrient"] = nutrient
        anova["n_obs"] = int(nutrient_df.shape[0])
        results.append(anova)
    combined = pd.concat(results, ignore_index=True)
    return combined[
        ["nutrient", "effect", "df", "sum_sq", "F", "PR(>F)", "eta_sq", "n_obs"]
    ].sort_values(["nutrient", "effect"]).reset_index(drop=True)


def compute_retention_changes(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-replicate changes from raw to processed states."""

    raw_df = tidy_df.loc[tidy_df["process"] == "raw", ["species_code", "species", "replicate", "nutrient", "value"]]
    raw_df = raw_df.rename(columns={"value": "raw_value"})
    processed = tidy_df.loc[
        tidy_df["process"].isin(["boiled", "blanched"]),
        ["species_code", "species", "process", "replicate", "nutrient", "value"],
    ].rename(columns={"value": "processed_value"})

    merged = processed.merge(raw_df, on=["species_code", "species", "replicate", "nutrient"], how="left")
    merged["change_pct"] = ((merged["processed_value"] - merged["raw_value"]) / merged["raw_value"]) * 100.0
    merged["retention_pct"] = (merged["processed_value"] / merged["raw_value"]) * 100.0
    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged.sort_values(["species_code", "process", "replicate", "nutrient"]).reset_index(drop=True)


def compute_thermal_resilience_index(retention_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a simple thermal resilience score from nutrient retention."""

    scored = retention_df.copy()
    scored["resilience_component"] = 1.0 - (scored["retention_pct"] - 100.0).abs() / 100.0
    scored["resilience_component"] = scored["resilience_component"].clip(lower=0.0)

    def _summarize(frame: pd.DataFrame, grouping: List[str]) -> pd.DataFrame:
        summary = (
            frame.groupby(grouping)
            .agg(
                resilience_score=("resilience_component", "mean"),
                resilience_sd=("resilience_component", "std"),
                component_n=("resilience_component", "size"),
                mean_retention_pct=("retention_pct", "mean"),
                mean_change_pct=("change_pct", "mean"),
                nutrient_count=("nutrient", "nunique"),
            )
            .reset_index()
        )
        summary["resilience_se"] = summary["resilience_sd"] / np.sqrt(summary["component_n"].clip(lower=1))
        summary["ci_low"] = (summary["resilience_score"] - 1.96 * summary["resilience_se"]).clip(lower=0.0)
        summary["ci_high"] = (summary["resilience_score"] + 1.96 * summary["resilience_se"]).clip(upper=1.0)
        return summary

    by_process = _summarize(scored, ["species", "species_code", "process"])
    by_process["rank"] = by_process.groupby("process")["resilience_score"].rank(ascending=False, method="dense")

    overall = _summarize(scored, ["species", "species_code"])
    overall["process"] = "overall"
    overall["rank"] = overall["resilience_score"].rank(ascending=False, method="dense")

    combined = pd.concat([by_process, overall], ignore_index=True)
    return combined.sort_values(["process", "rank", "species"]).reset_index(drop=True)


def summarize_experimental_design(sample_metadata: pd.DataFrame) -> pd.DataFrame:
    """Build a manuscript-ready design table summarizing chemistry and spectra."""

    frame = sample_metadata.copy()
    max_nutrients = int(frame["nutrient_count"].max()) if not frame.empty else 0
    grouped = (
        frame.groupby(["species", "process"])
        .agg(
            chemistry_samples=("analysis_sample_id", "nunique"),
            matched_samples=("total_scans", lambda values: int((values > 0).sum())),
            nutrients_per_sample=("nutrient_count", "mean"),
            missing_measurements=("nutrient_count", lambda values: int((max_nutrients - values).sum())),
            total_scans=("total_scans", "sum"),
            mean_scans_per_sample=("total_scans", "mean"),
            unique_sessions=("unique_sessions", "max"),
            unique_spots=("unique_spots", "max"),
        )
        .reset_index()
    )
    grouped["nutrients_per_sample"] = grouped["nutrients_per_sample"].round(2)
    grouped["mean_scans_per_sample"] = grouped["mean_scans_per_sample"].round(2)

    if grouped.empty:
        return grouped

    overall = pd.DataFrame(
        [
            {
                "species": "Overall",
                "process": "all",
                "chemistry_samples": int(frame["analysis_sample_id"].nunique()),
                "matched_samples": int((frame["total_scans"] > 0).sum()),
                "nutrients_per_sample": round(float(frame["nutrient_count"].mean()), 2),
                "missing_measurements": int((max_nutrients - frame["nutrient_count"]).sum()),
                "total_scans": int(frame["total_scans"].sum()),
                "mean_scans_per_sample": round(float(frame["total_scans"].mean()), 2),
                "unique_sessions": int(frame["unique_sessions"].max()),
                "unique_spots": int(frame["unique_spots"].max()),
            }
        ]
    )
    return pd.concat([grouped, overall], ignore_index=True)


def format_nutrient_composition_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a wide publication-style nutrient composition table."""

    frame = summary_df.copy()
    frame["nutrient_label"] = frame["nutrient"].map(NUTRIENT_LABELS).fillna(frame["nutrient"])
    frame["value_display"] = frame.apply(
        lambda row: "{0:.2f} +/- {1:.2f}".format(
            float(row["mean_value"]),
            float(row["sd_value"]) if pd.notna(row["sd_value"]) else 0.0,
        ),
        axis=1,
    )
    wide = (
        frame.pivot_table(
            index=["species", "process"],
            columns="nutrient_label",
            values="value_display",
            aggfunc="first",
            observed=False,
        )
        .reset_index()
    )
    replicate_counts = (
        frame.groupby(["species", "process"], observed=False)["n"].max().rename("replicates").reset_index()
    )
    return replicate_counts.merge(wide, on=["species", "process"], how="left")


def summarize_factorial_results(factorial_models: pd.DataFrame) -> pd.DataFrame:
    """Tidy factorial results for manuscript tables."""

    frame = factorial_models.loc[factorial_models["effect"] != "Residual"].copy()
    frame["effect"] = frame["effect"].map(EFFECT_LABELS).fillna(frame["effect"])
    frame["nutrient_label"] = frame["nutrient"].map(NUTRIENT_LABELS).fillna(frame["nutrient"])
    frame["p_value"] = frame["PR(>F)"]
    frame["significance"] = frame["p_value"].apply(_significance_stars)
    frame["is_dominant_effect"] = (
        frame.groupby("nutrient")["eta_sq"].rank(ascending=False, method="dense").eq(1)
    )
    columns = [
        "nutrient_label",
        "effect",
        "df",
        "F",
        "p_value",
        "eta_sq",
        "significance",
        "is_dominant_effect",
        "n_obs",
    ]
    return frame[columns].sort_values(["nutrient_label", "effect"]).reset_index(drop=True)


def summarize_resilience_contributions(retention_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize which nutrients drive resilience within each species."""

    frame = retention_df.copy()
    frame["resilience_component"] = 1.0 - (frame["retention_pct"] - 100.0).abs() / 100.0
    frame["resilience_component"] = frame["resilience_component"].clip(lower=0.0)
    summary = (
        frame.groupby(["species", "nutrient"])
        .agg(
            mean_resilience=("resilience_component", "mean"),
            mean_retention_pct=("retention_pct", "mean"),
            mean_change_pct=("change_pct", "mean"),
            observations=("resilience_component", "size"),
        )
        .reset_index()
    )
    summary["nutrient_label"] = summary["nutrient"].map(NUTRIENT_LABELS).fillna(summary["nutrient"])
    return summary.sort_values(["species", "mean_resilience"], ascending=[True, False]).reset_index(drop=True)


def summarize_best_classification_results(
    classification_summary: pd.DataFrame,
    leakage_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the main classification summary table."""

    if classification_summary.empty:
        return pd.DataFrame(
            columns=[
                "target",
                "model",
                "preprocessing",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "naive_accuracy",
                "naive_balanced_accuracy",
                "naive_macro_f1",
                "leakage_delta_balanced_accuracy",
            ]
        )

    best = (
        classification_summary.loc[classification_summary["split_strategy"] == "grouped"]
        .sort_values(["target", "balanced_accuracy", "macro_f1"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .first()
    )
    merged = best.merge(
        leakage_summary,
        on=["target", "model", "preprocessing"],
        how="left",
        suffixes=("", "_naive"),
    )
    merged["leakage_delta_balanced_accuracy"] = merged["naive_balanced_accuracy"] - merged["balanced_accuracy"]
    return merged[
        [
            "target",
            "model",
            "preprocessing",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "naive_accuracy",
            "naive_balanced_accuracy",
            "naive_macro_f1",
            "leakage_delta_balanced_accuracy",
        ]
    ].sort_values("target").reset_index(drop=True)


def _significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""
