"""Publication-oriented plotting helpers aligned to the manuscript figure plan."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import dendrogram, linkage

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})


SPECIES_COLORS = {
    "Solanum aethiopicum": "#c4602d",
    "Solanum melongena": "#2a6fbb",
    "Solanum torvum": "#1b9e77",
}
PROCESS_COLORS = {
    "raw": "#4c566a",
    "boiled": "#e76f51",
    "blanched": "#2a9d8f",
    "overall": "#6c757d",
}
PROCESS_STYLES = {"raw": "-", "boiled": "--", "blanched": ":", "overall": "-."}
PROCESS_MARKERS = {"raw": "o", "boiled": "s", "blanched": "^", "overall": "D"}


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


SPECIES_MAP = {
    "Solanum aethiopicum": "Aet",
    "Solanum melongena": "Mel",
    "Solanum torvum": "Tor",
}

def _species_short(species: str) -> str:
    return SPECIES_MAP.get(species, species.replace("Solanum ", "S. "))


def _pretty_target(target: str) -> str:
    return {
        "species": "Species",
        "process": "Process",
        "species_process": "Species + Process",
    }.get(target, target.replace("_", " ").title())


PROCESS_MAP = {
    "raw": "Raw",
    "boiled": "Boil",
    "blanched": "Blan",
    "overall": "All"
}

def _pretty_process(process: str) -> str:
    return PROCESS_MAP.get(process, process.replace("_", " ").title())


def _process_code(process: str) -> str:
    return {
        "raw": "Raw",
        "boiled": "Boi",
        "blanched": "Bln",
        "overall": "All",
    }.get(str(process).lower(), str(process))


def _species_process_key_text() -> str:
    return "Key: Aet=S. aethiopicum, Mel=S. melongena, Tor=S. torvum, Raw=raw, Boi=boiled, Bln=blanched"


def _short_nutrient_label(nutrient: str) -> str:
    return str(nutrient).replace("_", " ").title()


def _compact_nutrient_axis_label(label: str) -> str:
    return {
        "Vitamin C (mg/100ml)": "Vit. C (mg/100ml)",
    }.get(str(label), str(label))


def _add_figure_footer(fig: plt.Figure, text: str, y_pos: float = 0.01) -> None:
    fig.text(0.5, y_pos, text, ha="center", va="bottom", fontsize=9, color="#495057")


def _style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis in {"x", "y", "both"}:
        ax.grid(axis=grid_axis, color="#d9dfe7", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)


def _limit_numeric_ticks(
    ax: plt.Axes,
    x: int | None = None,
    y: int | None = None,
    *,
    integer_x: bool = False,
    integer_y: bool = False,
) -> None:
    if x is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=x, integer=integer_x, min_n_ticks=3))
    if y is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y, integer=integer_y, min_n_ticks=3))


def _annotate_bar_values(ax: plt.Axes, decimals: int = 0, fontsize: int = 9) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        offset = max(abs(height) * 0.02, 0.01)
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            height + offset,
            f"{height:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#374151",
        )


def _heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: Iterable[str],
    col_labels: Iterable[str],
    title: str,
    cmap: str = "RdBu_r",
) -> None:
    row_labels = list(row_labels)
    col_labels = list(col_labels)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    
    # Include values in the plots
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            # Use white text for dark backgrounds and black for light backgrounds
            text_color = "white" if np.abs(val) > np.max(np.abs(matrix))/2 else "black"
            ax.text(j, i, f"{val:.2g}", ha="center", va="center", color=text_color, fontsize=10)
            
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_mean_spectra(ax: plt.Axes, spectra: pd.DataFrame, meta: pd.DataFrame, title: str) -> None:
    wavelengths = pd.to_numeric(pd.Index(spectra.columns), errors="coerce")
    for (species, process), index in meta.groupby(["species", "process"]).groups.items():
        mean_values = spectra.loc[list(index)].mean(axis=0).to_numpy(dtype=float)
        ax.plot(
            wavelengths,
            mean_values,
            color=SPECIES_COLORS.get(species, "#444444"),
            linestyle=PROCESS_STYLES.get(process, "-"),
            linewidth=1.7,
            alpha=0.9,
            label="{0} | {1}".format(_species_short(species), _process_code(process)),
        )
    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    _limit_numeric_ticks(ax, x=6, y=5)


def _draw_no_data(ax: plt.Axes, title: str, message: str = "No data available") -> None:
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10, color="#666666")


def _compact_class_label(label: str) -> str:
    text = str(label)
    if " | " not in text:
        if text.startswith("Solanum "):
            return _species_short(text)
        return _process_code(text)
    species, process = text.split(" | ", 1)
    return "{0}\n{1}".format(_species_short(species), _process_code(process))


def plot_figure_1_framework(
    sample_metadata: pd.DataFrame,
    spectra: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Figure 1: study design and integrated workflow."""

    fig = plt.figure(figsize=(16, 10))
    axes = fig.subplot_mosaic(
        """
        AB
        CD
        """,
        gridspec_kw={"wspace": 0.30, "hspace": 0.35},
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08)

    counts = sample_metadata.groupby(["species", "process"]).size().unstack(fill_value=0)
    _heatmap(
        axes["A"],
        counts.to_numpy(dtype=float),
        [_species_short(label) for label in counts.index],
        [_pretty_process(label) for label in counts.columns],
        "Experimental design (replicates per cell)",
        cmap="Blues",
    )
    axes["A"].set_xlabel("Processing state")
    axes["A"].set_ylabel("Species")
    _panel_label(axes["A"], "A")

    axes["B"].axis("off")
    workflow_boxes = [
        (0.10, 0.72, "Raw fruit\nsampling"),
        (0.38, 0.72, "Domestic\nprocessing"),
        (0.72, 0.72, "Wet chemistry\n+ NIR acquisition"),
        (0.20, 0.28, "Factorial\nstatistics"),
        (0.48, 0.28, "Grouped CV\nclassification"),
        (0.78, 0.28, "Nutrient-spectrum\nintegration"),
    ]
    for x_pos, y_pos, label in workflow_boxes:
        axes["B"].text(
            x_pos,
            y_pos,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.5", "fc": "#f6f7fb", "ec": "#30475e", "lw": 1.2},
        )
    arrows = [
        ((0.18, 0.72), (0.30, 0.72)),
        ((0.46, 0.72), (0.58, 0.72)),
        ((0.72, 0.60), (0.24, 0.38)),
        ((0.72, 0.60), (0.50, 0.38)),
        ((0.72, 0.60), (0.76, 0.38)),
    ]
    for start, end in arrows:
        axes["B"].annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#30475e"})
    axes["B"].set_title("Sample-to-analysis workflow")
    _panel_label(axes["B"], "B")

    data_metrics = {
        "Chemistry samples": int(sample_metadata["analysis_sample_id"].nunique()),
        "Matched spectra": int(sample_metadata["total_scans"].sum()),
        "Wavelengths": int(spectra.shape[1]),
        "Species x process cells": int(counts.size),
    }
    axes["C"].bar(
        ["Chemistry\nsamples", "Matched\nspectra", "Wave-\nlengths", "Species x\nprocess cells"],
        list(data_metrics.values()),
        color=["#577590", "#43aa8b", "#f8961e", "#8f5fa8"],
    )
    _style_axes(axes["C"], grid_axis="y")
    axes["C"].set_ylabel("Count")
    axes["C"].set_title("Dataset structure")
    _limit_numeric_ticks(axes["C"], y=5, integer_y=True)
    _annotate_bar_values(axes["C"], decimals=0, fontsize=8)
    _panel_label(axes["C"], "C")

    axes["D"].axis("off")
    pipeline_labels = [
        "Pre-\nprocessing",
        "Explore\nspectra",
        "Grouped\nvalidation",
        "Integrate\nblocks",
    ]
    x_positions = [0.10, 0.35, 0.62, 0.88]
    for x_pos, label in zip(x_positions, pipeline_labels):
        axes["D"].text(
            x_pos,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.45", "fc": "#eef4f7", "ec": "#355070", "lw": 1.1},
        )
    for left, right in zip(x_positions[:-1], x_positions[1:]):
        axes["D"].annotate("", xy=(right - 0.08, 0.5), xytext=(left + 0.08, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.4})
    axes["D"].set_title("Analytical pipeline")
    _panel_label(axes["D"], "D")

    fig.suptitle("Figure 1. Study design and integrated analytical framework", fontsize=14, y=1.02)
    _save(fig, output_path)


def plot_figure_2_nutrient_resilience(
    summary_df: pd.DataFrame,
    retention_df: pd.DataFrame,
    resilience_df: pd.DataFrame,
    factorial_table: pd.DataFrame,
    resilience_contributions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Figure 2: nutrient responses and thermal resilience."""

    fig = plt.figure(figsize=(19.2, 10.2))
    axes = fig.subplot_mosaic(
        """
        A.B.C
        D.E.F
        """,
        gridspec_kw={"wspace": 0.06, "hspace": 0.35, "width_ratios": [1.0, 0.18, 0.98, 0.32, 1.08]},
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08)

    change_matrix = retention_df.groupby(["species", "process", "nutrient"])["change_pct"].mean().unstack("nutrient")
    standardized = change_matrix.apply(
        lambda column: (column - column.mean()) / (column.std(ddof=0) or 1.0),
        axis=0,
    ).fillna(0.0)
    _heatmap(
        axes["A"],
        standardized.to_numpy(dtype=float),
        ["{0} | {1}".format(_species_short(species), process) for species, process in standardized.index],
        [label.replace("_", " ").title() for label in standardized.columns],
        "Standardized nutrient changes",
    )
    _panel_label(axes["A"], "A")

    interaction_rows = factorial_table.loc[factorial_table["effect"] == "interaction"].copy()
    interaction_rows = interaction_rows.sort_values("eta_sq", ascending=False)
    top_nutrient = interaction_rows.iloc[0]["nutrient_label"] if not interaction_rows.empty else summary_df["nutrient"].iloc[0]
    source_nutrient = interaction_rows.iloc[0]["nutrient_label"] if not interaction_rows.empty else top_nutrient
    normalized_top = (
        str(top_nutrient)
        .lower()
        .replace("(mg/100ml)", "")
        .replace("(%)", "")
        .replace(" ", "_")
        .replace("__", "_")
        .strip("_")
    )
    selected = summary_df.loc[
        summary_df["nutrient"].str.lower().eq(normalized_top)
    ].copy()
    if selected.empty:
        selected = summary_df.loc[summary_df["nutrient"] == summary_df["nutrient"].iloc[0]].copy()
        source_nutrient = selected["nutrient"].iloc[0]
    order = ["raw", "boiled", "blanched"]
    x_positions = np.arange(len(order))
    for species, group in selected.groupby("species"):
        ordered = group.set_index("process").reindex(order).reset_index()
        axes["B"].plot(
            x_positions,
            ordered["mean_value"],
            marker="o",
            linewidth=2,
            color=SPECIES_COLORS.get(species, "#444444"),
            label=_species_short(species),
        )
    axes["B"].set_xticks(x_positions, [_pretty_process(label) for label in order])
    axes["B"].set_ylabel("Mean value")
    axes["B"].set_title("Top interaction nutrient: {0}".format(source_nutrient.replace("_", " ").title()))
    axes["B"].legend(frameon=False, fontsize=10)
    _style_axes(axes["B"], grid_axis="y")
    _limit_numeric_ticks(axes["B"], y=5)
    _panel_label(axes["B"], "B")

    effect_order = ["species", "process", "interaction"]
    for idx, effect in enumerate(effect_order):
        subset = factorial_table.loc[factorial_table["effect"] == effect].sort_values("eta_sq")
        y_positions = np.arange(subset.shape[0]) + idx * 0.08
        axes["C"].scatter(subset["eta_sq"], y_positions, s=36, label=effect.title())
    axes["C"].set_yticks(
        np.arange(factorial_table["nutrient_label"].nunique()),
        [_compact_nutrient_axis_label(label) for label in sorted(factorial_table["nutrient_label"].unique().tolist())],
    )
    axes["C"].tick_params(axis="y", pad=1)
    axes["C"].set_xlabel("Eta squared")
    axes["C"].set_title("Factorial effect sizes")
    axes["C"].legend(frameon=False, fontsize=10)
    _style_axes(axes["C"], grid_axis="x")
    _limit_numeric_ticks(axes["C"], x=5)
    _panel_label(axes["C"], "C")

    overall = resilience_df.loc[resilience_df["process"] == "overall"].sort_values("resilience_score", ascending=False)
    axes["D"].bar(
        overall["species"].map(_species_short),
        overall["resilience_score"],
        yerr=[
            overall["resilience_score"] - overall["ci_low"],
            overall["ci_high"] - overall["resilience_score"],
        ],
        color=[SPECIES_COLORS.get(species, "#444444") for species in overall["species"]],
        capsize=4,
    )
    axes["D"].set_ylim(0, 1.05)
    axes["D"].set_ylabel("Integrated resilience index")
    axes["D"].set_title("Overall thermal resilience with 95% CI")
    _style_axes(axes["D"], grid_axis="y")
    _limit_numeric_ticks(axes["D"], y=5)
    _annotate_bar_values(axes["D"], decimals=2, fontsize=8)
    _panel_label(axes["D"], "D")

    process_subset = resilience_df.loc[resilience_df["process"].isin(["boiled", "blanched"])].copy()
    for process in ["boiled", "blanched"]:
        group = process_subset.loc[process_subset["process"] == process].sort_values("rank")
        axes["E"].plot(
            group["rank"],
            group["resilience_score"],
            marker=PROCESS_MARKERS[process],
            linestyle=PROCESS_STYLES[process],
            color=PROCESS_COLORS[process],
            label=_pretty_process(process),
        )
        for row in group.itertuples(index=False):
            axes["E"].text(
                row.rank + 0.04,
                row.resilience_score,
                _species_short(row.species),
                fontsize=9.5,
                fontweight="bold",
                ha="left",
                va="center",
                color="#374151",
                bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.82},
            )
    axes["E"].set_xlabel("Rank")
    axes["E"].set_ylabel("Resilience score")
    axes["E"].set_title("Species ranking by processing method")
    axes["E"].legend(frameon=False, fontsize=10)
    _style_axes(axes["E"], grid_axis="y")
    _limit_numeric_ticks(axes["E"], x=4, y=5, integer_x=True)
    _panel_label(axes["E"], "E")

    contribution_matrix = (
        resilience_contributions.pivot_table(
            index="species",
            columns="nutrient_label",
            values="mean_resilience",
            aggfunc="mean",
        )
        .fillna(0.0)
        .sort_index()
    )
    _heatmap(
        axes["F"],
        contribution_matrix.to_numpy(dtype=float),
        [_species_short(label) for label in contribution_matrix.index],
        contribution_matrix.columns,
        "Nutrients driving resilience",
        cmap="viridis",
    )
    _panel_label(axes["F"], "F")

    fig.suptitle("Figure 2. Nutrient responses and thermal resilience", fontsize=14, y=1.02)
    _save(fig, output_path)


def plot_figure_3_spectral_structure(
    raw_spectra: pd.DataFrame,
    preprocessed_spectra: pd.DataFrame,
    meta: pd.DataFrame,
    pca_scores: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    spectral_variance: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Figure 3: spectral structure and variance decomposition."""

    fig = plt.figure(figsize=(18, 11))
    axes = fig.subplot_mosaic(
        """
        ABC
        DEF
        """,
        gridspec_kw={"wspace": 0.35, "hspace": 0.35},
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.09)

    _plot_mean_spectra(axes["A"], raw_spectra, meta, "Mean raw spectra")
    species_handles = [
        Line2D(
            [0],
            [0],
            color=SPECIES_COLORS.get(species, "#444444"),
            linewidth=2.0,
            label=_species_short(species),
        )
        for species in SPECIES_MAP
    ]
    process_handles = [
        Line2D(
            [0],
            [0],
            color="#4b5563",
            linewidth=2.0,
            linestyle=PROCESS_STYLES[process],
            label=_pretty_process(process),
        )
        for process in ["raw", "boiled", "blanched"]
    ]
    species_legend = axes["A"].legend(
        handles=species_handles,
        title="Species",
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=0.92,
        fontsize=8.8,
        title_fontsize=9.5,
        labelspacing=0.3,
        handlelength=1.8,
    )
    axes["A"].add_artist(species_legend)
    axes["A"].legend(
        handles=process_handles,
        title="Process",
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=0.92,
        fontsize=8.8,
        title_fontsize=9.5,
        labelspacing=0.3,
        handlelength=2.0,
    )
    _panel_label(axes["A"], "A")

    _plot_mean_spectra(axes["B"], preprocessed_spectra, meta, "Mean preprocessed spectra")
    _panel_label(axes["B"], "B")

    joined = meta.join(pca_scores[["PC1", "PC2"]], how="inner")
    for (species, process), group in joined.groupby(["species", "process"]):
        axes["C"].scatter(
            group["PC1"],
            group["PC2"],
            color=SPECIES_COLORS.get(species, "#444444"),
            marker=PROCESS_MARKERS.get(process, "o"),
            alpha=0.75,
            s=24,
        )
    position = axes["C"].get_position()
    axes["C"].set_position([position.x0, position.y0, position.width * 0.72, position.height])
    species_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=SPECIES_COLORS.get(species, "#444444"),
            markeredgecolor=SPECIES_COLORS.get(species, "#444444"),
            markersize=7,
            label=_species_short(species),
        )
        for species in SPECIES_MAP
    ]
    process_handles = [
        Line2D(
            [0],
            [0],
            marker=PROCESS_MARKERS[process],
            linestyle="None",
            color="#4b5563",
            markerfacecolor="#4b5563",
            markeredgecolor="#4b5563",
            markersize=7,
            label=_pretty_process(process),
        )
        for process in ["raw", "boiled", "blanched"]
    ]
    species_legend = axes["C"].legend(
        handles=species_handles,
        title="Species",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        handletextpad=0.4,
        labelspacing=0.4,
        fontsize=9.5,
        title_fontsize=10.5,
    )
    axes["C"].add_artist(species_legend)
    axes["C"].legend(
        handles=process_handles,
        title="Process",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.45),
        borderaxespad=0.0,
        handletextpad=0.4,
        labelspacing=0.4,
        fontsize=9.5,
        title_fontsize=10.5,
    )
    axes["C"].set_xlabel("PC1")
    axes["C"].set_ylabel("PC2")
    axes["C"].set_title("PCA scores")
    _style_axes(axes["C"], grid_axis="both")
    _limit_numeric_ticks(axes["C"], x=5, y=5)
    _panel_label(axes["C"], "C")

    dendrogram_input = (
        meta[["analysis_sample_id", "species", "process"]]
        .join(preprocessed_spectra, how="inner")
        .groupby(["analysis_sample_id", "species", "process"])
        .mean()
    )
    linkage_matrix = linkage(dendrogram_input.to_numpy(dtype=float), method="ward")
    labels = [
        "{0}-{1}".format(_species_short(species), process[:3])
        for _, species, process in dendrogram_input.index.to_list()
    ]
    dendrogram(linkage_matrix, labels=labels, ax=axes["D"], leaf_rotation=90, color_threshold=None)
    axes["D"].set_title("HCA dendrogram of sample-mean spectra")
    _panel_label(axes["D"], "D")

    axes["E"].bar(
        spectral_variance["effect"],
        spectral_variance["percent_variance"],
        color=["#577590", "#e76f51", "#2a9d8f", "#9aa1a8"][: spectral_variance.shape[0]],
    )
    axes["E"].set_ylabel("Percent of explained variance")
    axes["E"].set_title("Spectral variance partition")
    _style_axes(axes["E"], grid_axis="y")
    _limit_numeric_ticks(axes["E"], y=5)
    _annotate_bar_values(axes["E"], decimals=1, fontsize=8)
    _panel_label(axes["E"], "E")

    wavelengths = pd.to_numeric(pca_loadings.index, errors="coerce")
    axes["F"].plot(wavelengths, pca_loadings["PC1"], label="PC1 loading", color="#355070")
    if "PC2" in pca_loadings.columns:
        axes["F"].plot(wavelengths, pca_loadings["PC2"], label="PC2 loading", color="#bc6c25")
    for band in [970, 1200, 1450, 1700]:
        axes["F"].axvline(band, color="#999999", linewidth=0.8, linestyle="--", alpha=0.7)
    axes["F"].set_xlabel("Wavelength (nm)")
    axes["F"].set_ylabel("Loading weight")
    axes["F"].set_title("Principal spectral loadings")
    axes["F"].legend(frameon=False, fontsize=10)
    _limit_numeric_ticks(axes["F"], x=6, y=5)
    _panel_label(axes["F"], "F")

    fig.suptitle("Figure 3. Spectral structure and variance decomposition", fontsize=14, y=1.02)
    _add_figure_footer(fig, _species_process_key_text() + "; linestyle/marker encode process")
    _save(fig, output_path)


def plot_figure_4_classification_rigor(
    classification_summary: pd.DataFrame,
    leakage_summary: pd.DataFrame,
    best_confusion: pd.DataFrame,
    permutation_results: pd.DataFrame,
    vip_scores: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Figure 4: classification performance and validation rigor."""

    fig = plt.figure(figsize=(18, 11))
    axes = fig.subplot_mosaic(
        """
        ABC
        DEF
        """,
        gridspec_kw={"wspace": 0.4, "hspace": 0.35},
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.13)

    grouped = classification_summary.loc[classification_summary["split_strategy"] == "grouped"].copy()
    comparison = (
        grouped.groupby(["target", "model"], as_index=False)["balanced_accuracy"]
        .max()
        .sort_values(["target", "balanced_accuracy"], ascending=[True, False])
    )
    comparison_matrix = (
        comparison.pivot(index="target", columns="model", values="balanced_accuracy")
        .reindex(index=["process", "species", "species_process"], columns=["plsda", "svm", "rf"])
    )
    x_positions = np.arange(comparison_matrix.shape[0])
    width = 0.24
    model_colors = {"plsda": "#577590", "svm": "#4d908e", "rf": "#f8961e"}
    for idx, model_name in enumerate(comparison_matrix.columns):
        axes["A"].bar(
            x_positions + (idx - 1) * width,
            comparison_matrix[model_name],
            width=width,
            color=model_colors[model_name],
            label=model_name.upper(),
        )
    axes["A"].set_xticks(x_positions, [_pretty_target(target) for target in comparison_matrix.index])
    axes["A"].set_ylim(0, 1.0)
    axes["A"].set_ylabel("Balanced accuracy")
    axes["A"].set_title("Best grouped CV score by model and task")
    axes["A"].legend(frameon=False, fontsize=10, ncol=3, loc="upper right")
    _style_axes(axes["A"], grid_axis="y")
    _limit_numeric_ticks(axes["A"], y=5)
    _panel_label(axes["A"], "A")

    image = axes["B"].imshow(best_confusion.to_numpy(dtype=float), cmap="Blues")
    axes["B"].set_xticks(
        np.arange(best_confusion.shape[1]),
        [_compact_class_label(label) for label in best_confusion.columns],
        rotation=0,
    )
    axes["B"].set_yticks(np.arange(best_confusion.shape[0]), [_compact_class_label(label) for label in best_confusion.index])
    axes["B"].tick_params(axis="both", labelsize=9)
    axes["B"].set_title("Best grouped confusion matrix")
    plt.colorbar(image, ax=axes["B"], fraction=0.046, pad=0.04)
    _panel_label(axes["B"], "B")

    best_metrics = (
        grouped.sort_values(["target", "balanced_accuracy", "macro_f1"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .first()
    )
    metric_positions = np.arange(best_metrics.shape[0])
    width = 0.24
    for idx, metric in enumerate(["accuracy", "balanced_accuracy", "macro_f1"]):
        axes["C"].bar(metric_positions + (idx - 1) * width, best_metrics[metric], width=width, label=metric.replace("_", " ").title())
    axes["C"].set_xticks(metric_positions, best_metrics["target"].map(_pretty_target))
    axes["C"].set_ylim(0, 1.0)
    axes["C"].set_title("Best grouped metrics by task")
    axes["C"].legend(frameon=False, fontsize=10)
    _style_axes(axes["C"], grid_axis="y")
    _limit_numeric_ticks(axes["C"], y=5)
    _panel_label(axes["C"], "C")

    permuted = permutation_results.loc[permutation_results["kind"] == "permuted", "balanced_accuracy"]
    observed = permutation_results.loc[permutation_results["kind"] == "observed", "balanced_accuracy"].iloc[0]
    axes["D"].hist(permuted, bins=12, color="#adb5bd", edgecolor="white")
    axes["D"].axvline(observed, color="#d00000", linewidth=2, label="Observed")
    axes["D"].set_xlabel("Balanced accuracy")
    axes["D"].set_ylabel("Permutation count")
    axes["D"].set_title("Permutation null distribution")
    axes["D"].legend(frameon=False, fontsize=10)
    _style_axes(axes["D"], grid_axis="y")
    _limit_numeric_ticks(axes["D"], x=5, y=5, integer_y=True)
    _panel_label(axes["D"], "D")

    merged = best_metrics.merge(leakage_summary, on=["target", "model", "preprocessing"], how="left")
    x_positions = np.arange(merged.shape[0])
    axes["E"].bar(x_positions - 0.15, merged["balanced_accuracy"], width=0.3, label="Grouped CV", color="#355070")
    axes["E"].bar(x_positions + 0.15, merged["naive_balanced_accuracy"], width=0.3, label="Naive CV", color="#e76f51")
    axes["E"].set_xticks(x_positions, merged["target"].map(_pretty_target))
    axes["E"].set_ylim(0, 1.0)
    axes["E"].set_ylabel("Balanced accuracy")
    axes["E"].set_title("Leakage comparison")
    axes["E"].legend(frameon=False, fontsize=10)
    _style_axes(axes["E"], grid_axis="y")
    _limit_numeric_ticks(axes["E"], y=5)
    _panel_label(axes["E"], "E")

    axes["F"].plot(vip_scores["wavelength_nm"], vip_scores["vip_score"], color="#355070", linewidth=1.7)
    top_vip = vip_scores.nlargest(10, "vip_score").sort_values("wavelength_nm")
    axes["F"].scatter(top_vip["wavelength_nm"], top_vip["vip_score"], color="#e76f51", s=20)
    axes["F"].axhline(1.0, color="#999999", linestyle="--", linewidth=0.8)
    axes["F"].set_xlabel("Wavelength (nm)")
    axes["F"].set_ylabel("VIP score")
    axes["F"].set_title("PLS-DA VIP wavelength importance")
    _style_axes(axes["F"], grid_axis="y")
    _limit_numeric_ticks(axes["F"], x=6, y=5)
    _panel_label(axes["F"], "F")

    fig.suptitle("Figure 4. Classification performance and validation rigor", fontsize=14, y=1.02)
    _add_figure_footer(fig, _species_process_key_text(), y_pos=0.08)
    _save(fig, output_path)


def plot_figure_5_integration_story(
    association_df: pd.DataFrame,
    key_wavelengths: pd.DataFrame,
    nutrient_loadings: pd.DataFrame,
    region_summary: pd.DataFrame,
    band_correlation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Figure 5: nutrient-spectrum integration and interpretation."""

    fig = plt.figure(figsize=(18, 10.5))
    axes = fig.subplot_mosaic(
        """
        ABC
        DEE
        """,
        gridspec_kw={"wspace": 0.45, "hspace": 0.35},
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08)

    axes["A"].axhline(0, color="#cccccc", linewidth=0.8)
    axes["A"].axvline(0, color="#cccccc", linewidth=0.8)
    axes["A"].scatter(
        nutrient_loadings["component_1_loading"],
        nutrient_loadings["component_2_loading"],
        color="#355070",
        s=35,
    )
    for row in nutrient_loadings.itertuples(index=False):
        axes["A"].text(row.component_1_loading, row.component_2_loading, _short_nutrient_label(row.nutrient), fontsize=8)
    axes["A"].set_xlabel("Component 1")
    axes["A"].set_ylabel("Component 2")
    axes["A"].set_title("Nutrient loading space")
    _style_axes(axes["A"], grid_axis="both")
    _limit_numeric_ticks(axes["A"], x=5, y=5)
    _panel_label(axes["A"], "A")

    axes["B"].axis("off")
    axes["B"].set_xlim(0, 1)
    axes["B"].set_ylim(0, 1)
    top_edges = key_wavelengths.sort_values("abs_correlation", ascending=False).head(12)
    nutrients = top_edges["nutrient"].drop_duplicates().tolist()
    region_rows = region_summary.head(6).reset_index(drop=True)
    regions = region_rows["interval_nm"].tolist()
    region_codes = {interval: "R{0}".format(index + 1) for index, interval in enumerate(regions)}
    nutrient_y = np.linspace(0.86, 0.20, num=max(len(nutrients), 1))
    region_y = np.linspace(0.86, 0.20, num=max(len(regions), 1))
    nutrient_positions = {nutrient: float(y_pos) for nutrient, y_pos in zip(nutrients, nutrient_y)}
    region_positions = {region: float(y_pos) for region, y_pos in zip(regions, region_y)}
    axes["B"].text(0.11, 0.95, "Nutrients", fontsize=11, fontweight="bold", ha="left", va="center", color="#355070")
    axes["B"].text(0.83, 0.95, "Key regions", fontsize=11, fontweight="bold", ha="right", va="center", color="#bc6c25")
    line_scale = float(top_edges["abs_correlation"].max()) if not top_edges.empty else 1.0
    for nutrient in nutrients:
        y_pos = nutrient_positions[nutrient]
        axes["B"].text(
            0.08,
            y_pos,
            _short_nutrient_label(nutrient),
            ha="left",
            va="center",
            fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.26", "fc": "#eef4f7", "ec": "#355070", "lw": 1.2},
        )
    for region in regions:
        y_pos = region_positions[region]
        axes["B"].text(
            0.78,
            y_pos,
            region_codes[region],
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.24", "fc": "#fef3e2", "ec": "#bc6c25", "lw": 1.2},
        )
    for edge in top_edges.itertuples(index=False):
        source_y = nutrient_positions[edge.nutrient]
        region_match = region_rows.iloc[(region_rows["center_wavelength_nm"] - edge.wavelength_nm).abs().argmin()]
        if region_match["interval_nm"] not in regions:
            continue
        target_y = region_positions[region_match["interval_nm"]]
        strength = float(edge.abs_correlation) / (line_scale or 1.0)
        axes["B"].plot(
            [0.28, 0.72],
            [source_y, target_y],
            color="#7a7a7a",
            alpha=0.35 + 0.45 * strength,
            linewidth=0.9 + 1.5 * strength,
        )
    axes["B"].set_title("Nutrient-region association network")
    _panel_label(axes["B"], "B")

    axes["C"].plot(association_df["wavelength_nm"], association_df["component_1_weight"], color="#355070", label="Component 1")
    axes["C"].plot(association_df["wavelength_nm"], association_df["component_2_weight"], color="#bc6c25", label="Component 2")
    for row in region_summary.itertuples(index=False):
        axes["C"].axvspan(
            float(str(row.interval_nm).split("-")[0]),
            float(str(row.interval_nm).split("-")[1]),
            color="#999999",
            alpha=0.08,
        )
    axes["C"].set_xlabel("Wavelength (nm)")
    axes["C"].set_ylabel("Weight")
    axes["C"].set_title("Spectral loading profiles")
    axes["C"].legend(frameon=False, fontsize=10)
    _style_axes(axes["C"], grid_axis="y")
    _limit_numeric_ticks(axes["C"], x=6, y=5)
    _panel_label(axes["C"], "C")

    correlation_matrix = (
        band_correlation_df.pivot_table(index="nutrient", columns="band", values="correlation", aggfunc="mean")
        .fillna(0.0)
        .sort_index()
    )
    _heatmap(
        axes["D"],
        correlation_matrix.to_numpy(dtype=float),
        correlation_matrix.index,
        correlation_matrix.columns,
        "Nutrient-band correlations",
    )
    _panel_label(axes["D"], "D")

    axes["E"].axis("off")
    axes["E"].set_xlim(0, 1)
    axes["E"].set_ylim(0, 1)
    axes["E"].text(
        0.02,
        1.02,
        "Region key map",
        transform=axes["E"].transAxes,
        ha="left",
        va="bottom",
        fontsize=plt.rcParams["axes.titlesize"],
        color="#1f2937",
    )
    for idx, row in enumerate(region_rows.itertuples(index=False)):
        code = region_codes[row.interval_nm]
        top_y = 0.94 - idx * 0.155
        axes["E"].text(
            0.02,
            top_y,
            code,
            fontsize=10.5,
            fontweight="bold",
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.22", "fc": "#fef3e2", "ec": "#bc6c25", "lw": 1.1},
        )
        axes["E"].text(
            0.16,
            top_y,
            "{0}  {1}".format(row.interval_nm, row.assigned_bond),
            fontsize=10.2,
            fontweight="bold",
            ha="left",
            va="top",
            color="#1f2937",
        )
        detail = "{0} | {1}".format(_short_nutrient_label(row.dominant_nutrient), row.interpretation)
        axes["E"].text(
            0.16,
            top_y - 0.055,
            textwrap.fill(detail, width=38),
            fontsize=9.4,
            ha="left",
            va="top",
            color="#495057",
        )
    _panel_label(axes["E"], "E")

    fig.suptitle("Figure 5. Nutrient-spectrum integration and interpretation", fontsize=14, y=1.02)
    _save(fig, output_path)


def plot_supplementary_figure_s1_raw_spectra(
    raw_spectra: pd.DataFrame,
    meta: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Supplementary Figure S1: raw spectra for all scans."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    wavelengths = pd.to_numeric(pd.Index(raw_spectra.columns), errors="coerce")

    for species, group in meta.groupby("species"):
        axes[0, 0].plot(
            wavelengths,
            raw_spectra.loc[group.index].T,
            color=SPECIES_COLORS.get(species, "#444444"),
            alpha=0.08,
            linewidth=0.7,
        )
    axes[0, 0].set_title("All raw scans colored by species")
    axes[0, 0].set_xlabel("Wavelength (nm)")
    axes[0, 0].set_ylabel("Absorbance")
    _limit_numeric_ticks(axes[0, 0], x=6, y=5)
    _panel_label(axes[0, 0], "A")

    for process, group in meta.groupby("process"):
        axes[0, 1].plot(
            wavelengths,
            raw_spectra.loc[group.index].T,
            color=PROCESS_COLORS.get(process, "#444444"),
            alpha=0.08,
            linewidth=0.7,
        )
    axes[0, 1].set_title("All raw scans colored by process")
    axes[0, 1].set_xlabel("Wavelength (nm)")
    axes[0, 1].set_ylabel("Absorbance")
    _limit_numeric_ticks(axes[0, 1], x=6, y=5)
    _panel_label(axes[0, 1], "B")

    species_means = meta.join(raw_spectra, how="inner").groupby("species")[raw_spectra.columns].mean()
    for species in species_means.index:
        axes[1, 0].plot(
            wavelengths,
            species_means.loc[species].to_numpy(dtype=float),
            linewidth=2,
            color=SPECIES_COLORS.get(species, "#444444"),
            label=_species_short(species),
        )
    axes[1, 0].set_title("Mean raw spectra by species")
    axes[1, 0].set_xlabel("Wavelength (nm)")
    axes[1, 0].set_ylabel("Absorbance")
    axes[1, 0].legend(frameon=False, fontsize=10)
    _limit_numeric_ticks(axes[1, 0], x=6, y=5)
    _panel_label(axes[1, 0], "C")

    process_means = meta.join(raw_spectra, how="inner").groupby("process")[raw_spectra.columns].mean()
    for process in process_means.index:
        axes[1, 1].plot(
            wavelengths,
            process_means.loc[process].to_numpy(dtype=float),
            linewidth=2,
            color=PROCESS_COLORS.get(process, "#444444"),
            label=_pretty_process(process),
        )
    axes[1, 1].set_title("Mean raw spectra by process")
    axes[1, 1].set_xlabel("Wavelength (nm)")
    axes[1, 1].set_ylabel("Absorbance")
    axes[1, 1].legend(frameon=False, fontsize=10)
    _limit_numeric_ticks(axes[1, 1], x=6, y=5)
    _panel_label(axes[1, 1], "D")

    fig.suptitle("Supplementary Figure S1. Raw spectra across all samples", fontsize=14)
    _save(fig, output_path)


def plot_supplementary_figure_s2_preprocessing_comparison(
    preprocessing_showcase: Dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Supplementary Figure S2: preprocessing comparison panels."""

    methods = list(preprocessing_showcase.keys())
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)
    flat_axes = axes.ravel()
    shared_handles = []
    shared_labels = []
    for axis, method in zip(flat_axes, methods):
        _plot_mean_spectra(axis, preprocessing_showcase[method], meta, method.replace("_", " ").title())
        if not shared_handles:
            shared_handles, shared_labels = axis.get_legend_handles_labels()
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    for axis, label in zip(flat_axes, list("ABCDEF")):
        _panel_label(axis, label)
    for axis in flat_axes[len(methods):]:
        axis.axis("off")
    fig.suptitle("Supplementary Figure S2. Spectral preprocessing comparison", fontsize=14)
    if shared_handles:
        fig.legend(
            shared_handles[:9],
            shared_labels[:9],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=3,
            frameon=False,
            fontsize=9.0,
            title="Species | Process key",
        )
    _save(fig, output_path)


def plot_supplementary_figure_s3_outlier_detection(
    meta: pd.DataFrame,
    pca_scores: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Supplementary Figure S3: PCA-based outlier diagnostics."""

    joined = meta.join(pca_scores, how="inner")
    numeric_scores = joined[[column for column in pca_scores.columns if column.startswith("PC")]].copy()
    standardized = (numeric_scores - numeric_scores.mean()) / numeric_scores.std(ddof=0).replace(0, 1.0)
    joined["score_distance"] = np.sqrt((standardized.fillna(0.0) ** 2).sum(axis=1))
    cutoff = float(joined["score_distance"].quantile(0.975))
    top_outliers = joined.nlargest(12, "score_distance")

    fig, ax_grid = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = ax_grid.ravel()

    for species, group in joined.groupby("species"):
        axes[0].scatter(
            group["PC1"],
            group["PC2"],
            color=SPECIES_COLORS.get(species, "#444444"),
            alpha=0.55,
            s=18,
            label=_species_short(species),
        )
    axes[0].scatter(top_outliers["PC1"], top_outliers["PC2"], facecolors="none", edgecolors="black", s=70, linewidth=1.1)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("PCA score map with high-distance scans")
    axes[0].legend(frameon=False, fontsize=10)
    _limit_numeric_ticks(axes[0], x=5, y=5)
    _panel_label(axes[0], "A")

    axes[1].bar(np.arange(top_outliers.shape[0]), top_outliers["score_distance"], color="#bc6c25")
    axes[1].axhline(cutoff, color="#355070", linestyle="--", linewidth=1.0, label="97.5th percentile")
    code_map = {scan: f"O{i+1}" for i, scan in enumerate(top_outliers["scan_id"])}
    codes = [code_map[scan] for scan in top_outliers["scan_id"]]
    axes[1].set_xticks(np.arange(top_outliers.shape[0]), codes, rotation=45)
    axes[1].set_ylabel("Score distance")
    axes[1].set_title("Top candidate outliers")
    axes[1].legend(frameon=False, fontsize=10)
    _limit_numeric_ticks(axes[1], y=5)
    _panel_label(axes[1], "B")

    axes[2].hist(joined["score_distance"], bins=20, color="#adb5bd", edgecolor="white")
    axes[2].axvline(cutoff, color="#d00000", linewidth=1.6, linestyle="--")
    axes[2].set_xlabel("Score distance")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Distance distribution")
    _limit_numeric_ticks(axes[2], x=5, y=5, integer_y=True)
    _panel_label(axes[2], "C")

    axes[3].axis("off")
    key_text = "Outlier Code Key:\n" + "\n".join([f"{code}: {scan}" for scan, code in code_map.items()])
    axes[3].text(0.1, 0.9, key_text, ha="left", va="top", fontsize=12, transform=axes[3].transAxes, bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#adb5bd"))
    axes[3].set_title("Code Key Map")
    fig.suptitle("Supplementary Figure S3. Outlier detection diagnostics", fontsize=14)
    _save(fig, output_path)


def plot_supplementary_figure_s4_full_nutrient_interactions(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Supplementary Figure S4: nutrient interaction plots for all variables."""

    nutrients = sorted(summary_df["nutrient"].unique().tolist())
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    flat_axes = axes.ravel()
    process_order = ["raw", "boiled", "blanched"]
    x_positions = np.arange(len(process_order))

    for axis, nutrient in zip(flat_axes, nutrients):
        subset = summary_df.loc[summary_df["nutrient"] == nutrient].copy()
        for species, group in subset.groupby("species"):
            ordered = group.set_index("process").reindex(process_order).reset_index()
            axis.plot(
                x_positions,
                ordered["mean_value"],
                marker="o",
                linewidth=2,
                color=SPECIES_COLORS.get(species, "#444444"),
                label=_species_short(species),
            )
        axis.set_xticks(x_positions, [_pretty_process(label) for label in process_order], rotation=20)
        axis.set_title(nutrient.replace("_", " ").title())
        axis.set_ylabel("Mean value")
        _limit_numeric_ticks(axis, y=5)
    for axis, label in zip(flat_axes, list("ABCDEFGHI")):
        _panel_label(axis, label)
    for axis in flat_axes[len(nutrients):]:
        axis.axis("off")
    flat_axes[0].legend(frameon=False, fontsize=9, ncol=3)
    fig.suptitle("Supplementary Figure S4. Full nutrient interaction plots", fontsize=14)
    _save(fig, output_path)


def plot_supplementary_figure_s5_individual_nutrient_boxplots(
    chemistry: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Supplementary Figure S5: individual nutrient boxplots."""

    nutrients = sorted(chemistry["nutrient"].unique().tolist())
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    flat_axes = axes.ravel()
    order = []
    for species in sorted(chemistry["species"].unique().tolist()):
        for process in ["raw", "boiled", "blanched"]:
            order.append((species, process))

    for axis, nutrient in zip(flat_axes, nutrients):
        subset = chemistry.loc[chemistry["nutrient"] == nutrient].copy()
        box_data = []
        colors = []
        labels = []
        for species, process in order:
            values = subset.loc[
                (subset["species"] == species) & (subset["process"] == process),
                "value",
            ].dropna()
            box_data.append(values.to_numpy(dtype=float))
            colors.append(SPECIES_COLORS.get(species, "#cccccc"))
            labels.append("{0}\n{1}".format(_species_short(species), process[:3]))
        boxplot = axis.boxplot(box_data, patch_artist=True, widths=0.65, showfliers=False)
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        axis.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha="right")
        axis.set_title(nutrient.replace("_", " ").title())
        axis.set_ylabel("Value")
        _limit_numeric_ticks(axis, y=5)
    for axis, label in zip(flat_axes, list("ABCDEFGHI")):
        _panel_label(axis, label)
    for axis in flat_axes[len(nutrients):]:
        axis.axis("off")
    fig.suptitle("Supplementary Figure S5. Individual nutrient boxplots", fontsize=14)
    _save(fig, output_path)


def plot_supplementary_figure_s6_additional_pca(
    meta: pd.DataFrame,
    pca_scores: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Supplementary Figure S6: additional PCA score maps."""

    joined = meta.join(pca_scores, how="inner")
    pairs = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]
    fig, ax_grid = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = ax_grid.ravel()
    for axis, (x_col, y_col), label in zip(axes, pairs, list("ABC")):
        if x_col not in joined.columns or y_col not in joined.columns:
            _draw_no_data(axis, "{0} vs {1}".format(x_col, y_col))
            continue
        for (species, process), group in joined.groupby(["species", "process"]):
            axis.scatter(
                group[x_col],
                group[y_col],
                color=SPECIES_COLORS.get(species, "#444444"),
                marker=PROCESS_MARKERS.get(process, "o"),
                alpha=0.7,
                s=18,
                label="{0} | {1}".format(_species_short(species), process),
            )
        axis.set_xlabel(x_col)
        axis.set_ylabel(y_col)
        axis.set_title("{0} vs {1}".format(x_col, y_col))
        _limit_numeric_ticks(axis, x=5, y=5)
        _panel_label(axis, label)
    axes[3].axis("off")
    axes[3].text(
        0.05,
        0.85,
        "Key map",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )
    axes[3].text(
        0.05,
        0.68,
        "Color: Aet / Mel / Tor",
        fontsize=10,
        ha="left",
        va="top",
    )
    axes[3].text(
        0.05,
        0.52,
        "Marker: o = Raw, s = Boi, ^ = Bln",
        fontsize=10,
        ha="left",
        va="top",
    )
    axes[3].text(
        0.05,
        0.34,
        _species_process_key_text(),
        fontsize=9,
        ha="left",
        va="top",
        color="#495057",
    )
    fig.suptitle("Supplementary Figure S6. Additional PCA score plots", fontsize=14)
    _save(fig, output_path)


def plot_supplementary_figure_s7_hca_dendrogram(
    preprocessed_spectra: pd.DataFrame,
    meta: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Supplementary Figure S7: full HCA dendrogram."""

    dendrogram_input = (
        meta[["analysis_sample_id", "species", "process"]]
        .join(preprocessed_spectra, how="inner")
        .groupby(["analysis_sample_id", "species", "process"])
        .mean()
    )
    fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)
    linkage_matrix = linkage(dendrogram_input.to_numpy(dtype=float), method="ward")
    labels = [
        "{0}-{1}".format(_species_short(species), process[:3])
        for _, species, process in dendrogram_input.index.to_list()
    ]
    dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=90, color_threshold=None)
    ax.set_title("Full hierarchical clustering dendrogram of sample-mean spectra")
    ax.set_ylabel("Ward distance")
    _limit_numeric_ticks(ax, y=5)
    _panel_label(ax, "A")
    fig.suptitle("Supplementary Figure S7. Full HCA dendrogram", fontsize=14)
    _save(fig, output_path)


def plot_supplementary_figure_s8_asca_details(
    asca_variance_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Supplementary Figure S8: ASCA-style decomposition details."""

    fig, ax_grid = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = ax_grid.ravel()
    if asca_variance_table.empty:
        for axis, label in zip(axes, list("ABC")):
            _draw_no_data(axis, "ASCA-style details")
            _panel_label(axis, label)
        axes[3].axis("off")
        fig.suptitle("Supplementary Figure S8. ASCA-style decomposition details", fontsize=14)
        _save(fig, output_path)
        return

    ordered_effects = ["species", "process", "interaction", "residual"]
    percent_matrix = (
        asca_variance_table.pivot_table(
            index="effect",
            columns="component",
            values="percent_within_component",
            aggfunc="mean",
        )
        .reindex(ordered_effects)
        .fillna(0.0)
    )
    _heatmap(
        axes[0],
        percent_matrix.to_numpy(dtype=float),
        percent_matrix.index,
        percent_matrix.columns,
        "Variance share within component",
        cmap="viridis",
    )
    _panel_label(axes[0], "A")

    f_matrix = (
        asca_variance_table.loc[asca_variance_table["effect"] != "residual"]
        .pivot_table(index="effect", columns="component", values="F", aggfunc="mean")
        .reindex(["species", "process", "interaction"])
        .fillna(0.0)
    )
    _heatmap(
        axes[1],
        f_matrix.to_numpy(dtype=float),
        f_matrix.index,
        f_matrix.columns,
        "Component-level F statistics",
        cmap="magma",
    )
    _panel_label(axes[1], "B")

    stacked = (
        asca_variance_table.pivot_table(
            index="component",
            columns="effect",
            values="overall_percent_variance",
            aggfunc="sum",
        )
        .reindex(columns=ordered_effects)
        .fillna(0.0)
    )
    bottom = np.zeros(stacked.shape[0], dtype=float)
    palette = {"species": "#577590", "process": "#e76f51", "interaction": "#2a9d8f", "residual": "#adb5bd"}
    for effect in stacked.columns:
        axes[2].bar(stacked.index, stacked[effect], bottom=bottom, label=effect.title(), color=palette.get(effect, "#999999"))
        bottom += stacked[effect].to_numpy(dtype=float)
    axes[2].set_ylabel("Overall variance contribution (%)")
    axes[2].set_title("Overall decomposition by component")
    axes[2].legend(frameon=False, fontsize=9.5)
    _limit_numeric_ticks(axes[2], y=5)
    _panel_label(axes[2], "C")

    axes[3].axis("off")
    fig.suptitle("Supplementary Figure S8. ASCA-style decomposition details", fontsize=14)
    _save(fig, output_path)


def plot_supplementary_figure_s9_hyperparameter_tuning(
    tuning_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Supplementary Figure S9: hyperparameter tuning results."""

    targets = ["species", "process", "species_process"]
    fig, ax_grid = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    axes = ax_grid.ravel()
    key_blocks = []
    for axis, target, label in zip(axes, targets, list("ABC")):
        subset = tuning_results.loc[tuning_results["target"] == target].copy() if not tuning_results.empty else pd.DataFrame()
        if subset.empty:
            _draw_no_data(axis, _pretty_target(target))
            _panel_label(axis, label)
            continue
        subset = subset.sort_values("mean_balanced_accuracy", ascending=False).head(8).reset_index(drop=True)
        prefix = {"species": "S", "process": "P", "species_process": "C"}[target]
        subset["code"] = ["{0}{1:02d}".format(prefix, index + 1) for index in range(subset.shape[0])]
        x_positions = np.arange(subset.shape[0])
        axis.plot(x_positions, subset["mean_balanced_accuracy"], marker="o", color="#355070", linewidth=1.7)
        axis.scatter(x_positions, subset["mean_balanced_accuracy"], color="#bc6c25", s=24)
        axis.set_xticks(x_positions, subset["code"], rotation=0)
        axis.set_ylim(0, 1.0)
        axis.set_ylabel("Mean balanced accuracy")
        axis.grid(axis="y", alpha=0.2)
        _limit_numeric_ticks(axis, y=5)
        first_row = subset.iloc[0]
        axis.set_title("{0}: {1} ({2})".format(_pretty_target(target), str(first_row["model"]).upper(), first_row["preprocessing"]))
        _panel_label(axis, label)
        key_blocks.append(
            (
                _pretty_target(target),
                [
                    "{0}: {1}".format(code, textwrap.shorten(str(config), width=42, placeholder="..."))
                    for code, config in zip(subset["code"], subset["configuration_label"])
                ],
            )
        )
    axes[3].axis("off")
    axes[3].set_title("Configuration key map")
    y_pos = 0.96
    for block_title, rows in key_blocks:
        axes[3].text(0.02, y_pos, block_title, fontsize=11, fontweight="bold", ha="left", va="top")
        y_pos -= 0.06
        for row_text in rows:
            axes[3].text(0.04, y_pos, row_text, fontsize=8.5, ha="left", va="top")
            y_pos -= 0.05
        y_pos -= 0.04
    fig.suptitle("Supplementary Figure S9. Hyperparameter tuning curves", fontsize=14)
    _save(fig, output_path)


def plot_supplementary_figure_s10_confusion_matrices(
    classification_summary: pd.DataFrame,
    confusion_tables: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Render Supplementary Figure S10: full confusion matrices for all models."""

    targets = ["species", "process", "species_process"]
    models = ["plsda", "svm", "rf"]
    grouped = classification_summary.loc[classification_summary["split_strategy"] == "grouped"].copy()
    fig, axes = plt.subplots(len(targets), len(models), figsize=(18, 15), constrained_layout=True)
    for row_index, target in enumerate(targets):
        row_max = 0.0
        for model in models:
            subset = grouped.loc[(grouped["target"] == target) & (grouped["model"] == model)].copy()
            if subset.empty:
                continue
            best_row = subset.sort_values(["balanced_accuracy", "macro_f1"], ascending=False).iloc[0]
            confusion = confusion_tables.get(best_row["result_key"])
            if confusion is not None and not confusion.empty:
                row_max = max(row_max, float(confusion.to_numpy(dtype=float).max()))
        for column_index, model in enumerate(models):
            axis = axes[row_index, column_index]
            subset = grouped.loc[(grouped["target"] == target) & (grouped["model"] == model)].copy()
            if subset.empty:
                _draw_no_data(axis, "{0} | {1}".format(_pretty_target(target), model.upper()))
                continue
            best_row = subset.sort_values(["balanced_accuracy", "macro_f1"], ascending=False).iloc[0]
            confusion = confusion_tables.get(best_row["result_key"])
            if confusion is None or confusion.empty:
                _draw_no_data(axis, "{0} | {1}".format(_pretty_target(target), model.upper()))
                continue
            image = axis.imshow(confusion.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=row_max or None)
            axis.set_xticks(
                np.arange(confusion.shape[1]),
                [_compact_class_label(label) for label in confusion.columns],
                rotation=0,
                fontsize=8,
            )
            axis.set_yticks(np.arange(confusion.shape[0]), [_compact_class_label(label) for label in confusion.index], fontsize=8)
            axis.set_title("{0}\n{1}".format(_pretty_target(target), model.upper()))
            if column_index == len(models) - 1:
                plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis, label in zip(axes.ravel(), list("ABCDEFGHI")):
        _panel_label(axis, label)
    fig.suptitle("Supplementary Figure S10. Full confusion matrices", fontsize=14)
    _add_figure_footer(fig, _species_process_key_text())
    _save(fig, output_path)


def plot_supplementary_figure_s11_roc_curves(
    roc_curves: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Supplementary Figure S11: grouped one-vs-rest ROC curves."""

    targets = ["species", "process", "species_process"]
    fig, ax_grid = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = ax_grid.ravel()
    for axis, target, label in zip(axes, targets, list("ABC")):
        subset = roc_curves.loc[roc_curves["target"] == target].copy() if not roc_curves.empty else pd.DataFrame()
        if subset.empty:
            _draw_no_data(axis, _pretty_target(target))
            _panel_label(axis, label)
            continue
        axis.plot([0, 1], [0, 1], color="#cccccc", linestyle="--", linewidth=1.0)
        macro = subset.loc[subset["class_label"] == "macro-average"]
        axis.plot(macro["fpr"], macro["tpr"], color="#d00000", linewidth=2.2, label="Macro (AUC={0:.2f})".format(float(macro["auc"].iloc[0])))
        class_rows = subset.loc[subset["class_label"] != "macro-average"]
        for idx, (class_label, class_df) in enumerate(class_rows.groupby("class_label")):
            axis.plot(
                class_df["fpr"],
                class_df["tpr"],
                linewidth=1.0,
                alpha=0.45,
                label=_compact_class_label(class_label) if idx < 4 else None,
            )
        first_row = subset.iloc[0]
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_title("{0}: {1} ({2})".format(_pretty_target(target), str(first_row["model"]).upper(), first_row["preprocessing"]))
        axis.legend(frameon=False, fontsize=9)
        _limit_numeric_ticks(axis, x=5, y=5)
        _panel_label(axis, label)
    axes[3].axis("off")
    axes[3].set_title("Key map")
    axes[3].text(0.05, 0.82, _species_process_key_text(), fontsize=10, ha="left", va="top")
    axes[3].text(0.05, 0.62, "Displayed class legends use compact codes from the key above.", fontsize=9, ha="left", va="top", color="#495057")
    fig.suptitle("Supplementary Figure S11. Additional ROC curves", fontsize=14)
    _save(fig, output_path)


def plot_supplementary_figure_s12_loading_correlation_matrices(
    wavelength_statistics: pd.DataFrame,
    nutrient_loadings: pd.DataFrame,
    association_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render Supplementary Figure S12: full loading and correlation matrices."""

    fig, ax_grid = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = ax_grid.ravel()
    if wavelength_statistics.empty:
        for axis, label in zip(axes, list("ABC")):
            _draw_no_data(axis, "Loading and correlation matrices")
            _panel_label(axis, label)
        axes[3].axis("off")
        fig.suptitle("Supplementary Figure S12. Full loading and correlation matrices", fontsize=14)
        _save(fig, output_path)
        return

    corr_matrix = (
        wavelength_statistics.pivot_table(
            index="nutrient",
            columns="wavelength_nm",
            values="correlation",
            aggfunc="mean",
        )
        .sort_index()
        .fillna(0.0)
    )
    image = axes[0].imshow(corr_matrix.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r")
    xtick_positions = np.linspace(0, corr_matrix.shape[1] - 1, num=min(10, corr_matrix.shape[1]), dtype=int)
    xtick_labels = [int(float(corr_matrix.columns[position])) for position in xtick_positions]
    axes[0].set_xticks(xtick_positions, xtick_labels, rotation=45, ha="right")
    axes[0].set_yticks(np.arange(corr_matrix.shape[0]), corr_matrix.index)
    axes[0].set_title("Nutrient x wavelength correlation matrix")
    axes[0].set_xlabel("Wavelength (nm)")
    plt.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    _panel_label(axes[0], "A")

    loading_matrix = nutrient_loadings.set_index("nutrient")[["component_1_loading", "component_2_loading"]].fillna(0.0)
    image = axes[1].imshow(loading_matrix.to_numpy(dtype=float), aspect="auto", cmap="PuOr")
    axes[1].set_xticks(np.arange(loading_matrix.shape[1]), ["Component 1", "Component 2"])
    axes[1].set_yticks(np.arange(loading_matrix.shape[0]), loading_matrix.index)
    axes[1].set_title("Nutrient loading matrix")
    plt.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    _panel_label(axes[1], "B")

    axes[2].plot(association_df["wavelength_nm"], association_df["mean_abs_correlation"], color="#355070", linewidth=1.8, label="Mean |correlation|")
    if "component_1_weight" in association_df.columns:
        axes[2].plot(association_df["wavelength_nm"], association_df["component_1_weight"], color="#bc6c25", linewidth=1.2, label="Component 1 weight")
    if "component_2_weight" in association_df.columns:
        axes[2].plot(association_df["wavelength_nm"], association_df["component_2_weight"], color="#2a9d8f", linewidth=1.2, label="Component 2 weight")
    axes[2].set_xlabel("Wavelength (nm)")
    axes[2].set_ylabel("Magnitude")
    axes[2].set_title("Integrated spectral loading profiles")
    axes[2].legend(frameon=False, fontsize=10)
    _limit_numeric_ticks(axes[2], x=6, y=5)
    _panel_label(axes[2], "C")

    axes[3].axis("off")
    fig.suptitle("Supplementary Figure S12. Full loading and correlation matrices", fontsize=14)
    _save(fig, output_path)
