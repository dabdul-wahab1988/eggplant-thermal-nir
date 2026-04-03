"""Integration between nutrient and spectral blocks."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from eggplant_thermal_nir.data import pivot_nutrients_wide
from eggplant_thermal_nir.spectra import get_wavelength_columns


def aggregate_spectra_by_sample(spectra: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Average scan-level spectra to the analysis-sample level."""

    wavelength_columns = get_wavelength_columns(spectra)
    joined = meta[["analysis_sample_id", "species", "process", "replicate"]].join(
        spectra[wavelength_columns], how="inner"
    )
    aggregated = joined.groupby("analysis_sample_id")[wavelength_columns].mean()
    descriptors = (
        joined.groupby("analysis_sample_id")
        .agg(species=("species", "first"), process=("process", "first"), replicate=("replicate", "first"))
    )
    return descriptors.join(aggregated, how="inner")


def fit_correlation_integration(
    chemistry: pd.DataFrame,
    aggregated_spectra: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a low-rank nutrient-spectrum integration model and summarize key wavelengths."""

    if "nutrient" in chemistry.columns and "value" in chemistry.columns:
        chemistry_wide = pivot_nutrients_wide(chemistry)
    else:
        chemistry_wide = chemistry.copy()

    chemistry_wide = chemistry_wide.sort_index()
    wavelength_columns = get_wavelength_columns(aggregated_spectra)
    spectral_matrix = aggregated_spectra[wavelength_columns].sort_index()

    common_index = chemistry_wide.index.intersection(spectral_matrix.index)
    chemistry_aligned = chemistry_wide.loc[common_index].copy()
    spectra_aligned = spectral_matrix.loc[common_index].copy()
    chemistry_aligned = chemistry_aligned.apply(lambda column: column.fillna(column.mean()), axis=0)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(spectra_aligned.to_numpy(dtype=float))
    y_scaled = y_scaler.fit_transform(chemistry_aligned.to_numpy(dtype=float))

    n_components = max(1, min(2, x_scaled.shape[0] - 1, x_scaled.shape[1], y_scaled.shape[1]))
    pls = PLSRegression(n_components=n_components)
    pls.fit(x_scaled, y_scaled)

    association = pd.DataFrame({"wavelength_nm": pd.to_numeric(wavelength_columns, errors="coerce")})
    association["component_1_weight"] = pls.x_weights_[:, 0]
    if n_components > 1:
        association["component_2_weight"] = pls.x_weights_[:, 1]
    else:
        association["component_2_weight"] = np.nan

    nutrient_loadings = pd.DataFrame(
        {
            "nutrient": chemistry_aligned.columns,
            "component_1_loading": pls.y_loadings_[:, 0],
            "component_2_loading": pls.y_loadings_[:, 1] if n_components > 1 else np.nan,
        }
    )

    mean_abs_correlations = []
    key_rows = []
    for nutrient in chemistry_aligned.columns:
        nutrient_values = chemistry_aligned[nutrient].to_numpy(dtype=float)
        correlations = []
        for column_index, wavelength in enumerate(association["wavelength_nm"].tolist()):
            corr = np.corrcoef(spectra_aligned.iloc[:, column_index].to_numpy(dtype=float), nutrient_values)[0, 1]
            correlations.append(corr)
        correlations = np.asarray(correlations, dtype=float)
        mean_abs_correlations.append(np.abs(correlations))
        top_indices = np.argsort(np.abs(correlations))[::-1][:3]
        for idx in top_indices:
            wavelength = float(association.iloc[idx]["wavelength_nm"])
            key_rows.append(
                {
                    "nutrient": nutrient,
                    "wavelength_nm": wavelength,
                    "correlation": correlations[idx],
                    "abs_correlation": abs(correlations[idx]),
                    "interval_nm": "{0:.1f}-{1:.1f}".format(wavelength - 3.5, wavelength + 3.5),
                }
            )

    association["mean_abs_correlation"] = np.mean(np.vstack(mean_abs_correlations), axis=0)
    key_summary = pd.DataFrame(key_rows).sort_values(["nutrient", "abs_correlation"], ascending=[True, False])
    return (
        association.sort_values("wavelength_nm").reset_index(drop=True),
        key_summary.reset_index(drop=True),
        nutrient_loadings.sort_values("nutrient").reset_index(drop=True),
    )


def summarize_spectral_regions(key_wavelengths: pd.DataFrame, merge_tolerance_nm: float = 12.0) -> pd.DataFrame:
    """Merge nearby key wavelengths into interpretable spectral regions."""

    if key_wavelengths.empty:
        return pd.DataFrame(
            columns=[
                "interval_nm",
                "center_wavelength_nm",
                "assigned_bond",
                "associated_nutrients",
                "dominant_nutrient",
                "interpretation",
            ]
        )

    ordered = key_wavelengths.sort_values("wavelength_nm").reset_index(drop=True)
    clusters: List[List[pd.Series]] = []
    current_cluster: List[pd.Series] = []
    current_anchor = None
    for row in ordered.itertuples(index=False):
        wavelength = float(row.wavelength_nm)
        if current_anchor is None or abs(wavelength - current_anchor) <= merge_tolerance_nm:
            current_cluster.append(pd.Series(row._asdict()))
            if current_anchor is None:
                current_anchor = wavelength
        else:
            clusters.append(current_cluster)
            current_cluster = [pd.Series(row._asdict())]
            current_anchor = wavelength
    if current_cluster:
        clusters.append(current_cluster)

    rows = []
    for cluster in clusters:
        cluster_df = pd.DataFrame(cluster)
        start = float(cluster_df["wavelength_nm"].min()) - 3.5
        end = float(cluster_df["wavelength_nm"].max()) + 3.5
        center = float(cluster_df["wavelength_nm"].mean())
        nutrient_strength = (
            cluster_df.groupby("nutrient")["abs_correlation"].max().sort_values(ascending=False)
        )
        nutrients = ", ".join(nutrient_strength.index.tolist())
        dominant_nutrient = nutrient_strength.index[0]
        assigned_bond, interpretation = assign_bond_region(center)
        rows.append(
            {
                "interval_nm": "{0:.1f}-{1:.1f}".format(start, end),
                "center_wavelength_nm": round(center, 2),
                "assigned_bond": assigned_bond,
                "associated_nutrients": nutrients,
                "dominant_nutrient": dominant_nutrient,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows).sort_values("center_wavelength_nm").reset_index(drop=True)


def compute_band_correlation_matrix(
    chemistry_wide: pd.DataFrame,
    aggregated_spectra: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate spectra into interpretable bands and correlate them with nutrients."""

    wavelength_columns = get_wavelength_columns(aggregated_spectra)
    spectral_matrix = aggregated_spectra[wavelength_columns].sort_index()
    common_index = chemistry_wide.index.intersection(spectral_matrix.index)
    chemistry_aligned = chemistry_wide.loc[common_index].copy()
    spectra_aligned = spectral_matrix.loc[common_index].copy()

    bands = {
        "Water 960-980": (960.0, 980.0),
        "C-H 1150-1250": (1150.0, 1250.0),
        "Mixed 1375-1450": (1375.0, 1450.0),
        "O-H/N-H 1450-1510": (1450.0, 1510.0),
        "C-H 1670-1700": (1670.0, 1700.0),
    }
    band_values = {}
    numeric_columns = pd.to_numeric(pd.Index(wavelength_columns), errors="coerce")
    for label, (lower, upper) in bands.items():
        mask = (numeric_columns >= lower) & (numeric_columns <= upper)
        selected = [column for column, keep in zip(wavelength_columns, mask) if keep]
        if not selected:
            continue
        band_values[label] = spectra_aligned[selected].mean(axis=1)

    band_frame = pd.DataFrame(band_values, index=spectra_aligned.index)
    rows = []
    for nutrient in chemistry_aligned.columns:
        for band in band_frame.columns:
            correlation = band_frame[band].corr(chemistry_aligned[nutrient])
            rows.append({"nutrient": nutrient, "band": band, "correlation": correlation})
    return pd.DataFrame(rows)


def compute_wavelength_correlation_table(
    chemistry_wide: pd.DataFrame,
    aggregated_spectra: pd.DataFrame,
    association_df: pd.DataFrame | None = None,
    vip_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute nutrient-wise wavelength correlations with optional model annotations."""

    wavelength_columns = get_wavelength_columns(aggregated_spectra)
    spectral_matrix = aggregated_spectra[wavelength_columns].sort_index()
    common_index = chemistry_wide.index.intersection(spectral_matrix.index)
    chemistry_aligned = chemistry_wide.loc[common_index].copy()
    spectra_aligned = spectral_matrix.loc[common_index].copy()

    numeric_wavelengths = pd.to_numeric(pd.Index(wavelength_columns), errors="coerce").to_numpy(dtype=float)
    rows = []
    for nutrient in chemistry_aligned.columns:
        nutrient_values = chemistry_aligned[nutrient].to_numpy(dtype=float)
        for column_index, wavelength_nm in enumerate(numeric_wavelengths):
            correlation = np.corrcoef(
                spectra_aligned.iloc[:, column_index].to_numpy(dtype=float),
                nutrient_values,
            )[0, 1]
            assigned_bond, interpretation = assign_bond_region(wavelength_nm)
            rows.append(
                {
                    "nutrient": nutrient,
                    "wavelength_nm": wavelength_nm,
                    "correlation": correlation,
                    "abs_correlation": abs(correlation) if pd.notna(correlation) else np.nan,
                    "assigned_bond": assigned_bond,
                    "interpretation": interpretation,
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    if association_df is not None and not association_df.empty:
        association_subset = association_df[
            ["wavelength_nm", "component_1_weight", "component_2_weight", "mean_abs_correlation"]
        ].drop_duplicates(subset=["wavelength_nm"])
        table = table.merge(association_subset, on="wavelength_nm", how="left")

    if vip_scores is not None and not vip_scores.empty:
        vip_subset = vip_scores[["wavelength_nm", "vip_score"]].drop_duplicates(subset=["wavelength_nm"])
        table = table.merge(vip_subset, on="wavelength_nm", how="left")

    return table.sort_values(["nutrient", "wavelength_nm"]).reset_index(drop=True)


def assign_bond_region(wavelength_nm: float) -> Tuple[str, str]:
    """Map wavelength regions to broad vibrational bond assignments."""

    wavelength = float(wavelength_nm)
    if 950.0 <= wavelength <= 990.0:
        return "O-H", "Water-dominated overtone region linked to hydration and bound moisture."
    if 1150.0 <= wavelength <= 1250.0:
        return "C-H", "Carbohydrate- and lipid-related overtone region."
    if 1375.0 <= wavelength <= 1490.0:
        return "O-H / matrix water", "Strong hydroxyl and matrix-water combination region."
    if 1490.0 <= wavelength <= 1570.0:
        return "N-H / amide", "Protein-associated amide and hydrogen-bonded absorptions."
    if 1670.0 <= wavelength <= 1760.0:
        return "C-H", "Lipid and carbohydrate combination-band region."
    return "Mixed overtone", "Composite plant-matrix overtone region with overlapping assignments."
