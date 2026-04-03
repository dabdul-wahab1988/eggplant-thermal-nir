"""Data ingestion and harmonization utilities."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Mapping, Tuple, Union

import pandas as pd


PathLike = Union[str, Path]

SPECIES_MAP = {
    "A": "Solanum aethiopicum",
    "M": "Solanum melongena",
    "T": "Solanum torvum",
}

PROCESS_MAP = {
    "R": "raw",
    "B": "boiled",
    "C": "blanched",
}

WORKBOOK_VALUE_COLUMNS = {
    "moisture": ("%Moisture", "%"),
    "fat": ("%fat (wet base)", "%"),
    "fibre": ("%fibre (wet base)", "%"),
    "protein": ("%PROTEIN", "%"),
    "carb": ("%CARBOHYDRATE", "%"),
    "Vitamin C": ("conc(mg/100ml)", "mg/100ml"),
}

ARCHIVE_NAMES = (
    "20220630_EGGPLANT_ODILIA",
    "20220706_eggplant_Odilia2",
    "20220707_EGGPLANT_ODILIA3",
)


def _normalize_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def _sample_code_from_parts(species_code: str, process: str, replicate: int) -> str:
    if process == "raw":
        return "R{0}{1}".format(species_code, replicate)
    if process == "boiled":
        return "B{0}{1}".format(species_code, replicate)
    if process == "blanched":
        return "BL{0}{1}".format(species_code, replicate)
    raise ValueError("Unsupported process: {0}".format(process))


def _analysis_sample_id(species_code: str, process: str, replicate: int) -> str:
    return "{0}_{1}_{2}".format(species_code, process, int(replicate))


def parse_workbook_sample_code(sample_code: str) -> Dict[str, object]:
    """Parse workbook sample IDs such as RM1, BM2, and BLA3."""

    text = str(sample_code).strip().upper()
    if text.startswith("BL"):
        process = "blanched"
        species_code = text[2]
        replicate = int(text[3:])
    elif text.startswith("B"):
        process = "boiled"
        species_code = text[1]
        replicate = int(text[2:])
    elif text.startswith("R"):
        process = "raw"
        species_code = text[1]
        replicate = int(text[2:])
    else:
        raise ValueError("Unsupported workbook sample code: {0}".format(sample_code))
    return {
        "sample_code": text,
        "species_code": species_code,
        "species": SPECIES_MAP[species_code],
        "process": process,
        "replicate": replicate,
        "analysis_sample_id": _analysis_sample_id(species_code, process, replicate),
    }


def parse_spectral_filename(file_path: PathLike) -> Dict[str, object]:
    """Parse spectral file names into harmonized metadata."""

    path = Path(file_path)
    stem = path.stem
    prefix = stem.split("Column")[0].strip()
    pattern = re.compile(r"^(?P<class_code>[A-Z]{3})(?P<replicate>\d{2})R(?P<spot>\d+)$")
    match = pattern.match(prefix)
    if match is None:
        raise ValueError("Unsupported spectral filename format: {0}".format(stem))

    class_code = match.group("class_code")
    species_code = class_code[0]
    process_code = class_code[1]
    session_code = class_code[2]
    replicate = int(match.group("replicate"))
    spot = int(match.group("spot"))
    process = PROCESS_MAP[process_code]
    analysis_sample_id = _analysis_sample_id(species_code, process, replicate)

    repeat_match = re.search(r"\((\d+)\)$", stem)
    file_repeat = int(repeat_match.group(1)) if repeat_match else 0

    return {
        "scan_id": stem,
        "class_code": class_code,
        "species_code": species_code,
        "species": SPECIES_MAP[species_code],
        "process_code": process_code,
        "process": process,
        "session_code": session_code,
        "replicate": replicate,
        "spot": spot,
        "file_repeat": file_repeat,
        "analysis_sample_id": analysis_sample_id,
        "sample_code": _sample_code_from_parts(species_code, process, replicate),
        "sample_unit_id": "{0}_spot{1}_session{2}".format(analysis_sample_id, spot, session_code),
        "file_path": str(path.resolve()),
        "archive_name": path.parent.name,
    }


def _standardize_rows(
    frame: pd.DataFrame,
    sample_col: str,
    value_col: str,
    nutrient: str,
    unit: str,
) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned = cleaned.loc[cleaned[sample_col].notna()].copy()
    cleaned["sample_code"] = cleaned[sample_col].astype(str).str.strip().str.upper()
    cleaned["value"] = pd.to_numeric(cleaned[value_col], errors="coerce")
    cleaned = cleaned.loc[cleaned["sample_code"].ne("")].copy()
    meta = cleaned["sample_code"].map(parse_workbook_sample_code).apply(pd.Series)
    standardized = pd.concat(
        [
            cleaned.reset_index(drop=True),
            meta.reset_index(drop=True)[["species_code", "species", "process", "replicate", "analysis_sample_id"]],
        ],
        axis=1,
    )
    standardized["nutrient"] = nutrient
    standardized["unit"] = unit
    standardized["source_sheet"] = nutrient
    return standardized


def load_nutrient_workbook(path: PathLike) -> Dict[str, pd.DataFrame]:
    """Load and standardize nutrient sheets from the workbook."""

    workbook_path = _normalize_path(path)
    nutrient_tables: Dict[str, pd.DataFrame] = {}

    for sheet_name, (value_col, unit) in WORKBOOK_VALUE_COLUMNS.items():
        raw = pd.read_excel(workbook_path, sheet_name=sheet_name)
        if sheet_name in ("moisture", "fat"):
            sample_col = "samples"
        elif sheet_name == "fibre":
            sample_col = "sample"
        elif sheet_name in ("protein", "carb"):
            sample_col = raw.columns[0]
        else:
            sample_col = "samples"
        nutrient_name = sheet_name.lower().replace(" ", "_")
        nutrient_tables[nutrient_name] = _standardize_rows(raw, sample_col, value_col, nutrient_name, unit)

    ash_raw = pd.read_excel(workbook_path, sheet_name="ash")
    ash_rows = ash_raw.iloc[1:].copy()
    ash_rows.columns = ["sample_code", "weight_crucible", "sample_weight", "post_ashing_weight", "value"]
    nutrient_tables["ash"] = _standardize_rows(ash_rows, "sample_code", "value", "ash", "%")

    return nutrient_tables


def combine_nutrient_tables(nutrient_tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine standardized nutrient tables into one tidy long dataset."""

    tidy = pd.concat([table.copy() for table in nutrient_tables.values()], ignore_index=True)
    tidy["replicate"] = tidy["replicate"].astype(int)
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy = tidy.sort_values(["species_code", "process", "replicate", "nutrient"]).reset_index(drop=True)
    return tidy


def pivot_nutrients_wide(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """Create a wide nutrient table indexed by analysis sample ID."""

    wide = tidy_df.pivot_table(
        index="analysis_sample_id",
        columns="nutrient",
        values="value",
        aggfunc="mean",
    )
    wide = wide.sort_index()
    wide.columns.name = None
    return wide


def _parse_spectral_file(path: Path) -> Tuple[Dict[str, object], Dict[str, float]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = None
    for index, line in enumerate(lines):
        if line.startswith("Wavelength (nm),Absorbance (AU)"):
            header_index = index
            break
    if header_index is None:
        raise ValueError("Wavelength header not found in {0}".format(path))

    meta_rows = list(csv.reader(lines[:header_index]))
    header_meta: Dict[str, object] = {}
    for row in meta_rows:
        if not row:
            continue
        key = row[0].strip().rstrip(":")
        values = [value.strip() for value in row[1:] if value.strip()]
        if not key:
            continue
        if not values:
            header_meta[key] = ""
        elif len(values) == 1:
            header_meta[key] = values[0]
        else:
            header_meta[key] = values

    data = pd.read_csv(path, skiprows=header_index)
    data.columns = [column.strip() for column in data.columns]
    data = data[["Wavelength (nm)", "Absorbance (AU)", "Reference Signal (unitless)", "Sample Signal (unitless)"]]
    wavelengths = data["Wavelength (nm)"].astype(float)
    absorbance = data["Absorbance (AU)"].astype(float)

    spectral_row = {"{0:.6f}".format(wl): value for wl, value in zip(wavelengths, absorbance)}
    spectral_row["scan_id"] = path.stem

    parsed = parse_spectral_filename(path)
    meta = dict(parsed)
    host_datetime_raw = header_meta.get("Host Date-Time", "")
    meta["host_datetime_raw"] = host_datetime_raw
    meta["host_datetime"] = pd.to_datetime(host_datetime_raw, dayfirst=True, errors="coerce")
    meta["system_temp_c"] = pd.to_numeric(header_meta.get("System Temp (C)", None), errors="coerce")
    meta["detector_temp_c"] = pd.to_numeric(header_meta.get("Detector Temp (C)", None), errors="coerce")
    meta["humidity_pct"] = pd.to_numeric(header_meta.get("Humidity (%)", None), errors="coerce")
    meta["lamp_pd"] = pd.to_numeric(header_meta.get("Lamp PD", None), errors="coerce")
    meta["digital_resolution"] = pd.to_numeric(header_meta.get("Digital Resolution", None), errors="coerce")
    meta["n_wavelengths"] = int(len(data))

    return meta, spectral_row


def load_spectral_archives(root: PathLike) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load spectral metadata and absorbance matrix from the project archive folders."""

    root_path = _normalize_path(root)
    csv_files: List[Path] = []
    for archive_name in ARCHIVE_NAMES:
        archive_dir = root_path / archive_name
        csv_files.extend(sorted(archive_dir.glob("*.csv")))

    metadata_rows: List[Dict[str, object]] = []
    spectral_rows: List[Dict[str, float]] = []
    for csv_file in csv_files:
        meta_row, spectral_row = _parse_spectral_file(csv_file)
        metadata_rows.append(meta_row)
        spectral_rows.append(spectral_row)

    spectral_meta = pd.DataFrame(metadata_rows).sort_values(
        ["species_code", "process", "replicate", "session_code", "spot", "file_repeat", "scan_id"]
    )
    spectral_meta = spectral_meta.set_index("scan_id", drop=False)

    spectra = pd.DataFrame(spectral_rows).set_index("scan_id", drop=True)
    spectra = spectra.loc[spectral_meta.index]
    ordered_columns = sorted(spectra.columns, key=float)
    spectra = spectra[ordered_columns].astype(float)
    spectra = spectra.interpolate(axis=1, limit_direction="both")
    spectra = spectra.ffill(axis=1).bfill(axis=1)

    return spectral_meta, spectra


def build_sample_metadata(
    nutrient_tables: Mapping[str, pd.DataFrame], spectral_meta: pd.DataFrame
) -> pd.DataFrame:
    """Build a harmonized sample table linking chemistry and spectral availability."""

    chemistry = combine_nutrient_tables(nutrient_tables)
    chemistry_meta = (
        chemistry.groupby("analysis_sample_id")
        .agg(
            sample_code=("sample_code", "first"),
            species_code=("species_code", "first"),
            species=("species", "first"),
            process=("process", "first"),
            replicate=("replicate", "first"),
            nutrient_count=("nutrient", "nunique"),
        )
        .reset_index()
    )

    spectral_summary = (
        spectral_meta.groupby("analysis_sample_id")
        .agg(
            total_scans=("scan_id", "size"),
            unique_sessions=("session_code", "nunique"),
            unique_spots=("spot", "nunique"),
            first_scan_datetime=("host_datetime", "min"),
            last_scan_datetime=("host_datetime", "max"),
        )
        .reset_index()
    )

    sample_metadata = chemistry_meta.merge(spectral_summary, on="analysis_sample_id", how="left")
    sample_metadata["total_scans"] = sample_metadata["total_scans"].fillna(0).astype(int)
    sample_metadata["unique_sessions"] = sample_metadata["unique_sessions"].fillna(0).astype(int)
    sample_metadata["unique_spots"] = sample_metadata["unique_spots"].fillna(0).astype(int)
    sample_metadata = sample_metadata.sort_values(["species_code", "process", "replicate"]).reset_index(drop=True)
    return sample_metadata
