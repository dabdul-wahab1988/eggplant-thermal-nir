# eggplant-thermal-nir

> **Species-specific nutrient retention and near-infrared thermal fingerprints of cultivated and underutilised eggplants under boiling and blanching**

![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![Version](https://img.shields.io/badge/version-0.1.0-informational)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-beta-yellow)

---

## Overview

`eggplant-thermal-nir` is the reproducible analysis package for a factorial food-science study comparing nutrient composition and near-infrared (NIR) spectral signatures of three eggplant species — *Solanum melongena* (cultivated), *Solanum aethiopicum*, and *Solanum torvum* (underutilised African species) — under two domestic thermal processing conditions (boiling and blanching).

The package ingests raw wet-chemistry and NIR spectral data, runs the full analytical pipeline, and produces all manuscript tables and figures in a single command.

### What the study addresses

| Research gap | How this package addresses it |
|---|---|
| No standardised cross-species nutrient-retention comparison | 3 × 3 factorial design: 3 species × 3 process states, common analytical panel |
| NIR rarely used on cooked eggplant | 487 scans of raw, boiled, and blanched tissue at 901–1700 nm |
| Scan-level leakage in food NIR classification | Grouped cross-validation locked to biological-sample level |
| Nutrient data and spectra rarely integrated | Multiblock PLS integration linking proximate fractions to spectral regions |

---

## Package Structure

```
src/eggplant_thermal_nir/
├── __init__.py          # Public API: AnalysisConfig, run_analysis, generate_artifacts
├── _version.py          # Version string (0.1.0)
├── config.py            # AnalysisConfig dataclass — all runtime parameters
├── pipeline.py          # End-to-end orchestration: run_analysis() / generate_artifacts()
├── data.py              # Workbook ingestion, NIR archive parsing, sample harmonisation
├── spectra.py           # Spectral preprocessing (SNV, MSC, SG derivatives, detrending)
├── statistics.py        # Factorial ANOVA, nutrient summaries, thermal resilience index
├── chemometrics.py      # PCA, ASCA/ANOVA-PCA, grouped classification, permutation tests
├── integration.py       # Multiblock PLS nutrient–spectrum integration
├── plotting.py          # All manuscript figures (Figures 1–5, Supplementary S1–S12)
├── cli.py               # Command-line entry point
├── logging_utils.py     # JSONL run-log and stage-timer utilities
└── py.typed             # PEP 561 marker
```

---

## Dataset at a Glance

| Parameter | Value |
|---|---|
| Factorial design | 3 species × 3 process states (raw / boiled / blanched) |
| Biological replicates per cell | 3 |
| Total chemistry samples | 27 |
| Nutrients analysed | Ash, Carbohydrate, Fat, Fibre, Moisture, Protein, Vitamin C |
| Total NIR scans | 487 |
| Spectral range | 901.2 – 1700.6 nm (228 wavelengths) |
| Mean scans per sample | 18.04 |
| Acquisition sessions | 2 (P, S, T session codes) |
| Random seed (all stochastic steps) | 42 |

---

## Installation

**From this repository (recommended for reproduction):**

```bash
git clone https://github.com/dabdul-wahab1988/eggplant-thermal-nir.git
cd eggplant-thermal-nir
pip install -e ".[dev]"
```

> **Note:** The `pyproject.toml` (available in the manuscript project root) declares `src`-layout packaging. If you only have this repository, you can install the package source directly with:
> ```bash
> pip install -e .
> ```
> after adding a minimal `pyproject.toml` — or import from `src/` by setting `PYTHONPATH=src`.

**Python version requirement:** `>=3.9, <4.0`

**Core dependencies:**

| Package | Minimum version |
|---|---|
| pandas | 2.0 |
| numpy | 1.24 |
| matplotlib | 3.7 |
| scikit-learn | 1.3 |
| statsmodels | 0.14 |
| scipy | 1.10 |

---

## Quick Start

### 1 — Run the full pipeline

```bash
# Using the CLI entry point
eggplant-analysis-pipeline --project-root /path/to/project run

# Or via the module
python -m eggplant_thermal_nir.cli --project-root /path/to/project run
```

### 2 — Use the Python API

```python
from eggplant_thermal_nir import AnalysisConfig, generate_artifacts

cfg = AnalysisConfig(project_root="/path/to/project")
results = generate_artifacts(cfg)
```

### 3 — Customise configuration

```python
from eggplant_thermal_nir import AnalysisConfig, run_analysis

cfg = AnalysisConfig(
    project_root="/path/to/project",
    primary_preprocess="sg1_snv",          # default preprocessing for exploration/integration
    classification_preprocesses=("raw", "snv", "sg1_snv"),
    classification_models=("plsda", "svm", "rf"),
    classification_targets=("species", "process", "species_process"),
    n_splits=3,                             # grouped CV folds
    random_state=42,
)
results = run_analysis(cfg)
```

---

## Configuration Reference

All parameters live in `AnalysisConfig` (`config.py`):

| Parameter | Default | Description |
|---|---|---|
| `project_root` | `Path(".")` | Root directory containing the workbook and NIR archives |
| `workbook_name` | `"RESULT FOR PROJECT.xlsx"` | Wet-chemistry workbook filename |
| `spectral_archive_names` | Three dated archive folders | NIR archive subdirectory names |
| `primary_preprocess` | `"sg1_snv"` | Preprocessing used for PCA, ASCA, and integration |
| `classification_preprocesses` | `("raw", "snv", "sg1_snv")` | Preprocessing pipelines benchmarked for classification |
| `classification_models` | `("plsda", "svm", "rf")` | Model families compared |
| `classification_targets` | `("species", "process", "species_process")` | Classification task labels |
| `n_splits` | `3` | Number of grouped CV folds |
| `pca_components` | `3` | PCA components retained |
| `random_state` | `42` | Fixed seed for all stochastic operations |
| `output_dir` | `manuscript/artifacts/` | Artifact output directory |
| `logs_dir` | `logs/` | JSONL log output directory |

---
## Analytical Methods Summary

### Preprocessing pipelines (`spectra.py`)

Nine options available: `raw`, `snv`, `msc`, `detrend`, `sg1`, `sg2`, `sg_snv`, `sg1_snv`, `sg2_snv`.
Default for exploration and integration: **`sg1_snv`** (Savitzky–Golay first derivative + SNV).
For classification, all three candidate pipelines are benchmarked inside grouped CV — preprocessing is not selected before validation.

### Factorial statistics (`statistics.py`)

Two-way fixed-effects ANOVA (Type II SS) per nutrient:

$$y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}$$

Effect size reported as η² = SS_effect / SS_total.

Thermal resilience index per species-by-process combination:

$$c = \max\!\left(0,\ 1 - \frac{|R - 100|}{100}\right), \quad R = 100 \times \frac{x_\text{proc}}{x_\text{raw}}$$

### Chemometrics (`chemometrics.py`)

- **PCA** on column-standardised preprocessed spectra (3 components retained)
- **ANOVA-linked PCA** — PC scores regressed on species, process, and interaction to decompose structured spectral variance
- **Grouped classification** — PLS-DA, SVM (RBF, balanced), Random Forest (balanced subsample) with `StratifiedGroupKFold` splits at biological-sample level
- **Leakage quantification** — naive `StratifiedKFold` baseline run in parallel for each best grouped model
- **Permutation testing** — 24 label permutations per target
- **Hyperparameter tuning** inside grouped CV (SVM: C, gamma; RF: n_estimators, max_depth, min_samples_leaf; PLS: n_components)

### Multiblock integration (`integration.py`)

PLS regression with up to 2 latent components on sample-mean spectra vs. standardised nutrient matrix:

$$X = TP^\top + E, \qquad Y = UQ^\top + F$$

Top-3 wavelengths per nutrient merged into 12 nm intervals and annotated to functional-group bond regions (O–H, C–H overtone / combination bands).

---

## Reproducibility

| Item | Value |
|---|---|
| Package version | 0.1.0 |
| Python | ≥ 3.9, < 4.0 (tested on 3.13.5) |
| NumPy | 2.1.3 |
| Pandas | 2.2.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.6.1 |
| statsmodels | 0.14.4 |
| Global random seed | 42 |
| Cache | `manuscript/cache/analysis_results.pkl` (auto-generated) |
| Run logs | `logs/run_*.jsonl`, `logs/statistics_*.jsonl` |

---

## Citation

If you use this package or the data pipeline in your work, please cite the associated manuscript:

> Abdul-Wahab, D. *et al.* (in preparation). *Species-specific nutrient retention and near-infrared thermal fingerprints of cultivated and underutilised eggplants under boiling and blanching.*

---

## License

This project is licensed under the **MIT License**.
© Dickson Abdul-Wahab. See [LICENSE](LICENSE) for details.
