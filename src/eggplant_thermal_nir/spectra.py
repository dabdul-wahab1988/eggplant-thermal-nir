"""Spectral preprocessing utilities."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.signal import detrend as scipy_detrend
from scipy.signal import savgol_filter


def get_wavelength_columns(frame: pd.DataFrame) -> List[str]:
    """Return columns that represent wavelengths, sorted numerically."""

    columns = []
    for column in frame.columns:
        try:
            float(column)
        except (TypeError, ValueError):
            continue
        columns.append(str(column))
    return sorted(columns, key=float)


def _window_length(n_columns: int, desired: int = 11) -> int:
    window = min(desired, n_columns if n_columns % 2 == 1 else n_columns - 1)
    if window < 5:
        window = 5 if n_columns >= 5 else n_columns
    if window % 2 == 0:
        window -= 1
    return max(window, 3)


def _msc(values: np.ndarray) -> np.ndarray:
    reference = values.mean(axis=0)
    corrected = np.zeros_like(values)
    for row_index, row in enumerate(values):
        slope, intercept = np.polyfit(reference, row, 1)
        if slope == 0:
            corrected[row_index, :] = row - intercept
        else:
            corrected[row_index, :] = (row - intercept) / slope
    return corrected


def _snv(values: np.ndarray) -> np.ndarray:
    row_means = values.mean(axis=1, keepdims=True)
    row_stds = values.std(axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1.0
    return (values - row_means) / row_stds


def preprocess_spectra(spectra: pd.DataFrame, method: str, **kwargs) -> pd.DataFrame:
    """Apply a named preprocessing method to the spectral matrix."""

    method_key = method.lower().strip()
    wavelength_columns = get_wavelength_columns(spectra)
    values = spectra[wavelength_columns].to_numpy(dtype=float)
    window = _window_length(values.shape[1], desired=int(kwargs.get("window_length", 11)))
    polyorder = min(int(kwargs.get("polyorder", 2)), window - 1)

    if method_key == "raw":
        transformed = values.copy()
    elif method_key == "snv":
        transformed = _snv(values)
    elif method_key == "msc":
        transformed = _msc(values)
    elif method_key == "detrend":
        transformed = scipy_detrend(values, axis=1, type="linear")
    elif method_key == "first_derivative":
        transformed = savgol_filter(values, window_length=window, polyorder=polyorder, deriv=1, axis=1)
    elif method_key == "second_derivative":
        transformed = savgol_filter(values, window_length=window, polyorder=polyorder, deriv=2, axis=1)
    elif method_key == "sg_snv":
        smoothed = savgol_filter(values, window_length=window, polyorder=polyorder, deriv=0, axis=1)
        transformed = _snv(smoothed)
    elif method_key == "sg1_snv":
        transformed = savgol_filter(values, window_length=window, polyorder=polyorder, deriv=1, axis=1)
        transformed = _snv(transformed)
    elif method_key == "sg2_snv":
        transformed = savgol_filter(values, window_length=window, polyorder=polyorder, deriv=2, axis=1)
        transformed = _snv(transformed)
    else:
        raise ValueError("Unsupported preprocessing method: {0}".format(method))

    return pd.DataFrame(transformed, index=spectra.index, columns=wavelength_columns)

