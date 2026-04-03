"""Chemometric analysis helpers."""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, balanced_accuracy_score, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import SVC

from eggplant_thermal_nir.spectra import preprocess_spectra

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


def run_pca(spectra: pd.DataFrame, n_components: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run PCA on the spectral matrix."""

    scaler = StandardScaler()
    transformed = scaler.fit_transform(spectra.to_numpy(dtype=float))
    n_components = max(2, min(int(n_components), transformed.shape[0], transformed.shape[1]))
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(transformed)
    columns = ["PC{0}".format(index + 1) for index in range(scores.shape[1])]
    score_frame = pd.DataFrame(scores, index=spectra.index, columns=columns)
    loading_frame = pd.DataFrame(pca.components_.T, index=spectra.columns, columns=columns)
    return score_frame, loading_frame


def summarize_factorial_spectral_effects(meta: pd.DataFrame, pca_scores: pd.DataFrame) -> pd.DataFrame:
    """Use PCA scores as a design-aware summary of spectral factorial effects."""

    joined = meta[["species", "process"]].join(pca_scores, how="inner")
    frames: List[pd.DataFrame] = []
    for column in pca_scores.columns:
        model = smf.ols("{0} ~ C(species) * C(process)".format(column), data=joined).fit()
        anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "effect"})
        anova["component"] = column
        frames.append(anova)
    summary = pd.concat(frames, ignore_index=True)
    return summary[["component", "effect", "df", "sum_sq", "F", "PR(>F)"]]


def build_grouped_cv_splits(
    meta: pd.DataFrame,
    target: Optional[str] = None,
    n_splits: int = 3,
    random_state: int = 42,
    grouped: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Construct grouped cross-validation splits that keep scans from the same sample together."""

    groups = meta["analysis_sample_id"].astype(str).to_numpy()
    n_unique = len(np.unique(groups))
    n_splits = max(2, min(int(n_splits), n_unique))

    if target is None:
        y = meta["process"].astype(str).to_numpy()
    elif target == "species_process":
        y = (meta["species"].astype(str) + " | " + meta["process"].astype(str)).to_numpy()
    else:
        y = meta[target].astype(str).to_numpy()

    if grouped:
        if StratifiedGroupKFold is not None:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = splitter.split(np.zeros(len(meta)), y, groups)
        else:  # pragma: no cover
            splitter = GroupKFold(n_splits=n_splits)
            splits = splitter.split(np.zeros(len(meta)), y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(np.zeros(len(meta)), y)
    return [(train_index, test_index) for train_index, test_index in splits]


def _target_series(meta: pd.DataFrame, target: str) -> pd.Series:
    if target == "species_process":
        return meta["species"].astype(str) + " | " + meta["process"].astype(str)
    return meta[target].astype(str)


def _fit_predict_plsda(
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_test: np.ndarray,
    n_components: Optional[int] = None,
) -> np.ndarray:
    label_binarizer = LabelBinarizer()
    y_binary = label_binarizer.fit_transform(y_train)
    if y_binary.ndim == 1:
        y_binary = y_binary.reshape(-1, 1)
    if n_components is None:
        n_components = max(1, min(6, x_train.shape[0] - 1, x_train.shape[1], y_binary.shape[1] or 1))
    else:
        n_components = max(1, min(int(n_components), x_train.shape[0] - 1, x_train.shape[1], 6))
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=n_components)),
        ]
    )
    model.fit(x_train, y_binary)
    predictions = model.predict(x_test)
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        predictions = np.column_stack([1.0 - predictions.ravel(), predictions.ravel()])
        classes = label_binarizer.classes_
        if len(classes) == 1:
            return np.repeat(classes[0], x_test.shape[0])
        return np.array([classes[int(score[1] >= score[0])] for score in predictions])
    return np.array([label_binarizer.classes_[index] for index in predictions.argmax(axis=1)])


def _fit_predict_classifier(
    model_name: str,
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_test: np.ndarray,
    random_state: int,
    hyperparams: Optional[Mapping[str, object]] = None,
) -> np.ndarray:
    hyperparams = dict(hyperparams or {})
    if model_name == "plsda":
        return _fit_predict_plsda(x_train, y_train, x_test, n_components=hyperparams.get("n_components"))
    if model_name == "svm":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel=str(hyperparams.get("kernel", "rbf")),
                        class_weight="balanced",
                        gamma=hyperparams.get("gamma", "scale"),
                        C=float(hyperparams.get("C", 1.0)),
                        probability=bool(hyperparams.get("probability", False)),
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=int(hyperparams.get("n_estimators", 300)),
            max_depth=hyperparams.get("max_depth"),
            min_samples_leaf=int(hyperparams.get("min_samples_leaf", 1)),
            random_state=random_state,
            class_weight="balanced_subsample",
        )
    else:
        raise ValueError("Unsupported model: {0}".format(model_name))
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test))


def _evaluate_predictions(
    processed: pd.DataFrame,
    meta: pd.DataFrame,
    target: str,
    model_name: str,
    random_state: int,
    n_splits: int,
    grouped: bool,
    y_override: Optional[pd.Series] = None,
    hyperparams: Optional[Mapping[str, object]] = None,
) -> Tuple[List[str], List[str], List[Dict[str, object]]]:
    target_values = y_override.copy() if y_override is not None else _target_series(meta, target)
    observed: List[str] = []
    predicted: List[str] = []
    fold_metrics: List[Dict[str, object]] = []
    fold_no = 0
    for train_index, test_index in build_grouped_cv_splits(
        meta,
        target=target,
        n_splits=n_splits,
        random_state=random_state,
        grouped=grouped,
    ):
        x_train = processed.iloc[train_index].to_numpy(dtype=float)
        x_test = processed.iloc[test_index].to_numpy(dtype=float)
        y_train = target_values.iloc[train_index].tolist()
        y_test = target_values.iloc[test_index].tolist()
        y_pred = _fit_predict_classifier(
            model_name,
            x_train,
            y_train,
            x_test,
            random_state,
            hyperparams=hyperparams,
        )
        observed.extend(y_test)
        predicted.extend(y_pred.tolist())
        fold_no += 1
        fold_metrics.append(
            {
                "fold": fold_no,
                "n_test": len(y_test),
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro"),
            }
        )
    return observed, predicted, fold_metrics


def evaluate_classifiers(
    spectra: pd.DataFrame,
    meta: pd.DataFrame,
    target: str,
    random_state: int = 42,
    n_splits: int = 3,
    preprocess_methods: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    split_strategy: str = "grouped",
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    """Evaluate classification models under grouped or naive validation."""

    preprocess_methods = tuple(preprocess_methods or ("raw", "snv", "sg1_snv"))
    models = tuple(models or ("plsda", "svm", "rf"))
    grouped = split_strategy == "grouped"
    target_values = _target_series(meta, target)
    labels = np.array(sorted(target_values.unique().tolist()))

    rows: List[Dict[str, object]] = []
    confusion_tables: Dict[str, pd.DataFrame] = {}
    fold_rows: List[Dict[str, object]] = []

    for preprocess_name in preprocess_methods:
        processed = preprocess_spectra(spectra, preprocess_name)
        for model_name in models:
            observed, predicted, fold_metrics = _evaluate_predictions(
                processed=processed,
                meta=meta,
                target=target,
                model_name=model_name,
                random_state=random_state,
                n_splits=n_splits,
                grouped=grouped,
            )

            key = "{0}::{1}::{2}::{3}".format(target, split_strategy, preprocess_name, model_name)
            cm = pd.DataFrame(confusion_matrix(observed, predicted, labels=labels), index=labels, columns=labels)
            confusion_tables[key] = cm
            rows.append(
                {
                    "target": target,
                    "split_strategy": split_strategy,
                    "preprocessing": preprocess_name,
                    "model": model_name,
                    "accuracy": accuracy_score(observed, predicted),
                    "balanced_accuracy": balanced_accuracy_score(observed, predicted),
                    "macro_f1": f1_score(observed, predicted, average="macro"),
                    "n_labels": len(labels),
                    "n_groups": meta["analysis_sample_id"].nunique(),
                    "n_scans": processed.shape[0],
                    "result_key": key,
                }
            )
            for fold_metric in fold_metrics:
                fold_rows.append(
                    {
                        "target": target,
                        "split_strategy": split_strategy,
                        "preprocessing": preprocess_name,
                        "model": model_name,
                        **fold_metric,
                    }
                )

    summary = pd.DataFrame(rows).sort_values(
        ["target", "split_strategy", "balanced_accuracy", "macro_f1"],
        ascending=[True, True, False, False],
    )
    fold_df = pd.DataFrame(fold_rows).sort_values(
        ["target", "split_strategy", "preprocessing", "model", "fold"]
    )
    return summary.reset_index(drop=True), confusion_tables, fold_df.reset_index(drop=True)


def evaluate_grouped_classifiers(
    spectra: pd.DataFrame,
    meta: pd.DataFrame,
    target: str,
    random_state: int = 42,
    n_splits: int = 3,
    preprocess_methods: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    """Evaluate grouped-validation classification models."""
    return evaluate_classifiers(
        spectra=spectra,
        meta=meta,
        target=target,
        random_state=random_state,
        n_splits=n_splits,
        preprocess_methods=preprocess_methods,
        models=models,
        split_strategy="grouped",
    )


def evaluate_leakage_baseline(
    spectra: pd.DataFrame,
    meta: pd.DataFrame,
    grouped_summary: pd.DataFrame,
    random_state: int = 42,
    n_splits: int = 3,
) -> pd.DataFrame:
    """Compare grouped validation results against naive scan-level validation."""

    if grouped_summary.empty:
        return pd.DataFrame(
            columns=[
                "target",
                "model",
                "preprocessing",
                "naive_accuracy",
                "naive_balanced_accuracy",
                "naive_macro_f1",
            ]
        )

    rows: List[Dict[str, object]] = []
    best_rows = (
        grouped_summary.loc[grouped_summary["split_strategy"] == "grouped"]
        .sort_values(["target", "balanced_accuracy", "macro_f1"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .first()
    )
    for row in best_rows.itertuples(index=False):
        naive_summary, _, _ = evaluate_classifiers(
            spectra=spectra,
            meta=meta,
            target=row.target,
            random_state=random_state,
            n_splits=n_splits,
            preprocess_methods=(row.preprocessing,),
            models=(row.model,),
            split_strategy="naive",
        )
        if naive_summary.empty:
            continue
        naive_row = naive_summary.iloc[0]
        rows.append(
            {
                "target": row.target,
                "model": row.model,
                "preprocessing": row.preprocessing,
                "naive_accuracy": naive_row["accuracy"],
                "naive_balanced_accuracy": naive_row["balanced_accuracy"],
                "naive_macro_f1": naive_row["macro_f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def build_supported_preprocessing_table() -> pd.DataFrame:
    """Describe spectral preprocessing methods available in the package."""

    return pd.DataFrame(
        [
            {"method": "raw", "category": "baseline", "description": "No spectral preprocessing."},
            {"method": "snv", "category": "scatter correction", "description": "Standard normal variate normalization."},
            {"method": "msc", "category": "scatter correction", "description": "Multiplicative scatter correction against the mean reference."},
            {"method": "detrend", "category": "baseline correction", "description": "Linear detrending per scan."},
            {"method": "first_derivative", "category": "derivative", "description": "Savitzky-Golay first derivative."},
            {"method": "second_derivative", "category": "derivative", "description": "Savitzky-Golay second derivative."},
            {"method": "sg_snv", "category": "smoothed normalization", "description": "Savitzky-Golay smoothing followed by SNV."},
            {"method": "sg1_snv", "category": "smoothed derivative", "description": "Savitzky-Golay first derivative followed by SNV."},
            {"method": "sg2_snv", "category": "smoothed derivative", "description": "Savitzky-Golay second derivative followed by SNV."},
        ]
    )


def run_hyperparameter_grid_search(
    spectra: pd.DataFrame,
    meta: pd.DataFrame,
    target: str,
    preprocess_name: str,
    model_name: str,
    param_grid: Mapping[str, Sequence[object]],
    random_state: int = 42,
    n_splits: int = 3,
) -> pd.DataFrame:
    """Run a compact grouped-CV hyperparameter search for one task/model."""

    processed = preprocess_spectra(spectra, preprocess_name)
    keys = list(param_grid.keys())
    rows: List[Dict[str, object]] = []
    for values in product(*(param_grid[key] for key in keys)):
        params = dict(zip(keys, values))
        _, _, fold_metrics = _evaluate_predictions(
            processed=processed,
            meta=meta,
            target=target,
            model_name=model_name,
            random_state=random_state,
            n_splits=n_splits,
            grouped=True,
            hyperparams=params,
        )
        fold_df = pd.DataFrame(fold_metrics)
        row = {
            "target": target,
            "preprocessing": preprocess_name,
            "model": model_name,
            "mean_accuracy": fold_df["accuracy"].mean(),
            "mean_balanced_accuracy": fold_df["balanced_accuracy"].mean(),
            "mean_macro_f1": fold_df["macro_f1"].mean(),
        }
        for key, value in params.items():
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_balanced_accuracy", ascending=False).reset_index(drop=True)


def compute_grouped_roc_curves(
    spectra: pd.DataFrame,
    meta: pd.DataFrame,
    target: str,
    preprocess_name: str,
    model_name: str,
    random_state: int = 42,
    n_splits: int = 3,
) -> pd.DataFrame:
    """Compute grouped cross-validated one-vs-rest ROC curves for a selected classifier."""

    processed = preprocess_spectra(spectra, preprocess_name)
    target_values = _target_series(meta, target)
    classes = np.array(sorted(target_values.unique().tolist()))
    observed_labels: List[str] = []
    score_blocks: List[np.ndarray] = []

    for train_index, test_index in build_grouped_cv_splits(
        meta,
        target=target,
        n_splits=n_splits,
        random_state=random_state,
        grouped=True,
    ):
        x_train = processed.iloc[train_index].to_numpy(dtype=float)
        x_test = processed.iloc[test_index].to_numpy(dtype=float)
        y_train = target_values.iloc[train_index].tolist()
        y_test = target_values.iloc[test_index].tolist()
        _, scores, model_classes = _fit_predict_with_scores(
            model_name,
            x_train,
            y_train,
            x_test,
            random_state,
        )
        aligned_scores = _align_score_matrix(scores, model_classes, classes)
        observed_labels.extend(y_test)
        score_blocks.append(aligned_scores)

    score_matrix = np.vstack(score_blocks)
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(classes)
    y_true = label_binarizer.transform(observed_labels)
    if y_true.ndim == 1:
        y_true = np.column_stack([1 - y_true, y_true])
        class_labels = classes
    else:
        class_labels = label_binarizer.classes_

    rows: List[Dict[str, object]] = []
    macro_fpr = np.linspace(0, 1, 200)
    interpolated_tprs = []
    for class_index, class_label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_true[:, class_index], score_matrix[:, class_index])
        class_auc = auc(fpr, tpr)
        interpolated_tpr = np.interp(macro_fpr, fpr, tpr)
        interpolated_tpr[0] = 0.0
        interpolated_tprs.append(interpolated_tpr)
        for x_val, y_val in zip(fpr, tpr):
            rows.append(
                {
                    "target": target,
                    "class_label": class_label,
                    "curve_type": "one_vs_rest",
                    "fpr": x_val,
                    "tpr": y_val,
                    "auc": class_auc,
                    "model": model_name,
                    "preprocessing": preprocess_name,
                }
            )

    macro_tpr = np.mean(np.vstack(interpolated_tprs), axis=0)
    macro_auc = auc(macro_fpr, macro_tpr)
    for x_val, y_val in zip(macro_fpr, macro_tpr):
        rows.append(
            {
                "target": target,
                "class_label": "macro-average",
                "curve_type": "macro",
                "fpr": x_val,
                "tpr": y_val,
                "auc": macro_auc,
                "model": model_name,
                "preprocessing": preprocess_name,
            }
        )
    return pd.DataFrame(rows)


def _fit_predict_with_scores(
    model_name: str,
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_test: np.ndarray,
    random_state: int,
    hyperparams: Optional[Mapping[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a classifier and return predictions plus score matrix."""

    hyperparams = dict(hyperparams or {})
    classes = np.array(sorted(pd.Index(y_train).unique().tolist()))
    if model_name == "plsda":
        label_binarizer = LabelBinarizer()
        y_binary = label_binarizer.fit_transform(y_train)
        if y_binary.ndim == 1:
            y_binary = y_binary.reshape(-1, 1)
        n_components = hyperparams.get("n_components")
        if n_components is None:
            n_components = max(1, min(6, x_train.shape[0] - 1, x_train.shape[1], y_binary.shape[1] or 1))
        scaler = StandardScaler()
        x_scaled_train = scaler.fit_transform(x_train)
        x_scaled_test = scaler.transform(x_test)
        pls = PLSRegression(n_components=int(n_components))
        pls.fit(x_scaled_train, y_binary)
        scores = pls.predict(x_scaled_test)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        if scores.shape[1] == 1 and len(label_binarizer.classes_) == 2:
            scores = np.column_stack([1.0 - scores.ravel(), scores.ravel()])
        predictions = np.array([label_binarizer.classes_[idx] for idx in scores.argmax(axis=1)])
        return predictions, scores, label_binarizer.classes_

    if model_name == "svm":
        estimator = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel=str(hyperparams.get("kernel", "rbf")),
                        class_weight="balanced",
                        gamma=hyperparams.get("gamma", "scale"),
                        C=float(hyperparams.get("C", 1.0)),
                        probability=False,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif model_name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=int(hyperparams.get("n_estimators", 300)),
            max_depth=hyperparams.get("max_depth"),
            min_samples_leaf=int(hyperparams.get("min_samples_leaf", 1)),
            random_state=random_state,
            class_weight="balanced_subsample",
        )
    else:
        raise ValueError("Unsupported model: {0}".format(model_name))

    estimator.fit(x_train, y_train)
    predictions = np.asarray(estimator.predict(x_test))
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(x_test)
        model_classes = estimator.classes_
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x_test)
        model_classes = estimator.classes_
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
    else:  # pragma: no cover
        model_classes = classes
        score_index = pd.Categorical(predictions, categories=model_classes).codes
        scores = np.eye(len(model_classes))[score_index]
    return predictions, np.asarray(scores, dtype=float), np.asarray(model_classes)


def _align_score_matrix(scores: np.ndarray, model_classes: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((scores.shape[0], len(all_classes)), dtype=float)
    class_map = {label: idx for idx, label in enumerate(model_classes.tolist())}
    for column_index, class_label in enumerate(all_classes.tolist()):
        if class_label in class_map:
            aligned[:, column_index] = scores[:, class_map[class_label]]
    return aligned


def compute_plsda_vip(
    spectra: pd.DataFrame,
    meta: pd.DataFrame,
    target: str,
    preprocess_name: str,
    n_components: int = 4,
) -> pd.DataFrame:
    """Fit a PLS-DA surrogate model and compute wavelength VIP scores."""

    processed = preprocess_spectra(spectra, preprocess_name)
    target_values = _target_series(meta, target)
    encoder = LabelBinarizer()
    y_matrix = encoder.fit_transform(target_values)
    if y_matrix.ndim == 1:
        y_matrix = y_matrix.reshape(-1, 1)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(processed.to_numpy(dtype=float))
    n_components = max(1, min(int(n_components), x_scaled.shape[0] - 1, x_scaled.shape[1], 6))
    pls = PLSRegression(n_components=n_components)
    pls.fit(x_scaled, y_matrix)

    x_scores = pls.x_scores_
    x_weights = pls.x_weights_
    y_loadings = pls.y_loadings_
    score_contribution = np.sum((x_scores ** 2), axis=0) * np.sum((y_loadings ** 2), axis=0)
    score_contribution = np.where(score_contribution == 0, 1.0, score_contribution)

    weight_norm = np.sum(x_weights ** 2, axis=0)
    weight_norm = np.where(weight_norm == 0, 1.0, weight_norm)
    vip = np.sqrt(
        x_weights.shape[0]
        * np.sum(score_contribution * (x_weights ** 2 / weight_norm), axis=1)
        / np.sum(score_contribution)
    )
    return (
        pd.DataFrame(
            {
                "wavelength_nm": pd.to_numeric(processed.columns, errors="coerce"),
                "vip_score": vip,
                "target": target,
                "preprocessing": preprocess_name,
            }
        )
        .sort_values("wavelength_nm")
        .reset_index(drop=True)
    )


def run_grouped_permutation_test(
    spectra: pd.DataFrame,
    meta: pd.DataFrame,
    target: str,
    preprocess_name: str,
    model_name: str,
    random_state: int = 42,
    n_splits: int = 3,
    n_permutations: int = 24,
) -> pd.DataFrame:
    """Estimate a null distribution for grouped-validation balanced accuracy."""

    processed = preprocess_spectra(spectra, preprocess_name)
    observed, predicted, _ = _evaluate_predictions(
        processed=processed,
        meta=meta,
        target=target,
        model_name=model_name,
        random_state=random_state,
        n_splits=n_splits,
        grouped=True,
    )
    results = [
        {
            "iteration": 0,
            "kind": "observed",
            "balanced_accuracy": balanced_accuracy_score(observed, predicted),
        }
    ]

    rng = np.random.default_rng(random_state)
    base_target = _target_series(meta, target)
    grouped_labels = base_target.groupby(meta["analysis_sample_id"]).first()
    for iteration in range(1, int(n_permutations) + 1):
        permutation_seed = int(rng.integers(0, 1_000_000))
        permuted_groups = grouped_labels.sample(frac=1.0, random_state=permutation_seed).to_numpy()
        permuted_map = dict(zip(grouped_labels.index.tolist(), permuted_groups.tolist()))
        permuted_target = meta["analysis_sample_id"].map(permuted_map)
        null_observed, null_predicted, _ = _evaluate_predictions(
            processed=processed,
            meta=meta,
            target=target,
            model_name=model_name,
            random_state=permutation_seed,
            n_splits=n_splits,
            grouped=True,
            y_override=permuted_target,
        )
        results.append(
            {
                "iteration": iteration,
                "kind": "permuted",
                "balanced_accuracy": balanced_accuracy_score(null_observed, null_predicted),
            }
        )
    return pd.DataFrame(results)
