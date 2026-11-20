"""
Configuration utilities for LaDIS dataset preparation.

Dataset configuration files describe semantic decisions that should not be
inferred silently from the observed data, together with reproducible
preparation settings.

The configuration is intentionally independent of the input file format.
The same specification can therefore be used with CSV, Excel, Parquet, or
other supported tabular sources.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PREPARATION_CONFIG: dict[str, Any] = {"dataset": {"name": None, "source_database": None},
    "input": {"sheet_name": 0, "numeric_inference_threshold": 1.0}, "audit": {"quasi_constant_threshold": 0.995, 
    "high_cardinality_text_threshold": 0.90, "almost_unique_threshold": 0.995, "approximate_constant_atol": 1e-12,
    "approximate_constant_rtol": 1e-9}, "roles": {"recoverable_constant_columns": [], "provenance_columns": [], "identifier_columns_to_exclude": [],
    "expected_identifier_candidates": [], "remove_approximate_constant_columns": True}, "variable_types": {"integer_tolerance": 1e-9, "overrides": {}, "numeric_categorical_columns": []},
    "categorical_analysis": {"rare_category_threshold_pct": 0.1, "minimum_rare_category_count": 10}, "derived_features": {"sum_rules": {},
    "rate_rules": {}, "std_var_detection": {"enabled": True, "std_suffix": "_std", "var_suffix": "_var", "rtol": 1e-5, "atol": 1e-8,
    "minimum_match_pct": 99.9, "minimum_valid_coverage_pct": 99.9}}, "split": {"train_size": 0.80, "validation_size": 0.10, "test_size": 0.10,
    "random_state": 42, "keep_exact_duplicates_together": True, "maximum_acceptable_deviation_pct": 0.5, "ensure_categorical_train_coverage": True}, "port_tokenization": {"columns": [],
    "top_k_ports": 50, "minimum_exact_count": 20}, "model_preprocessing": {"skewness_log_threshold": 2.0}, "storage": {"parquet_compression": "zstd",
    "joblib_compression": 3, "bundle_version": "1.0"}}


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """
    Recursively update a nested configuration dictionary.

    Values supplied by the user replace defaults while unspecified entries
    retain their default values.
    """

    result = deepcopy(base)
    for key, value in updates.items():
        if (key in result and isinstance(result[key], dict) and isinstance(value, Mapping)):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _validate_preparation_config(config: Mapping[str, Any]) -> None:
    """
    Validate dataset-independent preparation settings.
    """

    num_thr = float(config["input"]["numeric_inference_threshold"])
    if not 0.0 <= num_thr <= 1.0:
        raise ValueError("`numeric_inference_threshold` must be between 0 and 1.")
    split_config = config["split"]
    train_size = float(split_config["train_size"])
    val_size = float(split_config["validation_size"])
    test_size = float(split_config["test_size"])
    for split_name, split_size in {"train": train_size, "validation": val_size, "test": test_size}.items():
        if not 0.0 < split_size < 1.0:
            raise ValueError(f"The {split_name} split proportion must be between 0 and 1.")
    total_size = train_size + val_size + test_size
    if abs(total_size - 1.0) > 1e-12:
        raise ValueError("Train, validation, and test proportions must sum to 1.")
    random_state = split_config["random_state"]
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("`random_state` must be an integer.")
    keep_duplicates_together = split_config["keep_exact_duplicates_together"]
    if not isinstance(keep_duplicates_together, bool):
        raise TypeError("`keep_exact_duplicates_together` must be Boolean.")
    max_deviation_pct = float(split_config["maximum_acceptable_deviation_pct"])
    if not 0.0 <= max_deviation_pct <= 100.0:
        raise ValueError("`maximum_acceptable_deviation_pct` must be between 0 and 100.")
    roles = config["roles"]
    recoverable = set(roles["recoverable_constant_columns"])
    provenance = set(roles["provenance_columns"])
    identifiers = set(roles["identifier_columns_to_exclude"])
    overlaps = {"recoverable/provenance": recoverable & provenance, "recoverable/identifier": recoverable & identifiers,
        "provenance/identifier": provenance & identifiers}
    invalid_overlaps = {name: sorted(columns) for name, columns in overlaps.items() if columns}
    if invalid_overlaps:
        raise ValueError(f"Dataset column roles must be mutually exclusive. Overlaps: {invalid_overlaps}")
    port_config = config["port_tokenization"]
    port_columns = port_config["columns"]
    if isinstance(port_columns, str) or not isinstance(port_columns, Sequence):
        raise TypeError("`port_tokenization.columns` must be a sequence of column names.")
    if len(port_columns) != len(set(port_columns)):
        raise ValueError("`port_tokenization.columns` contains repeated variables.")
    if any(not isinstance(column, str) or not column for column in port_columns):
        raise ValueError("Every configured port column must be a non-empty string.")
    top_k_ports = port_config["top_k_ports"]
    if isinstance(top_k_ports, bool) or not isinstance(top_k_ports, int):
        raise TypeError("`top_k_ports` must be an integer.")
    if top_k_ports < 1:
        raise ValueError("`top_k_ports` must be at least 1.")
    minimum_exact_count = port_config["minimum_exact_count"]
    if isinstance(minimum_exact_count, bool) or not isinstance(minimum_exact_count, int):
        raise TypeError("`minimum_exact_count` must be an integer.")
    if minimum_exact_count < 1:
        raise ValueError("`minimum_exact_count` must be at least 1.")
    storage_config = config["storage"]
    parquet_compression = storage_config["parquet_compression"]
    if parquet_compression is not None and (not isinstance(parquet_compression, str) or not parquet_compression):
        raise ValueError("`parquet_compression` must be a non-empty string or None.")
    joblib_compression = storage_config["joblib_compression"]
    if isinstance(joblib_compression, bool) or not isinstance(joblib_compression, int):
        raise TypeError("`joblib_compression` must be an integer.")
    if not 0 <= joblib_compression <= 9:
        raise ValueError("`joblib_compression` must be between 0 and 9.")
    bundle_version = storage_config["bundle_version"]
    if not isinstance(bundle_version, str) or not bundle_version:
        raise ValueError("`bundle_version` must be a non-empty string.")
    ensure_cat_coverage = split_config["ensure_categorical_train_coverage"]
    if not isinstance(ensure_cat_coverage, bool):
        raise TypeError("`ensure_categorical_train_coverage` must be Boolean.")


def load_preparation_config(path: str | Path) -> dict[str, Any]:
    """
    Load and validate a LaDIS dataset-preparation YAML file.

    User-defined values are recursively merged with the package defaults.

    Parameters
    ----------
    path: str or pathlib.Path
        Path to the YAML dataset specification.

    Returns
    -------
    dict
        Complete validated preparation configuration.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Preparation configuration not found: {path}")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Preparation configuration files must use YAML (`.yaml` or `.yml`).")
    with path.open("r", encoding="utf-8") as file:
        user_config = yaml.safe_load(file)
    if user_config is None:
        user_config = {}
    if not isinstance(user_config, Mapping):
        raise TypeError("The preparation configuration must contain a YAML mapping.")
    config = _deep_update(DEFAULT_PREPARATION_CONFIG, user_config)
    _validate_preparation_config(config)
    return config