"""
High-level dataset-preparation pipeline for LaDIS.

This module orchestrates the complete preparation sequence from a raw tabular
dataset to model-ready train, validation, and test partitions.

Preparation remains dataset-independent. Dataset-specific semantic decisions
are supplied through an optional YAML configuration, while conservative package
defaults are used when no configuration file is provided.

The input dataset is never modified in place.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from .audit import build_column_audit, build_infinite_value_summary, build_quality_summary
from .config import DEFAULT_PREPARATION_CONFIG, load_preparation_config
from .derived_features import build_derived_feature_set, identify_rate_candidates
from .features import build_initial_feature_set, validate_identifier_candidates
from .loading import load_tabular_data
from .ports import prepare_port_splits
from .schema import standardize_dataframe
from .splitting import split_train_validation_test
from .storage import save_preparation_bundle
from .types import build_model_variable_types, classify_variable_types


PREPARATION_STAGE_ORDER = ("loading", "schema", "audit", "features", "derived_features", "types", "splitting", "ports", "storage")


def _resolve_preparation_config(config_path: str | Path | None) -> dict[str, Any]:
    """
    Resolve a complete preparation configuration.

    Package defaults are copied when no YAML file is supplied so that the
    shared default configuration is never modified.
    """

    if config_path is None:
        return deepcopy(DEFAULT_PREPARATION_CONFIG)
    return load_preparation_config(config_path)


def _build_effective_numeric_categorical_columns(var_types: dict[str, list[str]], conf_columns: list[str], port_columns: list[str]) -> tuple[list[str], list[str]]:
    """
    Build the effective numerical-categorical representation.

    Configured network-port variables are categorical by construction after
    port tokenization. Numerical port variables are therefore automatically
    added to the model-facing categorical representation when necessary.

    Returns
    -------
    effective_columns: list of str
        Numerical variables represented categorically by the model.
    automatically_added_port_columns: list of str
        Port variables added automatically because they were numerically stored.
    """

    effective_columns = list(conf_columns)
    automatically_added_port_columns: list[str] = []
    num_groups = {"continuous", "binary", "discrete"}
    num_columns = {column for group_name in num_groups for column in var_types[group_name]}
    for column in port_columns:
        if column in num_columns and column not in effective_columns:
            effective_columns.append(column)
            automatically_added_port_columns.append(column)
    return (effective_columns, automatically_added_port_columns)


def prepare_dataset(source: str | Path | pd.DataFrame, *, config_path: str | Path | None = None, output_path: str | Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    """
    Run the complete LaDIS dataset-preparation pipeline.

    The pipeline loads and standardizes a tabular dataset, audits data quality,
    assigns feature roles, removes validated deterministic derived variables,
    infers model-facing variable types, creates reproducible train/validation/
    test partitions, and optionally tokenizes configured network-port variables.

    When ``output_path`` is provided, the final preparation state is also stored
    as a reproducible LaDIS preparation bundle.

    Parameters
    ----------
    source: str, pathlib.Path, or pandas.DataFrame
        Raw tabular dataset.
    config_path: str or pathlib.Path, optional
        YAML preparation configuration. Package defaults are used when omitted.
    output_path: str or pathlib.Path, optional
        Destination directory for a stored preparation bundle. No bundle is
        written when omitted.
    overwrite: bool, default=False
        Whether an existing preparation bundle may be replaced.

    Returns
    -------
    dict
        Complete in-memory preparation result containing ``data_splits``,
        ``artifacts``, ``reports``, ``metadata``, ``manifest``, and
        ``bundle_path``.
    """

    if not isinstance(overwrite, bool):
        raise TypeError("`overwrite` must be Boolean.")
    config = _resolve_preparation_config(config_path)
    # Loading
    raw_data, input_metadata = load_tabular_data(source, sheet_name=config["input"]["sheet_name"])
    # Schema standardization
    standardized_data, column_name_mapping, num_conversion_report, num_coercions = standardize_dataframe(raw_data, num_thr=config["input"]["numeric_inference_threshold"])
    # Structural and data-quality audit
    audit_config = config["audit"]
    column_audit = build_column_audit(standardized_data, quasi_cte_thr=audit_config["quasi_constant_threshold"], high_card_text_thr=audit_config["high_cardinality_text_threshold"],
        almost_unique_thr=audit_config["almost_unique_threshold"], approx_cte_atol=audit_config["approximate_constant_atol"], approx_cte_rtol=audit_config["approximate_constant_rtol"])
    infinite_val_summary = build_infinite_value_summary(column_audit)
    quality_summary = build_quality_summary(standardized_data, column_audit)
    # Feature-role assignment
    role_config = config["roles"]
    id_cands = validate_identifier_candidates(column_audit, expected_id_cands=role_config["expected_identifier_candidates"])
    feature_data, column_roles, selection_artifacts = build_initial_feature_set(standardized_data, column_audit, recoverable_cte_columns=role_config["recoverable_constant_columns"],
        provenance_columns=role_config["provenance_columns"], id_columns_to_exclude=role_config["identifier_columns_to_exclude"],
        remove_approx_cte_columns=role_config["remove_approximate_constant_columns"])
    # Diagnostic only: rate-like names never imply automatic removal
    rate_cands = identify_rate_candidates(feature_data)
    # Deterministic derived-feature removal
    derived_config = config["derived_features"]
    std_var_config = derived_config["std_var_detection"]
    model_feature_data, derived_feature_report, derived_artifacts = build_derived_feature_set(feature_data, sum_rules=derived_config["sum_rules"],
        rate_rules=derived_config["rate_rules"], detect_std_var=std_var_config["enabled"], std_suffix=std_var_config["std_suffix"], var_suffix=std_var_config["var_suffix"],
        rtol=std_var_config["rtol"], atol=std_var_config["atol"], min_match_pct=std_var_config["minimum_match_pct"],
        min_valid_coverage_pct=std_var_config["minimum_valid_coverage_pct"])
    # Variable-type inference
    var_type_config = config["variable_types"]
    var_types, var_type_report = classify_variable_types(model_feature_data, int_tol=var_type_config["integer_tolerance"], overrides=var_type_config["overrides"])
    if var_types["other"]:
        raise ValueError(f"Unsupported modelling variables remain after variable-type inference: {var_types['other']}. Configure their semantic type or remove them before modelling.")
    port_config = config["port_tokenization"]
    conf_port_columns = list(port_config["columns"])
    unknown_port_columns = sorted(set(conf_port_columns).difference(model_feature_data.columns))
    if unknown_port_columns:
        raise ValueError(f"Configured port variables were not found in the final modelling feature set: {unknown_port_columns}")
    effective_num_cat_columns, automatically_added_port_columns = (_build_effective_numeric_categorical_columns(var_types,
        list(var_type_config["numeric_categorical_columns"]), conf_port_columns))
    model_var_types = build_model_variable_types(var_types, num_cat_columns=effective_num_cat_columns)
    # Reproducible train/validation/test splitting
    split_config = config["split"]
    if split_config["ensure_categorical_train_coverage"]:
        train_coverage_columns = (list(model_var_types["binary"]) + [column for column in model_var_types["categorical"] if column not in conf_port_columns])
    else:
        train_coverage_columns = []
    data_splits, split_report, split_artifacts = split_train_validation_test(model_feature_data, train_size=split_config["train_size"],
        val_size=split_config["validation_size"], test_size=split_config["test_size"], random_state=split_config["random_state"],
        keep_exact_duplicates_together=split_config["keep_exact_duplicates_together"], max_deviation_pct=split_config["maximum_acceptable_deviation_pct"],
        train_coverage_columns=train_coverage_columns)
    ## Optional training-fitted network-port tokenization
    prepared_splits, port_report, port_artifacts = prepare_port_splits(data_splits, port_columns=conf_port_columns, top_k_ports=port_config["top_k_ports"],
        min_exact_count=port_config["minimum_exact_count"])
    # Reproducibility artifacts
    artifacts = {"standardized_column_order": standardized_data.columns.tolist(),
        "standardized_dtypes": {column: str(dtype) for column, dtype in standardized_data.dtypes.items()},
        "schema": {"numerical_coercions": num_coercions}, "feature_selection": selection_artifacts, "derived_features": derived_artifacts,
        "variable_types": {"inferred": var_types, "model": model_var_types,
        "configured_numeric_categorical_columns": list(var_type_config["numeric_categorical_columns"]),
        "effective_numeric_categorical_columns": effective_num_cat_columns, "automatically_categorical_port_columns": automatically_added_port_columns},
        "splitting": split_artifacts, "ports": port_artifacts}
    # Diagnostic reports
    reports = {"column_name_mapping": column_name_mapping, "numerical_conversion": num_conversion_report, "column_audit": column_audit,
        "infinite_values": infinite_val_summary, "quality_summary": quality_summary, "identifier_candidates": id_cands, "column_roles": column_roles,
        "rate_candidates": rate_cands, "derived_features": derived_feature_report, "variable_types": var_type_report, "splitting": split_report,
        "ports": port_report}
    # Dataset and pipeline metadata
    metadata = {"dataset": deepcopy(config["dataset"]), "input": input_metadata, "preparation_config": deepcopy(config),
        "pipeline_stage_order": list(PREPARATION_STAGE_ORDER), "stage_shapes": {"raw": list(raw_data.shape), "standardized": list(standardized_data.shape),
        "initial_features": list(feature_data.shape), "model_features": list(model_feature_data.shape), "train": list(prepared_splits["train"].shape),
        "validation": list(prepared_splits["validation"].shape), "test": list(prepared_splits["test"].shape)}}
    # Optional persistent preparation bundle
    manifest = None
    bundle_path = None
    if output_path is not None:
        storage_config = config["storage"]
        manifest = save_preparation_bundle(output_path, prepared_splits, artifacts=artifacts, reports=reports, metadata=metadata,
            parquet_compression=storage_config["parquet_compression"], joblib_compression=storage_config["joblib_compression"],
            bundle_version=storage_config["bundle_version"], overwrite=overwrite)
        bundle_path = str(Path(output_path))
    return {"data_splits": prepared_splits, "artifacts": artifacts, "reports": reports, "metadata": metadata, "manifest": manifest, "bundle_path": bundle_path}