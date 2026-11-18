"""
Reconstruction utilities for LaDIS-generated network-traffic data.

This module restores generated observations to their original feature space by
enforcing numerical constraints, reconstructing deterministic variables and
metadata, recovering stored dtypes and column order, and validating the final
synthetic dataset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, columns: Sequence[str], *, dataframe_name: str) -> None:
    """
    Confirm that all required variables are present.
    """

    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{dataframe_name} is missing required variables: {missing_columns}")

    
def reconstruct_derived_features(df: pd.DataFrame, *, derived_feature_artifact: Mapping[str, Any]) -> pd.DataFrame:
    """
    Reconstruct deterministic variables using the stored validated rules.

    The reconstruction order is read directly from the preparation artifact.
    For this dataset, it is:
        1. sum rules
        2. rate rules
        3. variance rules

    Parameters
    ----------
    df: pandas.DataFrame
        Synthetic observations after basic numerical postprocessing.
    derived_feature_artifact: mapping
        Stored deterministic-feature reconstruction information.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the reconstructed deterministic variables.
    """

    rec_df = df.copy()
    rec_order = list(derived_feature_artifact["reconstruction_order"])
    for rule_group_name in rec_order:
        rule_group = derived_feature_artifact[rule_group_name]
        for target_column, rule in rule_group.items():
            operation = rule["operation"]
            # Sum relationships
            if operation == "sum":
                source_columns = list(rule["source_columns"])
                _require_columns(rec_df, source_columns, dataframe_name="Synthetic data before sum reconstruction")
                rec_df[target_column] = rec_df[source_columns].sum(axis=1)
            # Ratio/rate relationships
            elif operation == "ratio":
                numerator_column = rule["numerator"]
                denominator_column = rule["denominator"]
                scale = float(rule.get("scale", 1.0))
                _require_columns(rec_df, [numerator_column, denominator_column], dataframe_name="Synthetic data before rate reconstruction")
                denominator_vals = pd.to_numeric(rec_df[denominator_column], errors="coerce").to_numpy(dtype=float)
                if (~np.isfinite(denominator_vals)).any():
                    raise ValueError(f"Rate denominator '{denominator_column}' contains non-finite values.")
                if (denominator_vals <= 0).any():
                    raise ValueError(f"Rate denominator '{denominator_column}' must be strictly positive before rate reconstruction.")
                numerator_vals = pd.to_numeric(rec_df[numerator_column], errors="coerce").to_numpy(dtype=float)
                rec_df[target_column] = scale*numerator_vals / denominator_vals
            # Variance relationships
            elif operation == "square": 
                source_column = rule["source_column"]
                _require_columns(rec_df, [source_column], dataframe_name="Synthetic data before variance reconstruction")
                source_vals = pd.to_numeric(rec_df[source_column], errors="coerce").to_numpy(dtype=float)
                rec_df[target_column] = np.square(source_vals)
            else:
                raise ValueError(f"Unsupported deterministic-feature operation: '{operation}'")
    return rec_df


def cast_to_original_dtypes(df: pd.DataFrame, *, orig_dtypes: Mapping[str, str]) -> pd.DataFrame:
    """
    Cast reconstructed variables to their stored original dtypes.

    Integer variables are rounded before conversion. Non-finite values and
    values outside the int64 range cause an explicit error rather than silent
    overflow.
    """

    cast_df = df.copy()
    int64_limits = np.iinfo(np.int64)
    for column in cast_df.columns:
        if column not in orig_dtypes:
            raise KeyError(f"No original dtype was stored for '{column}'.")
        expected_dtype = str(orig_dtypes[column])
        if expected_dtype == "int64":
            values = pd.to_numeric(cast_df[column], errors="coerce").to_numpy(dtype=float)
            if (~np.isfinite(values)).any():
                raise ValueError(f"Integer variable '{column}' contains non-finite values.")
            values = np.rint(values)
            if ((values < int64_limits.min) | (values > int64_limits.max)).any():
                raise OverflowError(f"Integer variable '{column}' contains values outside the int64 range.")
            cast_df[column] = values.astype(np.int64)
        elif expected_dtype == "float64":
            cast_df[column] = pd.to_numeric(cast_df[column], errors="coerce").astype(np.float64)
        elif expected_dtype in {"object", "string", "category"}:
            cast_df[column] = (cast_df[column].astype(object))
        else:
            # This branch keeps the function reusable
            try:
                cast_df[column] = (cast_df[column].astype(expected_dtype))
            except Exception as error:
                raise TypeError(f"Could not cast '{column}' to stored dtype '{expected_dtype}'.") from error
    return cast_df


def reconstruct_to_original_space(synth_model_space: pd.DataFrame, *, restored_ports: pd.DataFrame, model_feature_order: Sequence[str],
    disc_count_columns: Sequence[str], nonneg_columns: Sequence[str], strictly_pos_mins: Mapping[str, float], 
    derived_feature_artifact: Mapping[str, Any], recov_cte_vals: Mapping[str, Any], prov_columns: Sequence[str], prov_vals: Mapping[str, Any],
    standardized_column_order: Sequence[str], standardized_dtypes: Mapping[str, str], excluded_id_columns: Sequence[str], 
    include_prov: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct samples to the original reconstructable feature space.

    Numerical variables are already in original scale. This function
    therefore does not inverse-transform or de-standardize them.

    Parameters
    ----------
    synth_model_space: pandas.DataFrame
        Raw sample. Port columns contain categorical tokens.
    restored_ports: pandas.DataFrame
        Concrete source and destination ports recovered from those tokens.
    model_feature_order: sequence of str
        Definitive order of variables used to train the model.
    disc_count_columns: sequence of str
        Numerical variables representing counts or integer-valued magnitudes.
    nonneg_columns: sequence of str
        Variables that must not contain negative values.
    strictly_pos_mins: mapping
        Variables that must be strictly positive and their lower bounds.
    derived_feature_artifact: mapping
        Stored sum, rate, and variance reconstruction rules.
    recov_cte_vals: mapping
        Constant variables that must be restored, such as the traffic label.
    prov_columns: sequence of str
        Provenance-only variables excluded from generative modelling. These
        variables may be constant or observation-dependent.
    prov_vals: mapping
        Constant provenance variables and their stored values. Only provenance
        variables represented here can be restored to synthetic observations.
    standardized_column_order : sequence of str
        Complete standardized column order before feature removal.
    standardized_dtypes: mapping
        Original standardized dtypes.
    excluded_id_columns: sequence of str
        Observation-specific variables that cannot be reconstructed.
    include_prov: bool, default=True
        Whether to restore provenance variables for which a constant value is
        available in ``prov_vals``. Variable provenance is always omitted because
        it cannot be truthfully assigned to synthetic observations.

    Returns
    -------
    rec_df: pandas.DataFrame
        Sample in the original reconstructable representation.
    correction_report: pandas.DataFrame
        Number of values changed by each postprocessing operation.
    """

    if not isinstance(synth_model_space, pd.DataFrame):
        raise TypeError("`synth_model_space` must be a pandas DataFrame.")
    if not isinstance(restored_ports, pd.DataFrame):
        raise TypeError("`restored_ports` must be a pandas DataFrame.")
    model_feature_order = list(model_feature_order)
    disc_count_columns = list(disc_count_columns)
    nonneg_columns = list(nonneg_columns)
    prov_columns = list(prov_columns)
    prov_vals = dict(prov_vals)
    excluded_id_columns = set(excluded_id_columns)
    # Every stored provenance value must correspond to a variable explicitly designated as provenance metadata during data preparation
    unknown_prov_vals = sorted(set(prov_vals).difference(prov_columns))
    if unknown_prov_vals:
        raise ValueError(f"Stored provenance values were found for variables that are not configured as provenance columns: {unknown_prov_vals}")
    _require_columns(synth_model_space, model_feature_order, dataframe_name="LaDIS synthetic model-space data")
    if len(synth_model_space) != len(restored_ports):
        raise ValueError("The restored-port table and LaDIS sample have different numbers of rows.")
    rec_df = (synth_model_space.loc[:, model_feature_order].reset_index(drop=True).copy())
    restored_ports = (restored_ports.reset_index(drop=True).copy())
    correction_records: list[dict] = []
    # Restore concrete port values
    for port_column in restored_ports.columns:
        if port_column not in rec_df.columns:
            raise ValueError(f"Restored port '{port_column}' is absent from the LaDIS feature schema.")
        rec_df[port_column] = restored_ports[port_column]
    # Convert model-generated numerical columns to numerical values
    num_columns = [column for column in model_feature_order if column not in {"protocol"}]
    for column in num_columns:
        rec_df[column] = pd.to_numeric(rec_df[column], errors="coerce")
        if rec_df[column].isna().any():
            raise ValueError(f"Variable '{column}' could not be converted fully to numerical values.")
    # Round discrete counts and magnitudes
    for column in disc_count_columns:
        if column not in rec_df.columns:
            continue
        orig_vals = (rec_df[column].to_numpy(dtype=float))
        rounded_vals = np.rint(orig_vals)
        non_integer_n = int((~np.isclose(orig_vals, rounded_vals, atol=1e-12, rtol=0.0)).sum())
        rec_df[column] = rounded_vals
        correction_records.append({"operation": "round_discrete", "column": column, "modified_n": non_integer_n})
    # Enforce nonnegative domains
    for column in nonneg_columns:
        if column not in rec_df.columns:
            continue
        values = rec_df[column].to_numpy(dtype=float)
        neg_mask = values < 0
        neg_n = int(neg_mask.sum())
        if neg_n > 0:
            values = values.copy()
            values[neg_mask] = 0.0
            rec_df[column] = values
        correction_records.append({"operation": "enforce_nonnegative", "column": column, "modified_n": neg_n})
    # Enforce strictly positive variables
    for (column, min_val) in strictly_pos_mins.items():
        if column not in rec_df.columns:
            continue
        min_val = float(min_val)
        values = rec_df[column].to_numpy(dtype=float)
        invalid_mask = values < min_val
        invalid_n = int(invalid_mask.sum())
        if invalid_n > 0:
            values = values.copy()
            values[invalid_mask] = min_val
            rec_df[column] = values
        correction_records.append({"operation": "enforce_strict_minimum", "column": column, "modified_n": invalid_n})
    # Validate concrete port domains
    for port_column in restored_ports.columns:
        port_values = pd.to_numeric(rec_df[port_column], errors="coerce").to_numpy(dtype=float)
        invalid_port_mask = (~np.isfinite(port_values) | (port_values < 0) | (port_values > 65535))
        if invalid_port_mask.any():
            raise ValueError(f"Restored port '{port_column}' contains values outside the valid 0–65535 range.")
        rec_df[port_column] = np.rint(port_values)
    # Reconstruct deterministic variables
    rec_df = reconstruct_derived_features(rec_df, derived_feature_artifact=derived_feature_artifact)
    # Restore recoverable constant metadata
    for (column, value) in recov_cte_vals.items():
        rec_df[column] = value
    # Only constant provenance values can be truthfully assigned to every synthetic observation
    if include_prov:
        for (column, value) in prov_vals.items():
            rec_df[column] = value
    # Recover the reconstructable original column order
    omitted_columns = set(excluded_id_columns)
    # Provenance columns with no stored constant value are observation-dependent
    nonrestorable_prov_columns = set(prov_columns).difference(prov_vals)
    omitted_columns.update(nonrestorable_prov_columns)
    if not include_prov:
        omitted_columns.update(prov_columns)
    rec_column_order = [column for column in standardized_column_order if column not in omitted_columns]
    _require_columns(rec_df, rec_column_order, dataframe_name="Reconstructed LaDIS synthetic data")
    rec_df = rec_df.loc[:, rec_column_order,].copy()
    # Restore stored original dtypes
    rec_df = cast_to_original_dtypes(rec_df, orig_dtypes=standardized_dtypes)
    correction_report = pd.DataFrame(correction_records)
    return (rec_df, correction_report)

    
def validate_reconstructed_synthetic_data(rec_df: pd.DataFrame, *, expected_column_order: Sequence[str], expected_dtypes: Mapping[str, str],
    excluded_id_columns: Sequence[str], derived_feature_artifact: Mapping[str, Any], traffic_label: Any, allowed_protocol_vals: Sequence[Any],
    include_prov: bool = True, prov_columns: Sequence[str] = (), prov_vals: Mapping[str, Any] | None = None, 
    val_binary_columns: bool = False, binary_allowed_vals: Mapping[str, Sequence[Any]] | None = None, atol: float = 1e-9, 
    rtol: float = 1e-7) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Validate a fully reconstructed synthetic dataset.

    Returns
    -------
    summary: pandas.Series
        Dataset-level validation summary.
    derived_val: pandas.DataFrame
        Per-variable checks for all reconstructed deterministic relationships.
    dtype_val: pandas.DataFrame
        Expected and observed dtypes.
    """

    excluded_columns = set(excluded_id_columns)
    prov_columns = list(prov_columns)
    prov_vals = ({} if prov_vals is None else dict(prov_vals))
    # Provenance variables without one deterministic stored value cannot be reconstructed for synthetic observations
    nonrestorable_prov_columns = set(prov_columns).difference(prov_vals)
    excluded_columns.update(nonrestorable_prov_columns)
    if not include_prov:
        excluded_columns.update(prov_columns)
    expected_columns = [column for column in expected_column_order if column not in excluded_columns]
    schema_matches = rec_df.columns.tolist() == expected_columns
    missing_n = int(rec_df.isna().sum().sum())
    num_columns = (rec_df.select_dtypes(include=np.number).columns.tolist())
    if num_columns:
        inf_n = int(np.isinf(
                rec_df[num_columns].to_numpy(dtype=float)).sum())
    else:
        inf_n = 0
    # Dtype validation
    dtype_records = []
    for column in rec_df.columns:
        expected_dtype = str(expected_dtypes[column])
        observed_dtype = str(rec_df[column].dtype)
        dtype_records.append({"column": column, "expected_dtype": expected_dtype, "observed_dtype": observed_dtype,
            "dtype_matches": expected_dtype == observed_dtype})
    dtype_val = pd.DataFrame(dtype_records)
    # Categorical and domain validation
    label_is_cte = bool("label" in rec_df.columns and rec_df["label"].eq(traffic_label).all())
    protocol_is_valid = bool(rec_df["protocol"].isin(list(allowed_protocol_vals)).all())
    ports_are_valid = True
    for port_column in ["src_port", "dst_port"]:
        if port_column in rec_df.columns:
            port_vals = rec_df[port_column].to_numpy(dtype=float)
            ports_are_valid &= bool(((port_vals >= 0) & (port_vals <= 65535) & np.isclose(port_vals, np.rint(port_vals))).all())
    # Binary-domain validation
    binary_val_performed = bool(val_binary_columns)
    binary_vals_are_valid = True
    invalid_binary_val_n = 0
    binary_columns_checked = 0
    if val_binary_columns:
        if binary_allowed_vals is None:
            raise ValueError("`binary_allowed_vals` must be supplied when `val_binary_columns=True`.")
        binary_allowed_vals = {column: list(values) for column, values in binary_allowed_vals.items()}
        if not binary_allowed_vals:
            raise ValueError("`binary_allowed_vals` cannot be empty when `val_binary_columns=True`.")
        missing_binary_columns = [column for column in binary_allowed_vals if column not in rec_df.columns]
        if missing_binary_columns:
            raise ValueError(f"Binary variables required for reconstruction validation are absent: {missing_binary_columns}")
        invalid_binary_defs = {column: values for column, values in binary_allowed_vals.items() if len(values) != 2}
        if invalid_binary_defs:
            raise ValueError(f"Every binary variable must define exactly two allowed values: {invalid_binary_defs}")
        binary_columns_checked = len(binary_allowed_vals)
        for column, allowed_values in binary_allowed_vals.items():
            invalid_mask = ~rec_df[column].isin(allowed_values)
            invalid_binary_val_n += int(invalid_mask.sum())
        binary_vals_are_valid = bool(invalid_binary_val_n == 0)
    elif binary_allowed_vals is not None:
        raise ValueError("`binary_allowed_vals` was supplied while `val_binary_columns=False`.")
    # Deterministic-feature validation
    derived_records = []
    for (target_column, rule) in derived_feature_artifact["sum_rules"].items():
        expected_vals = (rec_df[rule["source_columns"]].sum(axis=1).to_numpy(dtype=float))
        observed_vals = rec_df[target_column].to_numpy(dtype=float)
        differences = np.abs(observed_vals - expected_vals)
        derived_records.append({"column": target_column, "relationship": "sum",
            "successful": bool(np.allclose(observed_vals, expected_vals, atol=atol, rtol=rtol)), "maximum_absolute_error": float(differences.max())})
    for (target_column, rule) in derived_feature_artifact["rate_rules"].items():
        numerator_vals = rec_df[rule["numerator"]].to_numpy(dtype=float)
        denominator_vals = rec_df[rule["denominator"]].to_numpy(dtype=float)
        expected_vals = float(rule.get("scale", 1.0))*numerator_vals / denominator_vals
        observed_vals = rec_df[target_column].to_numpy(dtype=float)
        differences = np.abs(observed_vals - expected_vals)
        derived_records.append({"column": target_column, "relationship": "ratio",
            "successful": bool(np.allclose(observed_vals, expected_vals, atol=atol, rtol=rtol)), "maximum_absolute_error": float(differences.max())})
    for (target_column, rule) in derived_feature_artifact["variance_rules"].items():
        expected_vals = np.square(rec_df[rule["source_column"]].to_numpy(dtype=float))
        observed_vals = rec_df[target_column].to_numpy(dtype=float)
        differences = np.abs(observed_vals - expected_vals)
        derived_records.append({"column": target_column, "relationship": "square",
            "successful": bool(np.allclose(observed_vals, expected_vals, atol=atol, rtol=rtol)), "maximum_absolute_error": float(differences.max())})
    derived_val = pd.DataFrame(derived_records)
    all_derived_relationships_valid = bool(derived_val["successful"].all())
    all_dtypes_match = bool(dtype_val["dtype_matches"].all())
    summary = pd.Series({"rows": len(rec_df), "columns": rec_df.shape[1], "schema_matches": schema_matches, "missing_values": missing_n,
        "infinite_values": inf_n, "dtypes_match": all_dtypes_match, "label_is_constant": label_is_cte, "protocol_values_are_valid": protocol_is_valid,
        "ports_are_valid": ports_are_valid, "binary_validation_performed": binary_val_performed, "binary_columns_checked": binary_columns_checked,
        "binary_values_are_valid": binary_vals_are_valid, "invalid_binary_values": invalid_binary_val_n,
        "derived_relationships_are_valid": all_derived_relationships_valid}, name="value")
    return (summary, derived_val, dtype_val)