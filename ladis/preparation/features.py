"""
Feature-role assignment utilities for LaDIS dataset preparation.

This module combines automatic data-quality diagnostics with explicit
dataset-level semantic configuration to determine which variables are:
- retained as modelling features
- stored as recoverable constant metadata
- stored as provenance metadata
- excluded as observation-specific identifiers
- removed because they contain no learnable information

Potential identifiers detected automatically are never removed solely because
of the audit. Identifier exclusion must be explicitly configured.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def validate_identifier_candidates(column_audit: pd.DataFrame, *, expected_id_cands: Sequence[str]) -> list[str]:
    """
    Validate automatically detected identifier candidates.

    The audit can suggest potential identifier variables, but their semantic
    interpretation must be reviewed explicitly. This function ensures that the
    detected candidates match the dataset specification.

    Parameters
    ----------
    column_audit: pandas.DataFrame
        Column-level audit returned by ``build_column_audit``.
    expected_id_cands: sequence of str
        Identifier candidates that have been reviewed and are expected for the
        current dataset.

    Returns
    -------
    list of str
        Automatically detected identifier candidates in audit order.
    """

    if not isinstance(column_audit, pd.DataFrame):
        raise TypeError("`column_audit` must be a pandas DataFrame.")
    required_columns = {"column", "possible_id"}
    missing_columns = required_columns.difference(column_audit.columns)
    if missing_columns:
        raise ValueError(f"The column audit is missing required fields: {sorted(missing_columns)}")
    expected_id_cands = list(expected_id_cands)
    detected_id_cands = column_audit.loc[column_audit["possible_id"], "column"].tolist()
    unexpected_cands = sorted(set(detected_id_cands).difference(expected_id_cands))
    expected_not_detected = sorted(set(expected_id_cands).difference(detected_id_cands))
    if unexpected_cands:
        raise ValueError(f"The audit detected new possible identifiers that have not been reviewed: {unexpected_cands}")
    if expected_not_detected:
        raise ValueError(f"Some expected identifier variables were not detected by the audit: {expected_not_detected}")
    return detected_id_cands


def _extract_single_non_missing_value(series: pd.Series) -> Any:
    """
    Extract the unique non-missing value from a constant variable.

    Parameters
    ----------
    series: pandas.Series
        Variable expected to contain no more than one distinct non-missing
        value.

    Returns
    -------
    Any
        Unique native Python value. ``None`` is returned when the complete
        variable is missing.
    """

    unique_values = series.dropna().unique()
    if len(unique_values) == 0:
        return None
    if len(unique_values) > 1:
        raise ValueError(f"Column '{series.name}' was expected to be constant, but {len(unique_values)} distinct non-missing values were found.")
    value = unique_values[0]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def build_initial_feature_set(df: pd.DataFrame, column_audit: pd.DataFrame, *, recoverable_cte_columns: Sequence[str] = (),
    provenance_columns: Sequence[str] = (), id_columns_to_exclude: Sequence[str] = (), 
    remove_approx_cte_columns: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Assign semantic roles and construct the initial modelling feature set.

    Recoverable constants, provenance variables, identifiers, all-missing
    variables, and non-informative constants are excluded from model input.
    Quasi-constant variables are deliberately retained.

    Parameters
    ----------
    df: pandas.DataFrame
        Complete standardized dataset.
    column_audit: pandas.DataFrame
        Column-level report returned by ``build_column_audit``.
    recoverable_cte_columns: sequence of str, default=()
        Constant metadata that should be restored after synthetic generation.
    provenance_columns: sequence of str, default=()
        Variables that describe the origin of the observations but should not
        participate in generative modelling.
    id_columns_to_exclude: sequence of str, default=()
        Observation-specific identifiers that should neither be modelled nor
        reconstructed.
    remove_approx_cte_columns: bool, default=True
        Whether approximately constant numerical variables should be removed.

    Returns
    -------
    feature_df: pandas.DataFrame
        Initial modelling feature matrix.
    column_roles: pandas.DataFrame
        Role assigned to every standardized input variable.
    selection_artifacts: dict
        Metadata required for reconstruction and reproducibility.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if not isinstance(column_audit, pd.DataFrame):
        raise TypeError("`column_audit` must be a pandas DataFrame.")
    required_audit_columns = {"column", "all_missing", "constant", "approx_cte_num", "quasi_cte", "possible_id"}
    missing_audit_columns = required_audit_columns.difference(column_audit.columns)
    if missing_audit_columns:
        raise ValueError(f"The column audit is missing required fields: {sorted(missing_audit_columns)}")
    recoverable_cte_columns = list(recoverable_cte_columns)
    provenance_columns = list(provenance_columns)
    id_columns_to_exclude = list(id_columns_to_exclude)
    configured_columns = recoverable_cte_columns + provenance_columns + id_columns_to_exclude
    unknown_columns = sorted(set(configured_columns).difference(df.columns))
    if unknown_columns:
        raise ValueError(f"The following configured columns were not found in the DataFrame: {unknown_columns}")
    # Semantic roles must be mutually exclusive
    role_sets = {"recoverable constants": set(recoverable_cte_columns), "provenance": set(provenance_columns), 
        "identifiers": set(id_columns_to_exclude)}
    role_names = list(role_sets)
    for first_index, first_role in enumerate(role_names):
        for second_role in role_names[first_index + 1:]:
            overlap = sorted(role_sets[first_role].intersection(role_sets[second_role]))
            if overlap:
                raise ValueError(f"Dataset column roles must be mutually exclusive. Columns assigned to both '{first_role}' and "
                    f"'{second_role}': {overlap}")
    audit_lookup = column_audit.set_index("column")
    columns_missing_from_audit = sorted(set(df.columns).difference(audit_lookup.index))
    if columns_missing_from_audit:
        raise ValueError(f"The following DataFrame columns are absent from the audit: {columns_missing_from_audit}")
    # Recoverable metadata must genuinely be constant
    for column in recoverable_cte_columns:
        audit_row = audit_lookup.loc[column]
        is_constant_metadata = bool(audit_row["constant"] or audit_row["all_missing"] or audit_row["approx_cte_num"])
        if not is_constant_metadata:
            raise ValueError(f"Recoverable metadata column '{column}' is not constant according to the column audit.")
    # Recoverable constant metadata
    recoverable_cte_vals = {column: _extract_single_non_missing_value(df[column]) for column in recoverable_cte_columns}
    # Provenance metadata
    provenance_vals: dict[str, Any] = {}
    provenance_summaries: dict[str, dict] = {}
    for column in provenance_columns:
        column_data = df[column]
        non_missing_data = column_data.dropna()
        unique_values = non_missing_data.drop_duplicates()
        is_cte_provenance = bool(len(unique_values) <= 1)
        # Only genuinely constant provenance can be restored to every generated observation
        if is_cte_provenance:
            provenance_vals[column] = _extract_single_non_missing_value(column_data)
        value_counts = column_data.value_counts(dropna=False)
        provenance_summaries[column] = {"constant": is_cte_provenance, "unique_n": int(non_missing_data.nunique(dropna=True)),
            "missing_n": int(column_data.isna().sum()), "value_counts": value_counts.to_dict()}
    # Assign one explicit role to every standardized variable
    column_role_records: list[dict] = []
    for column in df.columns:
        audit_row = audit_lookup.loc[column]
        is_all_missing = bool(audit_row["all_missing"])
        is_constant = bool(audit_row["constant"])
        is_approximately_constant = bool(audit_row["approx_cte_num"])
        is_quasi_constant = bool(audit_row["quasi_cte"])
        if column in recoverable_cte_columns:
            role = "rec_cte_metadata"
            remove_from_features = True
            restore_after_generation = True
            reason = ("Constant metadata removed from modelling and stored as one value for later reconstruction.")
        elif column in provenance_columns:
            role = "prov_metadata"
            remove_from_features = True
            restore_after_generation = False
            if is_constant:
                reason = ("Constant dataset-provenance variable excluded from modelling and stored for traceability.")
            else:
                reason = ("Variable dataset-provenance information excluded from modelling and summarized for traceability; "
                    "it is not assigned to synthetic observations.")
        elif column in id_columns_to_exclude:
            role = "excluded_id"
            remove_from_features = True
            restore_after_generation = False
            reason = ("Observation-specific identifier excluded from modelling and not reconstructed for synthetic observations.")
        elif is_all_missing:
            role = "excluded_all_missing"
            remove_from_features = True
            restore_after_generation = False
            reason = ("Variable contains no non-missing information.")
        elif is_constant:
            role = "excluded_cte"
            remove_from_features = True
            restore_after_generation = False
            reason = ("Constant variable contains no learnable variability and was not designated for later reconstruction.")
        elif (is_approximately_constant and remove_approx_cte_columns):
            role = "excluded_approx_cte"
            remove_from_features = True
            restore_after_generation = False
            reason = ("Numerical variable is constant within the configured floating-point tolerance.")
        elif is_quasi_constant:
            role = "retained_quasi_cte_feature"
            remove_from_features = False
            restore_after_generation = False
            reason = ("Quasi-constant variable retained without modification.")
        else:
            role = "retained_feature"
            remove_from_features = False
            restore_after_generation = False
            reason = ("Variable retained as a modelling feature.")
        column_role_records.append({"column": column, "original_position": int(df.columns.get_loc(column)), "role": role,
                "remove_from_features": remove_from_features, "restore_after_generation": restore_after_generation, "reason": reason})
    column_roles = pd.DataFrame(column_role_records)
    # Preserve original dataset column order
    removed_columns = column_roles.loc[column_roles["remove_from_features"], "column"].tolist()
    retained_feature_columns = column_roles.loc[~column_roles["remove_from_features"], "column"].tolist()
    feature_df = df.loc[:, retained_feature_columns].copy()
    # Keep the historical artifact field names so that the new package remains compatible with the existing LaDIS reconstruction interface
    selection_artifacts = {"removed_columns": removed_columns, "retained_feature_columns": retained_feature_columns,
        "rec_cte_columns": recoverable_cte_columns, "rec_cte_vals": recoverable_cte_vals, "prov_columns": provenance_columns,
        "prov_vals": provenance_vals, "prov_summaries": provenance_summaries, 
        "constant_prov_columns": [column for column in provenance_columns if column in provenance_vals],
        "variable_prov_columns": [column for column in provenance_columns if column not in provenance_vals],
        "excluded_id_columns": id_columns_to_exclude, "excluded_nonrec_cte_columns": (column_roles.loc[column_roles["role"].isin(
            {"excluded_cte", "excluded_approx_cte", "excluded_all_missing"}), "column"].tolist()),
        "retained_quasi_cte_columns": (column_roles.loc[column_roles["role"] == "retained_quasi_cte_feature", "column"].tolist())}
    return (feature_df, column_roles, selection_artifacts)