"""
Structural and data-quality auditing utilities for LaDIS preparation.

This module identifies missing and infinite values, constant and
quasi-constant variables, high-cardinality text fields, and potential
identifier variables.

The audit is diagnostic only. It never removes or modifies columns.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_ID_NAME_PATTERNS = (
    # Generic identifiers, hashes, UUIDs, and capture-file references
    r"(^|_)(id|uuid|hash|pcap)($|_)",
    # Source and destination network addresses
    r"(^|_)(src|source|dst|destination)_(ip|mac)($|_)",
    # Explicit timestamp and datetime names
    r"(^|_)(timestamp|datetime|time_stamp)($|_)",
    # Flow-level start and end variables
    r"(^|_)(flow_)?(start|end)($|_)")


def _matches_identifier_pattern(column_name: str, patterns: Sequence[str]) -> bool:
    """
    Determine whether a column name matches an identifier-related pattern.
    """

    return any(re.search(pattern, column_name, flags=re.IGNORECASE) is not None for pattern in patterns)


def _is_approximately_constant_numeric(series: pd.Series, *, abs_tol: float, rel_tol: float) -> bool:
    """
    Determine whether a numerical variable is effectively constant.

    Missing values are ignored. Variables containing infinite values are not
    classified as approximately constant.
    """

    values = (pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float))
    if values.size == 0:
        return False
    if not np.isfinite(values).all():
        return False
    return bool(np.allclose(values, values[0], atol=abs_tol, rtol=rel_tol))


def build_column_audit(df: pd.DataFrame, *, quasi_cte_thr: float = 0.995, high_card_text_thr: float = 0.90, almost_unique_thr: float = 0.995,
    approx_cte_atol: float = 1e-12, approx_cte_rtol: float = 1e-9, id_name_patterns: Sequence[str] = DEFAULT_ID_NAME_PATTERNS) -> pd.DataFrame:
    """
    Generate a structural and data-quality report for every variable.

    For each column, the audit reports:
    - pandas dtype
    - missing-value prevalence
    - cardinality
    - dominant-value prevalence
    - positive and negative infinite values
    - exact constant status
    - approximate numerical constant status
    - quasi-constant status
    - identifier-like name status
    - high-cardinality text status
    - monotonic almost-unique numerical status
    - possible identifier status

    ``possible_id`` is diagnostic only. A flagged variable is never removed
    automatically.

    Parameters
    ----------
    df: pandas.DataFrame
        Standardized dataset to audit.
    quasi_cte_thr: float, default=0.995
        Minimum dominant-value proportion required to classify a variable as
        quasi-constant.
    high_card_text_thr: float, default=0.90
        Minimum unique-value proportion used to flag high-cardinality text.
    almost_unique_thr: float, default=0.995
        Minimum unique-value proportion used when detecting monotonic,
        identifier-like numerical variables.
    approx_cte_atol: float, default=1e-12
        Absolute tolerance used for approximately constant numerical values.
    approx_cte_rtol: float, default=1e-9
        Relative tolerance used for approximately constant numerical values.
    id_name_patterns: sequence of str
        Regular-expression patterns associated with identifier-like names.

    Returns
    -------
    pandas.DataFrame
        One-row-per-variable audit table.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    thresholds = {"quasi_constant_threshold": quasi_cte_thr, "high_cardinality_text_threshold": high_card_text_thr,
        "almost_unique_threshold": almost_unique_thr}
    for threshold_name, threshold_value in thresholds.items():
        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError(f"`{threshold_name}` must be between 0 and 1.")
    audit_records: list[dict] = []
    number_of_rows = len(df)
    for column in df.columns:
        series = df[column]
        # Missing-value statistics
        missing_count = int(series.isna().sum())
        missing_percentage = (100.0 * missing_count / number_of_rows if number_of_rows > 0 else np.nan)
        non_missing = series.dropna()
        non_missing_count = len(non_missing)
        # Cardinality and dominant-value statistics
        unique_count = int(non_missing.nunique(dropna=True))
        unique_ratio = (unique_count / non_missing_count if non_missing_count > 0 else np.nan)
        if non_missing_count > 0:
            value_counts = non_missing.value_counts(dropna=True)
            top_value = value_counts.index[0]
            top_value_count = int(value_counts.iloc[0])
            top_value_ratio = top_value_count / non_missing_count
            example_value = non_missing.iloc[0]
        else:
            top_value = np.nan
            top_value_count = 0
            top_value_ratio = np.nan
            example_value = np.nan
        # Broad dtype classification
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_text = (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))
        # Infinite-value diagnostics
        pos_inf_count = 0
        neg_inf_count = 0
        if is_numeric:
            num_vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            pos_inf_count = int(np.isposinf(num_vals).sum())
            neg_inf_count = int(np.isneginf(num_vals).sum())
        inf_count = pos_inf_count + neg_inf_count
        # Constant and low-variability flags
        all_missing = non_missing_count == 0
        constant = (unique_count == 1 and not all_missing)
        approx_cte_num = False
        if (is_numeric and not all_missing and not constant):
            approx_cte_num = _is_approximately_constant_numeric(series, abs_tol=approx_cte_atol, rel_tol=approx_cte_rtol)
        # Exact constants are deliberately excluded from the quasi-constant category
        quasi_cte = bool(pd.notna(top_value_ratio) and top_value_ratio >= quasi_cte_thr and not constant and not all_missing)
        # Identifier-related diagnostics
        name_suggests_id = _matches_identifier_pattern(column_name=column, patterns=id_name_patterns)
        high_card_text = bool(is_text and pd.notna(unique_ratio) and unique_ratio >= high_card_text_thr)
        monotonic_almost_unique_num = False
        if (is_numeric and non_missing_count > 1 and pd.notna(unique_ratio) and unique_ratio >= almost_unique_thr):
            monotonic_almost_unique_num = bool(non_missing.is_monotonic_increasing or non_missing.is_monotonic_decreasing)
        id_reasons: list[str] = []
        if name_suggests_id:
            id_reasons.append("identifier-like column name")
        if high_card_text:
            id_reasons.append("high-cardinality text")
        if monotonic_almost_unique_num:
            id_reasons.append("monotonic and almost-unique numeric variable")
        possible_id = len(id_reasons) > 0
        # Store complete variable-level diagnostics
        audit_records.append({"column": column, "dtype": str(series.dtype), "rows": number_of_rows, "non_missing_n": int(non_missing_count),
            "missing_n": missing_count, "missing_pct": missing_percentage, "unique_n": unique_count,
            "unique_pct_nonmissing": (100.0 * unique_ratio if pd.notna(unique_ratio) else np.nan), "top_value": top_value, 
            "top_value_n": top_value_count, "top_value_pct_nonmissing": (100.0 * top_value_ratio if pd.notna(top_value_ratio) else np.nan),
            "pos_inf_n": pos_inf_count, "neg_inf_n": neg_inf_count, "inf_n": inf_count, "all_missing": all_missing, "constant": constant,
            "approx_cte_num": approx_cte_num, "quasi_cte": quasi_cte, "name_suggests_id": name_suggests_id, "high_card_text": high_card_text,
            "mono_almost_unique_num": monotonic_almost_unique_num, "possible_id": possible_id, "id_reason": "; ".join(id_reasons), 
            "example": example_value})
    column_audit = pd.DataFrame(audit_records)
    # Prioritize potentially problematic variables for inspection
    column_audit = column_audit.sort_values(by=["possible_id", "all_missing", "constant", "approx_cte_num", "missing_pct"], ascending=[False, False,
        False, False, False]).reset_index(drop=True)
    return column_audit


def build_infinite_value_summary(column_audit: pd.DataFrame) -> pd.DataFrame:
    """
    Extract variables containing positive or negative infinite values.
    """

    required_columns = {"column", "pos_inf_n", "neg_inf_n", "inf_n"}
    missing_columns = required_columns.difference(column_audit.columns)
    if missing_columns:
        raise ValueError(f"The column audit is missing required variables: {sorted(missing_columns)}")
    infinite_summary = column_audit.loc[column_audit["inf_n"] > 0, ["column", "pos_inf_n", "neg_inf_n", "inf_n"]].copy()
    return infinite_summary.sort_values(by="inf_n", ascending=False).reset_index(drop=True)


def build_quality_summary(df: pd.DataFrame, column_audit: pd.DataFrame) -> pd.Series:
    """
    Build a dataset-level summary from the column audit.

    Exact duplicate rows are counted but are not removed.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    duplicate_rows_count = int(df.duplicated(keep="first").sum())
    duplicate_rows_percentage = (100.0 * duplicate_rows_count / len(df) if len(df) > 0 else np.nan)
    quality_summary = pd.Series({"rows": len(df), "columns": df.shape[1], "exact_duplicate_rows": duplicate_rows_count, 
        "exact_duplicate_rows_pct": duplicate_rows_percentage, "all_missing_columns": int(column_audit["all_missing"].sum()),
        "cte_columns": int(column_audit["constant"].sum()), "approx_cte_num_columns": int(column_audit["approx_cte_num"].sum()),
        "quasi_cte_columns": int(column_audit["quasi_cte"].sum()), "possible_id_columns": int(column_audit["possible_id"].sum()),
        "columns_with_missing_values": int((column_audit["missing_n"] > 0).sum()), "total_missing_values": int(column_audit["missing_n"].sum()),
        "columns_with_inf_values": int((column_audit["inf_n"] > 0).sum()), "total_inf_values": int(column_audit["inf_n"].sum())}, name="value")
    return quality_summary