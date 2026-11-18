"""
Schema-standardization utilities for LaDIS dataset preparation.

This module standardizes tabular column names, replaces blank text values with
missing values, identifies text columns that contain numerical information, and
records every schema transformation required for reproducibility.

The input DataFrame is never modified in place.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def normalize_column_name(name: Any) -> str:
    """
    Convert a column name to lowercase snake_case format.

    Parameters
    ----------
    name: Any
        Original column name.

    Returns
    -------
    str
        Normalized column name.
    """

    # Convert the name to text and remove surrounding whitespace
    normalized_name = str(name).strip()
    # Replace non-alphanumeric sequences with underscores
    normalized_name = re.sub(r"[^0-9a-zA-Z]+", "_", normalized_name)
    # Collapse repeated underscores and remove edge underscores
    normalized_name = re.sub(r"_+", "_", normalized_name).strip("_")
    normalized_name = normalized_name.lower()
    # Avoid empty column names after normalization
    if not normalized_name:
        normalized_name = "unnamed_column"
    return normalized_name


def _make_column_names_unique(normalized_names: list[str]) -> list[str]:
    """
    Make normalized column names unique while preserving their order.

    Repeated names receive suffixes such as ``_2``, ``_3``, and so on.
    """

    occurrence_counts: dict[str, int] = {}
    unique_names: list[str] = []
    for name in normalized_names:
        occurrence_counts[name] = occurrence_counts.get(name, 0) + 1
        occurrence_number = occurrence_counts[name]
        if occurrence_number == 1:
            unique_name = name
        else:
            unique_name = f"{name}_{occurrence_number}"
        unique_names.append(unique_name)
    return unique_names


def standardize_dataframe(df: pd.DataFrame, num_thr: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, list]]]:
    """
    Standardize column names and infer numerical text columns.

    A text column is converted to numeric when the proportion of its
    non-missing observations that can be parsed numerically is greater than or
    equal to ``num_thr``.

    Blank strings are treated as missing values before numerical inference.

    Parameters
    ----------
    df: pandas.DataFrame
        Input DataFrame.
    num_thr: float, default=1.0
        Minimum fraction of successfully parsed non-missing values required
        before converting a text column to numeric.

    Returns
    -------
    std_df: pandas.DataFrame
        Standardized copy of the input DataFrame.
    column_name_mapping: pandas.DataFrame
        Mapping between original and standardized column names.
    num_conv_report: pandas.DataFrame
        Numerical-conversion diagnostics for text columns.
    num_coercions: dict
        Exact values coerced to missing values during partial numerical
        conversion.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if not 0.0 <= num_thr <= 1.0:
        raise ValueError("`num_thr` must be between 0 and 1.")
    # Work on a copy so the raw input remains unchanged
    std_df = df.copy()
    # Column-name standardization
    original_names = list(std_df.columns)
    normalized_base_names = [normalize_column_name(name) for name in original_names]
    standardized_names = _make_column_names_unique(normalized_base_names)
    column_name_mapping = pd.DataFrame({"column_position": range(len(original_names)), "original_name": original_names,
        "normalized_base_name": normalized_base_names, "standardized_name": standardized_names})
    std_df.columns = standardized_names
    # Numerical inference for text columns
    conversion_records: list[dict] = []
    numerical_coercions: dict[str, dict[str, list]] = {}
    text_columns = std_df.select_dtypes(include=["object", "string"]).columns.tolist()
    for column in text_columns:
        series = std_df[column]
        original_dtype = str(series.dtype)
        # Treat blank strings as missing observations
        blank_mask = (series.astype("string").str.strip().eq("").fillna(False))
        blank_string_count = int(blank_mask.sum())
        if blank_string_count > 0:
            series = series.mask(blank_mask, np.nan)
            std_df[column] = series
        # Numerical parseability is evaluated only on valid observations
        non_missing_values = series.dropna()
        non_missing_count = len(non_missing_values)
        if non_missing_count == 0:
            conversion_records.append({"column": column, "original_dtype": original_dtype, "final_dtype": str(std_df[column].dtype),
                "non_missing_values": 0, "blank_strings_replaced": blank_string_count, "numeric_parse_rate": np.nan,
                "converted_to_numeric": False, "values_coerced_to_nan": 0, "conversion_status": "all_values_missing"})
            continue
        parsed_non_missing = pd.to_numeric(non_missing_values, errors="coerce")
        successfully_parsed = parsed_non_missing.notna()
        parse_rate = float(successfully_parsed.mean())
        convert_to_numeric = parse_rate >= num_thr
        coerced_count = 0
        if convert_to_numeric:
            failed_indices = parsed_non_missing.index[~successfully_parsed]
            coerced_count = len(failed_indices)
            # Preserve the exact observations altered by partial conversion
            if coerced_count > 0:
                numerical_coercions[column] = {"row_indices": failed_indices.tolist(), "original_values": (non_missing_values.loc[failed_indices]
                    .tolist())}
            std_df[column] = pd.to_numeric(std_df[column], errors="coerce")
            conversion_status = "converted_to_numeric"
        else:
            conversion_status = "retained_as_text"
        conversion_records.append({"column": column, "original_dtype": original_dtype, "final_dtype": str(std_df[column].dtype),
            "non_missing_values": non_missing_count, "blank_strings_replaced": blank_string_count, "numeric_parse_rate": parse_rate,
            "converted_to_numeric": convert_to_numeric, "values_coerced_to_nan": coerced_count, "conversion_status": conversion_status})
    numerical_conversion_report = pd.DataFrame(conversion_records)
    return (std_df, column_name_mapping, numerical_conversion_report, numerical_coercions)