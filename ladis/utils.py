"""
Internal validation and utility functions shared across LaDIS modules.

This module provides helpers for unpacking transformation results and for
validating mixed-type model inputs, DataFrame schemas, and latent-space
representations used throughout the pipeline.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _unpack_dataframe_result(result: Any, *, result_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardise the output of a reversible preprocessing function.

    Some shared preprocessing functions return only a transformed DataFrame,
    whereas others return a tuple containing the transformed data and a
    diagnostics table. This helper converts both interfaces into the same
    two-object representation.

    Parameters
    ----------
    result: Any
        Object returned by a shared preprocessing function.
    result_name: str
        Human-readable name used in error messages.

    Returns
    -------
    transformed_data: pandas.DataFrame
        Definitive transformed observations.
    diagnostics: pandas.DataFrame
        Diagnostics returned by the function, or an empty DataFrame when the
        function does not provide diagnostics.
    """

    if isinstance(result, pd.DataFrame):
        return (result.copy(), pd.DataFrame())
    if isinstance(result, tuple):
        if not result:
            raise RuntimeError(f"`{result_name}` returned an empty tuple.")
        if not isinstance(result[0], pd.DataFrame):
            raise TypeError(f"The first object returned by `{result_name}` must be a pandas DataFrame.")
        transformed_data = result[0].copy()
        diagnostics = (result[1].copy() if (len(result) > 1 and isinstance(result[1], pd.DataFrame)) else pd.DataFrame())
        return (transformed_data, diagnostics)
    raise TypeError(f"`{result_name}` must return a pandas DataFrame or a tuple whose "
        f"first element is a DataFrame. Received: {type(result).__name__}.")


def _validate_dataframe_columns(df: pd.DataFrame, *, expected_columns: Sequence[str], df_name: str) -> None:
    """
    Confirm that a DataFrame preserves an exact expected column order.
    """
    
    expected_columns = list(expected_columns)
    observed_columns = df.columns.tolist()
    if observed_columns != expected_columns:
        missing_columns = sorted(set(expected_columns).difference(observed_columns))
        unexpected_columns = sorted(set(observed_columns).difference(expected_columns))
        raise RuntimeError(f"`{df_name}` does not preserve the expected schema.\n"
            f"Missing columns: {missing_columns}\n"
            f"Unexpected columns: {unexpected_columns}\n"
            f"Expected order: {expected_columns}\n"
            f"Observed order: {observed_columns}")


def _validate_ladis_mixed_type_data(*, num_data: pd.DataFrame, cat_data: pd.DataFrame, num_columns: Sequence[str], cat_columns: Sequence[str],
    cat_cards: Mapping[str, int], data_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate one numerical/categorical autoencoder input partition.
    """

    num_columns = list(num_columns)
    cat_columns = list(cat_columns)
    if num_data.columns.tolist() != num_columns:
        raise RuntimeError(f"The numerical columns of '{data_name}' do not preserve the expected order.")
    if cat_data.columns.tolist() != cat_columns:
        raise RuntimeError(f"The categorical columns of '{data_name}' do not preserve the expected order.")
    if len(num_data) != len(cat_data):
        raise RuntimeError(f"The numerical and categorical blocks of '{data_name}' contain different row counts.")
    if len(num_data) == 0:
        raise ValueError(f"'{data_name}' is empty.")
    num_vals = num_data.to_numpy(dtype=np.float32)
    if not np.isfinite(num_vals).all():
        raise RuntimeError(f"The numerical block of '{data_name}' contains non-finite values.")
    cat_num = cat_data.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(cat_num).all():
        raise RuntimeError(f"The categorical block of '{data_name}' contains non-finite values.")
    if not np.equal(cat_num, np.round(cat_num)).all():
        raise RuntimeError(f"The categorical block of '{data_name}' contains non-integer codes.")
    cat_vals = cat_num.astype(np.int64)
    for column_index, column in enumerate(cat_columns):
        if column not in cat_cards:
            raise RuntimeError(f"No categorical cardinality was provided for '{column}'.")
        cardinality = int(cat_cards[column])
        column_codes = (cat_vals[:, column_index])
        invalid_mask = ((column_codes < 0) | (column_codes >= cardinality))
        if invalid_mask.any():
            invalid_examples = (np.unique(column_codes[invalid_mask])[:10].tolist())
            raise RuntimeError(f"Categorical variable '{column}' in '{data_name}' contains codes outside [0, {cardinality - 1}]. "
                f"Examples: {invalid_examples}")
    return (num_vals, cat_vals)


def _validate_ladis_latent_dataframe(*, latent_data: pd.DataFrame, expected_columns: Sequence[str], data_name: str) -> np.ndarray:
    """
    Validate one continuous latent-space DataFrame.
    """

    expected_columns = list(expected_columns)
    if latent_data.empty:
        raise ValueError(f"'{data_name}' is empty.")
    if (latent_data.columns.tolist() != expected_columns):
        raise RuntimeError(f"'{data_name}' does not preserve the expected latent-column order.")
    latent_vals = latent_data.to_numpy(dtype=np.float64)
    if latent_vals.ndim != 2:
        raise RuntimeError(f"'{data_name}' is not a two-dimensional latent representation.")
    if not np.isfinite(latent_vals).all():
        raise RuntimeError(f"'{data_name}' contains non-finite latent values.")
    return latent_vals