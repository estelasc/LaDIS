"""
Variable-type inference utilities for LaDIS dataset preparation.

This module distinguishes between the variable type inferred directly from the
observed data and the semantic representation used by the generative model.

Numerical variables may therefore be identified automatically as discrete but
later designated as categorical through the dataset configuration. Network
ports are one example of this behavior.

The input DataFrame is never modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


VALID_VARIABLE_TYPES = {"continuous", "binary", "discrete", "categorical", "other"}


def _convert_numpy_scalar_to_python(value: Any) -> Any:
    """
    Convert a NumPy scalar to the equivalent native Python value.
    """

    if isinstance(value, np.generic):
        return value.item()
    return value


def _contains_only_integer_valued_observations(series: pd.Series, *, int_tol: float) -> bool:
    """
    Determine whether all valid numerical observations are effectively integer.

    Floating-point representations such as 1.0, 2.0, and 3.0 are considered
    integer-valued when they fall within the configured tolerance.
    """

    values = pd.to_numeric(series.dropna(), errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return False
    if not np.isfinite(values).all():
        return False
    return bool(np.all(np.isclose(values, np.round(values), atol=int_tol, rtol=0.0)))


def classify_variable_types(df: pd.DataFrame, *, int_tol: float = 1e-9, overrides: Mapping[str, str] | None = None) -> tuple[dict[str, list[str]],
    pd.DataFrame]:
    """
    Infer modelling variable types for all columns in a DataFrame.

    Automatic classification rules are:
    1. Text and pandas categorical variables -> categorical
    2. Boolean variables -> binary
    3. Numerical variables with exactly two observed values -> binary
    4. Other integer-valued numerical variables -> discrete
    5. Remaining numerical variables -> continuous
    6. Empty, constant, datetime, timedelta, or unsupported variables -> other

    Explicit manual overrides may subsequently replace the inferred type.

    Parameters
    ----------
    df: pandas.DataFrame
        Feature-only DataFrame.
    int_tol: float, default=1e-9
        Numerical tolerance used to determine whether values are integer-valued.
    overrides: mapping of str to str, optional
        Explicit column-to-type assignments.

    Returns
    -------
    var_types: dict
        Variables grouped by their final inferred type.
    classification_table: pandas.DataFrame
        Detailed one-row-per-variable classification report.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if int_tol < 0:
        raise ValueError("`int_tol` must be greater than or equal to zero.")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise TypeError("`overrides` must be a mapping from column names to variable types.")
    unknown_override_columns = sorted(set(overrides).difference(df.columns))
    if unknown_override_columns:
        raise ValueError(f"Manual overrides were provided for columns that are not present in the DataFrame: {unknown_override_columns}")
    invalid_override_types = {column: variable_type for column, variable_type in overrides.items() if variable_type not in VALID_VARIABLE_TYPES}
    if invalid_override_types:
        raise ValueError(f"Unsupported manual variable-type overrides were provided: {invalid_override_types}. Valid types are "
            f"{sorted(VALID_VARIABLE_TYPES)}.")
    var_types: dict[str, list[str]] = {"continuous": [], "binary": [], "discrete": [], "categorical": [], "other": []}
    classification_records: list[dict] = []
    for position, column in enumerate(df.columns):
        series = df[column]
        non_missing = series.dropna()
        non_missing_count = len(non_missing)
        unique_count = int(non_missing.nunique(dropna=True))
        example = (_convert_numpy_scalar_to_python(non_missing.iloc[0]) if non_missing_count > 0 else np.nan)
        is_categorical_storage = (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype))
        is_boolean = pd.api.types.is_bool_dtype(series)
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = (pd.api.types.is_datetime64_any_dtype(series))
        is_timedelta = (pd.api.types.is_timedelta64_dtype(series))
        integer_valued = False
        minimum = np.nan
        maximum = np.nan
        if is_numeric and non_missing_count > 0:
            numeric_non_missing = pd.to_numeric(non_missing, errors="coerce").dropna()
            if len(numeric_non_missing) > 0:
                minimum = _convert_numpy_scalar_to_python(numeric_non_missing.min())
                maximum = _convert_numpy_scalar_to_python(numeric_non_missing.max())
        # Automatic classification
        if non_missing_count == 0:
            automatic_type = "other"
            automatic_reason = ("The variable contains no non-missing observations.")
        elif unique_count <= 1:
            automatic_type = "other"
            automatic_reason = ("The variable is constant and should normally have been removed during feature-role assignment.")
        elif is_categorical_storage:
            automatic_type = "categorical"
            automatic_reason = ("The variable uses string, object, or categorical storage.")
        elif is_boolean:
            automatic_type = "binary"
            automatic_reason = ("The variable uses a Boolean data type.")
        elif is_numeric and unique_count == 2:
            automatic_type = "binary"
            automatic_reason = ("The numerical variable has exactly two distinct observed values.")
        elif is_numeric:
            integer_valued = _contains_only_integer_valued_observations(series, int_tol=int_tol)
            if integer_valued:
                automatic_type = "discrete"
                automatic_reason = ("All observed numerical values are integer-valued within the configured tolerance.")
            else:
                automatic_type = "continuous"
                automatic_reason = ("The numerical variable contains non-integer-valued observations.")
        elif is_datetime:
            automatic_type = "other"
            automatic_reason = ("Datetime variables require explicit model-specific handling.")
        elif is_timedelta:
            automatic_type = "other"
            automatic_reason = ("Timedelta variables require explicit model-specific handling.")
        else:
            automatic_type = "other"
            automatic_reason = ("The pandas data type is not supported by the automatic classification rules.")
        # Explicit manual override
        if column in overrides:
            final_type = overrides[column]
            manually_overridden = final_type != automatic_type
            if manually_overridden:
                final_reason = (f"Manually overridden from '{automatic_type}' to '{final_type}'.")
            else:
                final_reason = ("The manual assignment agrees with the automatically inferred type.")
        else:
            final_type = automatic_type
            manually_overridden = False
            final_reason = automatic_reason
        var_types[final_type].append(column)
        binary_values = None
        if final_type == "binary":
            binary_values = [_convert_numpy_scalar_to_python(value) for value in pd.unique(non_missing)]
        classification_records.append({"column": column, "original_position": position, "dtype": str(series.dtype), "non_missing_n": non_missing_count,
            "unique_n": unique_count, "integer_valued": integer_valued, "minimum": minimum, "maximum": maximum, "binary_values": binary_values,
            "automatic_type": automatic_type, "final_type": final_type, "manually_overridden": manually_overridden,
            "classification_reason": final_reason, "example": example})
    classification_table = pd.DataFrame(classification_records)
    return (var_types, classification_table)


def build_model_variable_types(var_types: Mapping[str, Sequence[str]], *, num_cat_columns: Sequence[str] = ()) -> dict[str, list[str]]:
    """
    Convert inferred variable types into model-facing semantic groups.

    Numerical categorical variables are removed from their automatically
    inferred numerical group and added to the categorical representation.

    This distinction allows numerical storage to be preserved in the original
    dataset while using categorical semantics during generative modelling.

    Parameters
    ----------
    var_types: mapping
        Output returned by ``classify_variable_types``.
    numeric_categorical_columns: sequence of str, default=()
        Numerical variables that should be modelled categorically.

    Returns
    -------
    dict
        Model-facing groups containing ``continuous``, ``binary``,
        ``discrete_count``, ``categorical``, and ``numeric_categorical``.
    """

    required_groups = {"continuous", "binary", "discrete", "categorical", "other"}
    missing_groups = required_groups.difference(var_types)
    if missing_groups:
        raise ValueError(f"The variable-type mapping is missing required groups: {sorted(missing_groups)}")
    num_cat_columns = list(num_cat_columns)
    if len(num_cat_columns) != len(set(num_cat_columns)):
        raise ValueError("`num_cat_columns` contains repeated variables.")
    all_typed_columns = {column for group in required_groups for column in var_types[group]}
    unknown_columns = sorted(set(num_cat_columns).difference(all_typed_columns))
    if unknown_columns:
        raise ValueError(f"Numeric categorical variables were not found in the classified feature set: {unknown_columns}")
    # These variables must genuinely use numerical storage before their semantic categorical treatment is applied
    invalid_num_cat_columns = [column for column in num_cat_columns if (column in var_types["categorical"] or column in var_types["other"])]
    if invalid_num_cat_columns:
        raise ValueError(f"The following configured numeric categorical variables were not classified as numerical: {invalid_num_cat_columns}")
    num_cat_set = set(num_cat_columns)
    continuous_columns = [column for column in var_types["continuous"] if column not in num_cat_set]
    binary_columns = [column for column in var_types["binary"] if column not in num_cat_set]
    discrete_count_columns = [column for column in var_types["discrete"] if column not in num_cat_set]
    categorical_columns = (list(var_types["categorical"]) + num_cat_columns)
    model_variable_types = {"continuous": continuous_columns, "binary": binary_columns, "discrete_count": discrete_count_columns,
        "categorical": categorical_columns, "numeric_categorical": num_cat_columns}
    return model_variable_types