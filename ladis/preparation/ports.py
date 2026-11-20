"""
Network-port tokenization utilities for LaDIS preparation.

This module converts configured source or destination port variables into
categorical tokens suitable for generative modelling.

The tokenizer is fitted exclusively on the training partition. Frequent ports
are preserved exactly, while remaining observations are assigned to semantic
port-range buckets. Validation and test data are transformed with the training
mapping so that no information from held-out partitions influences fitting.

Port tokenization is optional. Datasets without configured port variables pass
through unchanged. Input DataFrames are never modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .splitting import SPLIT_NAMES


PORT_BUCKET_TOKENS = ("missing", "zero", "well_known", "registered", "dynamic_private", "invalid")
PORT_RANGE_BOUNDS = {"well_known": (1, 1023), "registered": (1024, 49151), "dynamic_private": (49152, 65535)}


def _convert_numpy_scalar_to_python(value: Any) -> Any:
    """
    Convert a NumPy scalar to the equivalent native Python value.
    """

    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_port_settings(port_columns: Sequence[str], top_k_ports: int, min_exact_count: int) -> list[str]:
    """
    Validate configured port-tokenization settings.
    """

    if isinstance(port_columns, str) or not isinstance(port_columns, Sequence):
        raise TypeError("`port_columns` must be a sequence of column names.")
    port_columns = list(port_columns)
    if len(port_columns) != len(set(port_columns)):
        raise ValueError("`port_columns` contains repeated variables.")
    if any(not isinstance(column, str) or not column for column in port_columns):
        raise ValueError("Every configured port column must be a non-empty string.")
    if isinstance(top_k_ports, bool) or not isinstance(top_k_ports, (int, np.integer)):
        raise TypeError("`top_k_ports` must be an integer.")
    if int(top_k_ports) < 1:
        raise ValueError("`top_k_ports` must be at least 1.")
    if isinstance(min_exact_count, bool) or not isinstance(min_exact_count, (int, np.integer)):
        raise TypeError("`minimum_exact_count` must be an integer.")
    if int(min_exact_count) < 1:
        raise ValueError("`minimum_exact_count` must be at least 1.")
    return port_columns


def _parse_port_integer(value: Any) -> int | None:
    """
    Parse a scalar as an integer-valued network port when possible.
    """

    if pd.isna(value) or isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        return None
    return int(numeric_value)


def _port_value_to_token(value: Any, *, exact_ports: set[int]) -> str:
    """
    Convert one raw port value to its fitted categorical token.
    """

    if pd.isna(value):
        return "missing"
    port_value = _parse_port_integer(value)
    if port_value is None:
        return "invalid"
    if port_value in exact_ports:
        return f"exact_{port_value}"
    if port_value == 0:
        return "zero"
    if 1 <= port_value <= 1023:
        return "well_known"
    if 1024 <= port_value <= 49151:
        return "registered"
    if 49152 <= port_value <= 65535:
        return "dynamic_private"
    return "invalid"


def _tokenize_port_series(series: pd.Series, *, exact_ports: Sequence[int]) -> pd.Series:
    """
    Tokenize a port Series using a fitted set of exact ports.
    """

    exact_port_set = set(int(port) for port in exact_ports)
    tokens = [_port_value_to_token(value, exact_ports=exact_port_set) for value in series]
    return pd.Series(tokens, index=series.index, name=series.name, dtype="string")


def _build_empirical_distribution(values: Sequence[Any]) -> tuple[list[Any], list[float]]:
    """
    Build an empirical categorical distribution from observed values.
    """

    if not values:
        return ([], [])

    python_values = [_convert_numpy_scalar_to_python(value) for value in values]
    counts = pd.Series(python_values, dtype="object").value_counts(dropna=False, sort=False)
    probabilities = counts.to_numpy(dtype=float)
    probabilities = probabilities / probabilities.sum()
    return ([_convert_numpy_scalar_to_python(value) for value in counts.index.tolist()],
        probabilities.astype(float).tolist())


def fit_port_tokenizer(df_train: pd.DataFrame, *, port_columns: Sequence[str] = (), top_k_ports: int = 50, min_exact_count: int = 20) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fit optional network-port tokenization from training data only.

    Frequent valid non-zero ports are retained as exact categorical tokens when
    they meet the configured minimum training count, capped at ``top_k_ports``.
    All remaining observations are represented by semantic port-range buckets.

    Parameters
    ----------
    df_train: pandas.DataFrame
        Training feature matrix.
    port_columns: sequence of str, default=()
        Variables containing source or destination network ports.
    top_k_ports: int, default=50
        Maximum number of exact frequent ports retained per variable.
    min_exact_count: int, default=20
        Minimum training frequency required for a port to receive its own token.

    Returns
    -------
    port_report: pandas.DataFrame
        Training diagnostics for every configured port variable.
    port_artifacts: dict
        Fitted exact ports, token categories, empirical reconstruction
        distributions, and storage metadata.
    """

    if not isinstance(df_train, pd.DataFrame):
        raise TypeError("`df_train` must be a pandas DataFrame.")
    if df_train.empty:
        raise ValueError("`df_train` must contain at least one observation.")
    port_columns = _validate_port_settings(port_columns, top_k_ports, min_exact_count)
    unknown_columns = sorted(set(port_columns).difference(df_train.columns))
    if unknown_columns:
        raise ValueError(f"Configured port variables were not found in the training DataFrame: {unknown_columns}")
    report_records: list[dict] = []
    column_artifacts: dict[str, dict[str, Any]] = {}
    for column in port_columns:
        series = df_train[column]
        parsed_ports = [_parse_port_integer(value) for value in series]
        valid_ports = [port for port in parsed_ports if port is not None and 0 <= port <= 65535]
        valid_nonzero_ports = [port for port in valid_ports if port != 0]
        # Preserve only sufficiently frequent ports as exact training-derived tokens
        if valid_nonzero_ports:
            port_counts = pd.Series(valid_nonzero_ports, dtype="int64").value_counts()
            eligible_ports = [(int(port), int(count)) for port, count in port_counts.items()
                if int(count) >= int(min_exact_count)]
            eligible_ports.sort(key=lambda item: (-item[1], item[0]))
            exact_ports = [port for port, _ in eligible_ports[:int(top_k_ports)]]
        else:
            port_counts = pd.Series(dtype="int64")
            exact_ports = []
        token_series = _tokenize_port_series(series, exact_ports=exact_ports)
        token_counts = token_series.value_counts(dropna=False)
        # Store empirical distributions for later reconstruction of bucketed ports
        fallback_values: dict[str, list[Any]] = {}
        fallback_probabilities: dict[str, list[float]] = {}
        for bucket_token in ("well_known", "registered", "dynamic_private", "invalid"):
            bucket_values: list[Any] = []
            for value, token in zip(series.tolist(), token_series.tolist()):
                if token != bucket_token:
                    continue
                if bucket_token == "invalid":
                    bucket_values.append(_convert_numpy_scalar_to_python(value))
                else:
                    parsed_value = _parse_port_integer(value)
                    if parsed_value is not None:
                        bucket_values.append(int(parsed_value))
            values, probabilities = _build_empirical_distribution(bucket_values)
            fallback_values[bucket_token] = values
            fallback_probabilities[bucket_token] = probabilities
        exact_port_counts = {port: int(port_counts.get(port, 0)) for port in exact_ports}
        # Fixed semantic buckets remain available even when absent from training
        token_categories = [f"exact_{port}" for port in exact_ports] + list(PORT_BUCKET_TOKENS)
        missing_count = int((token_series == "missing").sum())
        invalid_count = int((token_series == "invalid").sum())
        zero_count = int((token_series == "zero").sum())
        exact_observation_count = int(sum(token_counts.get(f"exact_{port}", 0) for port in exact_ports))
        valid_port_count = int(len(valid_ports))
        unique_valid_port_count = int(len(set(valid_ports)))
        report_records.append({"column": column, "train_n": int(len(series)), "non_missing_n": int(series.notna().sum()), "missing_n": missing_count,
            "valid_port_n": valid_port_count, "invalid_port_n": invalid_count, "zero_port_n": zero_count, "unique_valid_port_n": unique_valid_port_count,
            "exact_port_n": len(exact_ports), "exact_observation_n": exact_observation_count,
            "exact_observation_pct": (100.0 * exact_observation_count / len(series) if len(series) > 0 else np.nan),
            "token_category_n": len(token_categories), "observed_token_n": int(token_series.nunique(dropna=False))})
        column_artifacts[column] = {"original_dtype": str(series.dtype), "original_is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
            "original_is_integer": bool(pd.api.types.is_integer_dtype(series)), "exact_ports": exact_ports, "exact_port_counts": exact_port_counts,
            "token_categories": token_categories, "training_token_counts": {str(token): int(count) for token, count in token_counts.items()},
            "fallback_values": fallback_values, "fallback_probabilities": fallback_probabilities}
    port_report = pd.DataFrame(report_records, columns=["column", "train_n", "non_missing_n", "missing_n", "valid_port_n", "invalid_port_n", "zero_port_n",
        "unique_valid_port_n", "exact_port_n", "exact_observation_n", "exact_observation_pct", "token_category_n", "observed_token_n"])
    port_artifacts = {"enabled": bool(port_columns), "fit_split": "train", "port_columns": port_columns, "top_k_ports": int(top_k_ports),
        "minimum_exact_count": int(min_exact_count), "column_artifacts": column_artifacts}
    return (port_report, port_artifacts)


def transform_port_columns(df: pd.DataFrame, port_artifacts: Mapping[str, Any]) -> pd.DataFrame:
    """
    Transform configured network-port variables using a fitted tokenizer.

    Parameters
    ----------
    df: pandas.DataFrame
        Feature matrix to transform.
    port_artifacts: mapping
        Artifacts returned by ``fit_port_tokenizer``.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with configured ports replaced by tokens.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if not isinstance(port_artifacts, Mapping):
        raise TypeError("`port_artifacts` must be a mapping.")
    required_fields = {"enabled", "port_columns", "column_artifacts"}
    missing_fields = required_fields.difference(port_artifacts)
    if missing_fields:
        raise ValueError(f"Port artifacts are missing required fields: {sorted(missing_fields)}")
    port_columns = list(port_artifacts["port_columns"])
    unknown_columns = sorted(set(port_columns).difference(df.columns))
    if unknown_columns:
        raise ValueError(f"Port variables required by the fitted tokenizer are missing: {unknown_columns}")
    transformed_data = df.copy()
    for column in port_columns:
        column_artifact = port_artifacts["column_artifacts"].get(column)
        if column_artifact is None:
            raise ValueError(f"Port artifacts do not contain fitted information for '{column}'.")
        transformed_data[column] = _tokenize_port_series(df[column],
            exact_ports=column_artifact["exact_ports"])
    return transformed_data


def _sample_range_excluding(low: int, high: int, size: int, *, excluded_values: set[int],
    rng: np.random.Generator) -> np.ndarray:
    """
    Sample integer ports uniformly from a range while excluding exact tokens.
    """

    if size == 0:
        return np.array([], dtype=int)
    available_value_count = (high - low + 1) - sum(low <= value <= high for value in excluded_values)
    if available_value_count <= 0:
        raise ValueError("No port values remain available for fallback reconstruction in the requested range.")
    sampled_values: list[int] = []
    while len(sampled_values) < size:
        batch_size = max(2 * (size - len(sampled_values)), 16)
        candidates = rng.integers(low, high + 1, size=batch_size)
        sampled_values.extend(int(value) for value in candidates if int(value) not in excluded_values)
    return np.asarray(sampled_values[:size], dtype=int)


def _restore_port_storage(series: pd.Series, column_artifact: Mapping[str, Any]) -> pd.Series:
    """
    Restore numerical port storage when the fitted training column was numeric.
    """

    if not bool(column_artifact["original_is_numeric"]):
        return series
    numeric_series = pd.to_numeric(series, errors="coerce")
    original_dtype = str(column_artifact["original_dtype"])
    if bool(column_artifact["original_is_integer"]) and not numeric_series.isna().any():
        try:
            return numeric_series.astype(original_dtype)
        except (TypeError, ValueError):
            pass
    return numeric_series


def inverse_transform_port_tokens(df: pd.DataFrame, port_artifacts: Mapping[str, Any], *, random_state: int = 42) -> pd.DataFrame:
    """
    Reconstruct port values from fitted categorical tokens.

    Exact-port and zero tokens are reconstructed deterministically. Semantic
    range buckets sample from their empirical training distributions when
    available. A uniform in-range fallback is used only when a valid bucket was
    never observed during fitting.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing fitted port tokens.
    port_artifacts: mapping
        Artifacts returned by ``fit_port_tokenizer``.
    random_state: int, default=42
        Seed controlling stochastic reconstruction of bucketed ports.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with port tokens reconstructed to values.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if not isinstance(port_artifacts, Mapping):
        raise TypeError("`port_artifacts` must be a mapping.")
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise TypeError("`random_state` must be an integer.")
    required_fields = {"enabled", "port_columns", "column_artifacts"}
    missing_fields = required_fields.difference(port_artifacts)
    if missing_fields:
        raise ValueError(f"Port artifacts are missing required fields: {sorted(missing_fields)}")
    port_columns = list(port_artifacts["port_columns"])
    unknown_columns = sorted(set(port_columns).difference(df.columns))
    if unknown_columns:
        raise ValueError(f"Tokenized port variables are missing from the DataFrame: {unknown_columns}")
    reconstructed_data = df.copy()
    rng = np.random.default_rng(int(random_state))
    for column in port_columns:
        column_artifact = port_artifacts["column_artifacts"].get(column)
        if column_artifact is None:
            raise ValueError(f"Port artifacts do not contain fitted information for '{column}'.")
        token_series = df[column].astype("string")
        allowed_tokens = set(column_artifact["token_categories"])
        observed_tokens = set(token_series.dropna().tolist())
        unknown_tokens = sorted(observed_tokens.difference(allowed_tokens))
        if token_series.isna().any():
            unknown_tokens.append("<NA>")
        if unknown_tokens:
            raise ValueError(f"Column '{column}' contains tokens not recognized by the fitted tokenizer: {unknown_tokens}")
        exact_ports = [int(port) for port in column_artifact["exact_ports"]]
        exact_token_map = {f"exact_{port}": port for port in exact_ports}
        reconstructed_series = pd.Series(index=df.index, dtype="object", name=column)
        for token in column_artifact["token_categories"]:
            token_mask = token_series == token
            token_count = int(token_mask.sum())
            if token_count == 0:
                continue
            if token in exact_token_map:
                reconstructed_series.loc[token_mask] = exact_token_map[token]
            elif token == "missing":
                reconstructed_series.loc[token_mask] = np.nan
            elif token == "zero":
                reconstructed_series.loc[token_mask] = 0
            elif token in PORT_RANGE_BOUNDS:
                fallback_values = column_artifact["fallback_values"].get(token, [])
                fallback_probabilities = column_artifact["fallback_probabilities"].get(token, [])
                if fallback_values:
                    sampled_values = rng.choice(np.asarray(fallback_values), size=token_count,
                        replace=True, p=np.asarray(fallback_probabilities, dtype=float))
                else:
                    low, high = PORT_RANGE_BOUNDS[token]
                    sampled_values = _sample_range_excluding(low, high, token_count,
                        excluded_values=set(exact_ports), rng=rng)
                reconstructed_series.loc[token_mask] = sampled_values
            elif token == "invalid":
                fallback_values = column_artifact["fallback_values"].get(token, [])
                fallback_probabilities = column_artifact["fallback_probabilities"].get(token, [])
                if fallback_values:
                    sampled_values = rng.choice(np.asarray(fallback_values, dtype="object"), size=token_count,
                        replace=True, p=np.asarray(fallback_probabilities, dtype=float))
                    reconstructed_series.loc[token_mask] = sampled_values
                else:
                    reconstructed_series.loc[token_mask] = -1
        reconstructed_data[column] = _restore_port_storage(reconstructed_series, column_artifact)
    return reconstructed_data


def prepare_port_splits(data_splits: Mapping[str, pd.DataFrame], *, port_columns: Sequence[str] = (), top_k_ports: int = 50,
    min_exact_count: int = 20) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """
    Fit port tokenization on training data and transform all dataset partitions.

    Port fitting never uses validation or test observations. When no port
    columns are configured, all partitions pass through unchanged.

    Parameters
    ----------
    data_splits: mapping
        Train, validation, and test DataFrames.
    port_columns: sequence of str, default=()
        Network-port variables to tokenize. An empty sequence disables this
        optional preparation step.
    top_k_ports: int, default=50
        Maximum number of exact frequent ports retained per variable.
    minimum_exact_count: int, default=20
        Minimum training frequency required for an exact-port token.

    Returns
    -------
    tokenized_splits: dict
        Train, validation, and test DataFrames after optional port tokenization.
    port_report: pandas.DataFrame
        Training-only fitting diagnostics.
    port_artifacts: dict
        Fitted tokenizer information required for held-out transformation and
        later synthetic-data reconstruction.
    """

    if not isinstance(data_splits, Mapping):
        raise TypeError("`data_splits` must be a mapping.")
    missing_splits = set(SPLIT_NAMES).difference(data_splits)
    if missing_splits:
        raise ValueError(f"`data_splits` is missing required partitions: {sorted(missing_splits)}")
    for split_name in SPLIT_NAMES:
        if not isinstance(data_splits[split_name], pd.DataFrame):
            raise TypeError(f"The '{split_name}' partition must be a pandas DataFrame.")
    port_columns = _validate_port_settings(port_columns, top_k_ports, min_exact_count)
    # Fit exclusively on train and reuse the same mapping for held-out partitions
    port_report, port_artifacts = fit_port_tokenizer(data_splits["train"], port_columns=port_columns, top_k_ports=top_k_ports, min_exact_count=min_exact_count)
    tokenized_splits = {split_name: transform_port_columns(data_splits[split_name], port_artifacts) for split_name in SPLIT_NAMES}
    return (tokenized_splits, port_report, port_artifacts)