"""
Preprocessing utilities for the LaDIS generative pipeline.

This module provides reversible numerical and categorical transformations,
network-port tokenization, count and mode dequantization, Gaussian quantile
transformation, and the corresponding inverse operations required to recover
the original model representation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from .utils import _validate_dataframe_columns, _unpack_dataframe_result


def _validate_columns(df: pd.DataFrame, columns: Sequence[str], *, arg_name: str) -> list[str]:
    """
    Validate a sequence of requested DataFrame columns.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame in which the columns should exist.
    columns: sequence of str
        Column names to validate.
    arg_name : str
        Name used in error messages.

    Returns
    -------
    list of str
        Validated column names in their supplied order.
    """
    columns = list(columns)
    # Repeated column names could cause a variable to be transformed twice
    duplicated_columns = sorted({column for column in columns if columns.count(column) > 1})
    if duplicated_columns:
        raise ValueError(f"`{arg_name}` contains repeated variables: {duplicated_columns}")
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"The following `{arg_name}` variables are absent from the DataFrame: {missing_columns}")
    return columns


def _to_python_scalar(value: Any) -> Any:
    """
    Convert NumPy scalar objects into native Python values.
    """

    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_numerical_transform_candidates(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """
    Calculate training statistics used to select numerical transformations.

    Parameters
    ----------
    df: pandas.DataFrame
        Training DataFrame.
    columns: sequence of str
        Numerical variables to analyse.

    Returns
    -------
    pandas.DataFrame
        One-row-per-variable summary containing range, skewness, zero
        prevalence, and cardinality information.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    columns = _validate_columns(df, columns, arg_name="columns")
    summary_records: list[dict] = []
    for column in columns:
        # Convert the variable to numeric and retain only finite observations
        values = (pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna())
        if values.empty:
            continue
        summary_records.append({"column": column, "valid_n": int(len(values)), "unique_n": int(values.nunique()), "minimum": values.min(),
            "maximum": values.max(), "mean": values.mean(), "std": values.std(), "skewness": values.skew(), "zero_pct": 100.0*(values == 0).mean(),
            "negative_pct": 100.0*(values < 0).mean()})
    return pd.DataFrame(summary_records)


def fit_reversible_numerical_preprocessor(df_train: pd.DataFrame, *, cont_columns: Sequence[str], 
        disc_columns: Sequence[str], skewness_log_thr: float = 2.0) -> dict:
    """
    Fit reversible numerical transformations using training data only.

    Strongly right-skewed numerical variables receive ``log1p`` followed by
    standardisation when all their training values are non-negative. Remaining
    numerical variables are standardised directly.

    Parameters
    ----------
    df_train: pandas.DataFrame
        Original-scale training feature DataFrame.
    cont_columns: sequence of str
        Continuous numerical variables.
    disc_columns: sequence of str
        Non-binary discrete count or magnitude variables.
    skewness_log_thr: float, default=2.0
        Minimum training skewness required for a non-negative numerical
        variable to receive ``log1p``.

    Returns
    -------
    dict
        Fitted numerical preprocessing artifact.
    """

    if not isinstance(df_train, pd.DataFrame):
        raise TypeError("`df_train` must be a pandas DataFrame.")
    cont_columns = _validate_columns(df_train, cont_columns, arg_name="continuous_columns")
    disc_columns = _validate_columns(df_train, disc_columns, arg_name="discrete_columns")
    overlapping_columns = sorted(set(cont_columns).intersection(disc_columns))
    if overlapping_columns:
        raise ValueError(f"A variable cannot be both continuous and discrete: {overlapping_columns}")
    # Preserve the original DataFrame order
    req_num_columns = set(cont_columns + disc_columns)
    num_columns = [column for column in df_train.columns if column in req_num_columns]
    transformation_summary = (summarize_numerical_transform_candidates(df=df_train, columns=num_columns))
    if len(transformation_summary) != len(num_columns):
        summarised_columns = set(transformation_summary["column"])
        unsummarised_columns = [column for column in num_columns if column not in summarised_columns]
        raise ValueError(f"Some numerical variables contain no valid finite training observations: {unsummarised_columns}")
    summary_lookup = transformation_summary.set_index("column")
    # Apply log1p only to strongly right-skewed variables
    log1p_columns = [column for column in num_columns if (summary_lookup.loc[column, "minimum"] >= 0 and pd.notna(
                summary_lookup.loc[column, "skewness"]) and summary_lookup.loc[column, "skewness"] >= skewness_log_thr)]
    standard_only_columns = [column for column in num_columns if column not in log1p_columns]
    log1p_scaler = None
    standard_scaler = None
    # Fit a scaler after applying log1p to the selected training variables
    if log1p_columns:
        log1p_training_values = np.log1p(df_train[log1p_columns].apply(pd.to_numeric, errors="coerce"))
        log1p_scaler = StandardScaler()
        log1p_scaler.fit(log1p_training_values)
    # Fit a separate scaler to variables that remain in their original scale
    if standard_only_columns:
        standard_training_values = (df_train[standard_only_columns].apply(pd.to_numeric, errors="coerce"))
        standard_scaler = StandardScaler()
        standard_scaler.fit(standard_training_values)
    training_min = {column: _to_python_scalar(summary_lookup.loc[column, "minimum"]) for column in num_columns}
    training_max = {column: _to_python_scalar(summary_lookup.loc[column, "maximum"]) for column in num_columns}
    # These variables can be constrained to zero after inverse transformation
    nonneg_columns = [column for column in num_columns if training_min[column] >= 0]
    numerical_artifact = {"numerical_columns": num_columns, "continuous_columns": list(cont_columns), "discrete_columns": list(disc_columns),
        "log1p_columns": log1p_columns, "standard_only_columns": standard_only_columns, "nonnegative_columns": nonneg_columns,
        "skewness_log_threshold": skewness_log_thr, "log1p_scaler": log1p_scaler, "standard_scaler": standard_scaler, 
        "training_minimum": training_min, "training_maximum": training_max, 
        "original_dtypes": {column: str(df_train[column].dtype) for column in num_columns}, "transformation_summary": transformation_summary}
    return numerical_artifact


def transform_numerical_features(df: pd.DataFrame,num_artifact: dict) -> pd.DataFrame:
    """
    Apply train-fitted numerical transformations to a dataset.

    Parameters
    ----------
    df: pandas.DataFrame
        Original-scale feature DataFrame.
    num_artifact: dict
        Artifact returned by ``fit_reversible_numerical_preprocessor``.

    Returns
    -------
    pandas.DataFrame
        Transformed numerical features in the fitted column order.
    """

    num_columns = num_artifact["numerical_columns"]
    _validate_columns(df, num_columns, arg_name="numerical_columns")
    log1p_columns = num_artifact["log1p_columns"]
    standard_only_columns = num_artifact["standard_only_columns"]
    transformed = pd.DataFrame(index=df.index, columns=num_columns, dtype=float)
    if log1p_columns:
        log1p_values = df[log1p_columns].apply(pd.to_numeric, errors="coerce")
        negative_columns = [column for column in log1p_columns if (log1p_values[column] < 0).any()]
        if negative_columns:
            raise ValueError(f"Negative values were found in variables fitted with log1p: {negative_columns}")
        transformed.loc[:, log1p_columns] = (num_artifact["log1p_scaler"].transform(np.log1p(log1p_values)))
    if standard_only_columns:
        standard_values = (df[standard_only_columns].apply(pd.to_numeric, errors="coerce"))
        transformed.loc[:, standard_only_columns] = (num_artifact["standard_scaler"].transform(standard_values))
    if transformed.isna().any().any():
        affected_columns = transformed.columns[transformed.isna().any()].tolist()
        raise ValueError(f"Missing values were produced during numerical transformation in the following variables: {affected_columns}")
    if not np.isfinite(transformed.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values were produced during numerical transformation.")
    return transformed


def inverse_transform_numerical_features(transformed_df: pd.DataFrame, num_artifact: dict, *, 
        round_disc: bool = True, enforce_nonneg: bool = True) -> pd.DataFrame:
    """
    Reverse the fitted numerical transformations.

    Discrete variables can be rounded to their nearest integer and variables
    observed as non-negative in training can be clipped at zero.

    No upper clipping is applied.

    Parameters
    ----------
    transformed_df: pandas.DataFrame
        Transformed numerical features.
    num_artifact: dict
        Fitted numerical preprocessing artifact.
    round_disc: bool, default=True
        Whether discrete variables should be rounded after inversion.
    enforce_nonneg: bool, default=True
        Whether variables with non-negative training support should be clipped
        at zero after inversion.

    Returns
    -------
    pandas.DataFrame
        Reconstructed original-scale numerical features.
    """

    num_columns = num_artifact["numerical_columns"]
    _validate_columns(transformed_df, num_columns, arg_name="numerical_columns",)
    log1p_columns = num_artifact["log1p_columns"]
    standard_only_columns = num_artifact["standard_only_columns"]
    restored = pd.DataFrame(index=transformed_df.index, columns=num_columns, dtype=float)
    if log1p_columns:
        restored_log_vals = (num_artifact["log1p_scaler"].inverse_transform(transformed_df[log1p_columns]))
        # Reverse log1p using expm1
        restored.loc[:, log1p_columns] = np.expm1(restored_log_vals)
    if standard_only_columns:
        restored.loc[:, standard_only_columns] = (num_artifact["standard_scaler"].inverse_transform(transformed_df[standard_only_columns]))
    if enforce_nonneg:
        for column in num_artifact["nonnegative_columns"]:
            restored[column] = restored[column].clip(lower=0)
    if round_disc:
        for column in num_artifact["discrete_columns"]:
            restored[column] = np.rint(restored[column]).astype("Int64")
    return restored


def fit_ordinal_categorical_encoder(df_train: pd.DataFrame, *, cat_columns: Sequence[str], 
        predefined_cats: Mapping[str, Sequence[Any]] | None = None,) -> dict:
    """
    Fit an ordinal categorical encoder using training data only.

    Categories are assigned integer codes from 0 to K-1. An additional code K
    is reserved for categories not observed during training. The unknown code
    is diagnostic and should not normally occur in the present random split.

    Predefined category sequences may be supplied for variables such as ports.

    Parameters
    ----------
    df_train: pandas.DataFrame
        Original-scale training feature DataFrame.
    cat_columns: sequence of str
        Variables to encode categorically.
    predefined_cats: mapping of str to sequence, optional
        Explicit category definitions for selected variables.

    Returns
    -------
    dict
        Categorical encoding artifact.
    """

    cat_columns = _validate_columns(df_train, cat_columns, arg_name="categorical_columns")
    if predefined_cats is None:
        predefined_cats = {}
    unknown_predefined_columns = sorted(set(predefined_cats).difference(cat_columns))
    if unknown_predefined_columns:
        raise ValueError(f"Predefined categories were supplied for variables that are not categorical inputs: {unknown_predefined_columns}")
    column_info: dict[str, dict] = {}
    for column in cat_columns:
        values = df_train[column]
        if values.isna().any():
            raise ValueError(f"Categorical variable '{column}' contains missing training values.")
        # Preserve the first-observed order
        obs_cats = [_to_python_scalar(value) for value in pd.unique(values)]
        if column in predefined_cats:
            cats = [_to_python_scalar(value) for value in predefined_cats[column]]
            if len(cats) != len(set(cats)):
                raise ValueError(f"Predefined categories for '{column}' contain duplicates.")
            cats_missing_from_definition = [category for category in obs_cats if category not in cats]
            if cats_missing_from_definition:
                raise ValueError(f"Training values in '{column}' are absent from its predefined category list: {cats_missing_from_definition}")
            cat_source = "predefined"
        else:
            cats = obs_cats
            cat_source = "training_observations"
        cat_to_code = {category: code for code, category in enumerate(cats)}
        column_info[column] = {"categories": cats, "category_to_code": cat_to_code, "number_of_categories": len(cats),
            "number_of_observed_training_categories": len(obs_cats), 
            "unobserved_predefined_categories": [category for category in cats if category not in obs_cats], "unknown_code": len(cats),
            "category_source": cat_source, "most_frequent_category": _to_python_scalar(values.value_counts().index[0]), 
            "original_dtype": str(values.dtype)}
    return {"categorical_columns": list(cat_columns), "column_information": column_info}


def transform_categorical_features(df: pd.DataFrame, cat_artifact: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply train-fitted ordinal encoding to categorical variables.

    Categories absent from training receive the reserved unknown code. A
    diagnostic table reports their frequency.

    Returns
    -------
    encoded_df: pandas.DataFrame
        Integer-encoded categorical variables.
    diagnostics: pandas.DataFrame
        Unknown-category statistics for every variable.
    """

    cat_columns = cat_artifact["categorical_columns"]
    _validate_columns(df, cat_columns, arg_name="categorical_columns")
    encoded_df = pd.DataFrame(index=df.index)
    diagnostic_records: list[dict] = []
    for column in cat_columns:
        information = cat_artifact["column_information"][column]
        cat_to_code = information["category_to_code"]
        unknown_code = information["unknown_code"]
        mapped_values = df[column].map(cat_to_code)
        unknown_mask = mapped_values.isna()
        encoded_df[column] = (mapped_values.fillna(unknown_code).astype(np.int64))
        diagnostic_records.append({"column": column, "rows": len(df), "unknown_category_n": int(unknown_mask.sum()),
            "unknown_category_pct": (100.0*unknown_mask.mean()), "unknown_code": unknown_code, 
            "defined_category_n": information["number_of_categories"],
            "observed_training_category_n": information["number_of_observed_training_categories"]})
    diagnostics = pd.DataFrame(diagnostic_records)
    return encoded_df, diagnostics


def inverse_transform_categorical_features(encoded_df: pd.DataFrame, cat_artifact: dict) -> pd.DataFrame:
    """
    Convert integer categorical codes back to their original categories.

    Unknown or invalid codes are returned as pandas missing values.
    """

    cat_columns = cat_artifact["categorical_columns"]
    _validate_columns(encoded_df, cat_columns, arg_name="categorical_columns")
    restored = pd.DataFrame(index=encoded_df.index)
    for column in cat_columns:
        categories = cat_artifact["column_information"][column]["categories"]
        restored_values = []
        for encoded_value in encoded_df[column]:
            if pd.isna(encoded_value):
                restored_values.append(pd.NA)
                continue
            # Generated values should already be integers
            code = int(np.rint(encoded_value))
            if 0 <= code < len(categories):
                restored_values.append(categories[code])
            else:
                restored_values.append(pd.NA)
        restored[column] = pd.Series(restored_values, index=encoded_df.index)
    return restored


# Fixed semantic tokens available for every port variable
PORT_BUCKET_TOKENS = ("missing", "zero", "well_known", "registered", "dynamic_private", "invalid")


def _validate_port_columns(df: pd.DataFrame, port_columns: Sequence[str]) -> list[str]:
    """
    Validate the requested port columns.
    """

    port_columns = list(port_columns)
    missing_columns = [column for column in port_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"The following port variables are absent from the DataFrame: {missing_columns}")
    return port_columns


def _port_bucket_token(port: int | float | pd.NA) -> str:
    """
    Assign a port value to a broad semantic range.

    Port ranges
    -----------
    0
        Zero or no-port value.
    1-1023
        Well-known or system ports.
    1024-49151
        Registered ports.
    49152-65535
        Dynamic, private, or ephemeral ports.
    Values outside 0-65535
        Invalid port values.
    """

    if pd.isna(port):
        return "missing"
    port = int(port)
    if port == 0:
        return "zero"
    if 1 <= port <= 1023:
        return "well_known"
    if 1024 <= port <= 49151:
        return "registered"
    if 49152 <= port <= 65535:
        return "dynamic_private"
    return "invalid"


def make_port_tokens(values: pd.Series, *, exact_ports: Sequence[int]) -> pd.Series:
    """
    Convert raw port numbers into categorical tokens.

    Frequent train-selected ports receive tokens such as ``exact_80``.
    All other ports are assigned to semantic range tokens.
    """

    num_ports = (pd.to_numeric(values, errors="coerce").round().astype("Int64"))
    exact_port_set = {int(port) for port in exact_ports}
    tokens: list[str] = []
    for value in num_ports:
        if pd.isna(value):
            tokens.append("missing")
            continue
        int_port = int(value)
        if int_port in exact_port_set:
            tokens.append(f"exact_{int_port}")
        else:
            tokens.append(_port_bucket_token(int_port))
    return pd.Series(tokens, index=values.index, dtype="string", name=values.name)


def fit_port_tokenizer(df_train: pd.DataFrame, *, port_columns: Sequence[str], top_k_ports: int = 50, min_exact_count: int = 20) -> dict:
    """
    Fit port tokenisation using training data only.

    The most frequent ports are retained exactly, subject to a minimum
    observation count. Remaining ports are grouped by semantic range.

    The empirical training-port distribution within each token is stored so
    that generated tokens can later be converted into concrete port values.

    Parameters
    ----------
    df_train: pandas.DataFrame
        Original-scale training data.
    port_columns: sequence of str
        Port variables to tokenise.
    top_k_ports: int, default=50
        Maximum number of exact ports retained per variable.
    min_exact_count: int, default=20
        Minimum training frequency required for a port to be preserved exactly.

    Returns
    -------
    dict
        Fitted port-tokenisation artifact.
    """

    # Input validation
    if not isinstance(df_train, pd.DataFrame):
        raise TypeError("`df_train` must be a pandas DataFrame.")
    if top_k_ports < 1:
        raise ValueError("`top_k_ports` must be at least 1.")
    if min_exact_count < 1:
        raise ValueError("`minimum_exact_count` must be at least 1.")
    # Validate the requested columns and preserve their supplied order
    port_columns = _validate_port_columns(df_train, port_columns)
    # Ensure that the same port variable was not provided more than once
    if len(port_columns) != len(set(port_columns)):
        raise ValueError("`port_columns` contains duplicated variable names.")
    # Store fitted information independently for every port variable
    column_info: dict[str, dict] = {}
    # Store one compact summary row for every port variable
    fit_summary_records: list[dict] = []
    for column in port_columns:
        # Convert the original values to nullable integer port numbers
        ports = (pd.to_numeric(df_train[column], errors="coerce").round().astype("Int64"))
        # Count the occurrence of every valid training port
        port_counts = ports.value_counts(dropna=True)
        # Retain only frequent ports that satisfy the minimum-count criterion
        eligible_exact_ports = port_counts.loc[port_counts >= min_exact_count]
        # Retain at most the `top_k_ports` most frequent eligible ports
        exact_ports = (eligible_exact_ports.head(top_k_ports).index.astype(int).tolist())
        exact_tokens = [f"exact_{port}" for port in exact_ports]
        # Include the fixed range tokens even if one of them is absent from the training set
        token_cats = (exact_tokens + list(PORT_BUCKET_TOKENS))
        train_tokens = make_port_tokens(values=ports, exact_ports=exact_ports)
        # Store the empirical distribution of concrete ports represented by each token
        token_empirical_values: dict[str, dict] = {}
        for token in token_cats:
            token_mask = train_tokens == token
            observed_ports = (ports.loc[token_mask].dropna().astype(int))
            if observed_ports.empty:
                token_empirical_values[token] = {"values": [], "probabilities": []}
            else:
                empirical_distribution = (observed_ports.value_counts(normalize=True))
                token_empirical_values[token] = {"values": (empirical_distribution.index.astype(int).tolist()),
                    "probabilities": (empirical_distribution.to_numpy(dtype=float).tolist())}     
        # Overall training distribution used only as a defensive fallback if a generated token has no empirical values
        overall_distribution = ports.dropna().astype(int).value_counts(normalize=True)
        if overall_distribution.empty:
            fallback_values = []
            fallback_probabilities = []
        else:
            fallback_values = (overall_distribution.index.astype(int).tolist())
            fallback_probabilities = (overall_distribution.to_numpy(dtype=float).tolist())
        # Store all fitted information for the current port variable
        column_info[column] = {"top_k_ports": top_k_ports, "minimum_exact_count": min_exact_count, "exact_ports": exact_ports, 
            "token_categories": token_cats, "token_empirical_values": token_empirical_values, "fallback_values": fallback_values,
            "fallback_probabilities": fallback_probabilities, "original_dtype": str(df_train[column].dtype)}
        # Build a concise diagnostic summary for the current port variable
        token_counts = train_tokens.value_counts(dropna=False)
        exact_token_mask = train_tokens.str.startswith("exact_", na=False)
        fit_summary_records.append({"column": column, "training_unique_ports": int(ports.nunique(dropna=True)),
            "exact_ports_retained": len(exact_ports), "resulting_token_categories": len(token_cats), 
            "observed_training_tokens": int(train_tokens.nunique()), "exact_token_observations_pct": (100.0*exact_token_mask.mean()),
            "bucketed_training_observations_pct": (100.0*(~train_tokens.str.startswith("exact_")).mean()), "most_frequent_token": token_counts.index[0],
            "most_frequent_token_pct": (100.0*token_counts.iloc[0] / len(train_tokens))})
    if set(column_info) != set(port_columns):
        missing_fitted_columns = sorted(set(port_columns).difference(column_info))
        raise RuntimeError(f"Some requested port variables were not fitted: {missing_fitted_columns}")
    return {"port_columns": port_columns, "column_information": column_info, "fit_summary": pd.DataFrame(fit_summary_records)}


def transform_port_columns(df: pd.DataFrame, port_artifact: dict, *, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply train-fitted port tokenisation to a dataset.

    Returns
    -------
    token_df: pandas.DataFrame
        Tokenised port variables, retaining the original port column names.
    diagnostics: pandas.DataFrame
        Split-level tokenisation diagnostics.
    """

    port_columns = port_artifact["port_columns"]
    _validate_port_columns(df, port_columns)
    token_df = pd.DataFrame(index=df.index)
    diagnostic_records: list[dict] = []
    for column in port_columns:
        info = port_artifact["column_information"][column]
        tokens = make_port_tokens(df[column], exact_ports=info["exact_ports"])
        known_token_mask = tokens.isin(info["token_categories"])
        if not known_token_mask.all():
            unknown_tokens = sorted(tokens.loc[~known_token_mask].unique().tolist())
            raise RuntimeError(f"Unexpected port tokens were produced for '{column}': {unknown_tokens}")
        token_df[column] = tokens
        exact_mask = tokens.str.startswith("exact_")
        diagnostic_records.append({"split": split_name, "column": column, "rows": len(tokens), "unique_raw_ports": int(df[column].nunique(dropna=True)),
            "unique_tokens": int(tokens.nunique(dropna=True)), "exact_token_rows": int(exact_mask.sum()), "bucket_token_rows": int((~exact_mask).sum()),
            "bucket_token_pct": (100.0*(~exact_mask).mean()), "unknown_token_rows": int((~known_token_mask).sum())})
    return (token_df, pd.DataFrame(diagnostic_records))


def inverse_transform_port_tokens(token_df: pd.DataFrame, port_artifact: dict, *, random_state: int = 42) -> pd.DataFrame:
    """
    Convert generated port tokens back into concrete port numbers.

    Exact tokens are reconstructed deterministically. Bucket tokens are
    converted by sampling from the empirical training-port distribution
    associated with that token.
    """

    port_columns = port_artifact["port_columns"]
    _validate_port_columns(token_df, port_columns)
    random_generator = np.random.default_rng(random_state)
    restored_ports = pd.DataFrame(index=token_df.index)
    for column in port_columns:
        info = port_artifact["column_information"][column]
        rec_vals: list[int | pd.NA] = []
        for token in token_df[column].astype("string"):
            if pd.isna(token):
                rec_vals.append(pd.NA)
                continue
            token = str(token)
            if token.startswith("exact_"):
                rec_vals.append(int(token.removeprefix("exact_")))
                continue
            empirical_info = info["token_empirical_values"].get(token, {"values": [], "probabilities": []})
            candidate_vals = empirical_info["values"]
            candidate_probs = empirical_info["probabilities"]
            # A zero-frequency token should rarely be generated
            if not candidate_vals:
                candidate_vals = info["fallback_values"]
                candidate_probs = info["fallback_probabilities"]
            sampled_port = random_generator.choice(candidate_vals, p=candidate_probs)
            rec_vals.append(int(sampled_port))
        restored_ports[column] = pd.Series(rec_vals, index=token_df.index, dtype="Int64")
    return restored_ports


def prepare_ladis_partition(*, df_orig_scale: pd.DataFrame, feature_order: Sequence[str], num_columns: Sequence[str],
    cat_columns: Sequence[str], cat_cards: Mapping[str, int], port_columns: Sequence[str], num_artifact: Mapping[str, Any],
    cat_artifact: Mapping[str, Any], port_artifact: Mapping[str, Any], split_name: str) -> tuple[pd.DataFrame, pd.DataFrame,
    pd.DataFrame, dict[str, pd.DataFrame], pd.Series]:
    """
    Construct one complete LaDIS partition representation.

    The resulting objects serve two different purposes:
        1. Numerical model data contain the reversible numerical
           transformations required by Gaussian diffusion.
        2. Categorical model data contain zero-based integer codes required
           by multinomial diffusion.
        3. Evaluation data retain numerical variables in their original scale
           and categorical variables as strings or port tokens.

    No preprocessing object is fitted in this function. All transformations
    come from the training-fitted preparation artifact.

    Parameters
    ----------
    df_orig_scale: pandas.DataFrame
        Original-scale modelling partition.
    feature_order: sequence of str
        Definitive common modelling-feature order.
    num_columns: sequence of str
        Numerical variables modelled through Gaussian diffusion.
    cat_columns: sequence of str
        Categorical variables modelled through multinomial diffusion.
    cat_cards: mapping
        Stored number of valid codes for each categorical variable.
    port_columns: sequence of str
        Raw source and destination port variables.
    num_artifact: mapping
        Training-fitted numerical preprocessing artifact.
    cat_artifact: mapping
        Training-fitted categorical encoding artifact.
    port_artifact: mapping
        Training-fitted port-tokenisation artifact.
    split_name: str
        Partition identifier used in diagnostics.

    Returns
    -------
    num_model_data: pandas.DataFrame
        Float32 transformed numerical variables.
    cat_model_data: pandas.DataFrame
        Int64 zero-based categorical codes.
    eval_data: pandas.DataFrame
        Original-scale numerical variables and tokenised categorical
        variables, ordered according to `feature_order`.
    diagnostics: dict
        Port-tokenisation, categorical-encoding, and code-range diagnostics.
    representation_summary: pandas.Series
        Compact structural and validity summary.
    """

    feature_order = list(feature_order)
    num_columns = list(num_columns)
    cat_columns = list(cat_columns)
    port_columns = list(port_columns)
    cat_cards = {str(column): int(cardinality) for column, cardinality in cat_cards.items()}
    if split_name not in {"train", "val", "test"}:
        raise ValueError("`split_name` must be 'train', 'val', or 'test'.")
    _validate_dataframe_columns(df_orig_scale, expected_columns=feature_order, df_name=f"{split_name} original-scale partition")
    if len(df_orig_scale) == 0:
        raise ValueError(f"The {split_name} partition is empty.")
    # Convert concrete ports into the tokens fitted on the training split
    port_transform_result = transform_port_columns(df=df_orig_scale, port_artifact=port_artifact, split_name=split_name)
    (tokenised_ports, port_diagnostics,) = _unpack_dataframe_result(port_transform_result, result_name="transform_port_columns")
    tokenised_ports = (tokenised_ports.loc[:, port_columns].reset_index(drop=True))
    _validate_dataframe_columns(tokenised_ports, expected_columns=port_columns, df_name=f"{split_name} tokenised ports")
    if len(tokenised_ports) != len(df_orig_scale):
        raise RuntimeError(f"The number of tokenised {split_name} ports differs from the number of original observations.")
    # Table used only for common statistical evaluation
    eval_data = (df_orig_scale.reset_index(drop=True).copy())
    for column in port_columns:
        eval_data[column] = (tokenised_ports[column].reset_index(drop=True).astype("string"))
    eval_data = eval_data.loc[:, feature_order].copy()
    # Port tokens must be stored as categorical strings in the common evaluation representation
    invalid_port_dtypes = {column: str(eval_data[column].dtype) for column in port_columns 
        if not (pd.api.types.is_string_dtype(eval_data[column].dtype) or pd.api.types.is_object_dtype(eval_data[column].dtype))}
    if invalid_port_dtypes:
        raise RuntimeError(f"The tokenised port variables do not use a categorical string dtype: {invalid_port_dtypes}")
    # Apply the training-fitted numerical transformation
    num_transform_result = transform_numerical_features(eval_data, num_artifact)
    (num_model_data, num_diagnostics) = _unpack_dataframe_result(num_transform_result, result_name="transform_numerical_features")
    num_model_data = (num_model_data.loc[:, num_columns].reset_index(drop=True).astype(np.float32))
    _validate_dataframe_columns(num_model_data, expected_columns=num_columns, df_name=f"{split_name} LaDIS numerical data")
    num_vals = num_model_data.to_numpy(dtype=np.float32)
    if np.isnan(num_vals).any():
        affected_columns = num_model_data.columns[num_model_data.isna().any()].tolist()
        raise RuntimeError(f"The transformed {split_name} numerical data contain missing values in: {affected_columns}")
    if np.isinf(num_vals).any():
        affected_columns = num_model_data.columns[np.isinf(num_model_data.to_numpy(dtype=np.float32)).any(axis=0)].tolist()
        raise RuntimeError(f"The transformed {split_name} numerical data contain infinite values in: {affected_columns}")
    # Encode tokenised ports and the remaining categorical variables
    cat_transform_result = transform_categorical_features(eval_data, cat_artifact)
    (cat_model_data, cat_diagnostics) = _unpack_dataframe_result(cat_transform_result, result_name="transform_categorical_features")
    cat_model_data = (cat_model_data.loc[:, cat_columns].reset_index(drop=True))
    # Coerce to numerical values before enforcing the integer model dtype
    for column in cat_columns:
        cat_model_data[column] = pd.to_numeric(cat_model_data[column], errors="raise")
        values = cat_model_data[column].to_numpy(dtype=float)
        if not np.equal(values, np.round(values)).all():
            raise RuntimeError(f"The encoded categorical variable '{column}' contains non-integer codes.")
    cat_model_data = cat_model_data.astype(np.int64)
    _validate_dataframe_columns(cat_model_data, expected_columns=cat_columns, df_name=f"{split_name} LaDIS categorical data")
    # Every code supplied to multinomial diffusion must be inside the zero-based interval defined by its stored cardinality
    code_range_records: list[dict[str, Any]] = []
    for column in cat_columns:
        if column not in cat_cards:
            raise RuntimeError(f"No categorical cardinality was stored for '{column}'.")
        cardinality = int(cat_cards[column])
        column_codes = cat_model_data[column].to_numpy(dtype=np.int64)
        invalid_mask = ((column_codes < 0) | (column_codes >= cardinality))
        invalid_n = int(invalid_mask.sum())
        if invalid_n:
            invalid_examples = np.unique(column_codes[invalid_mask])[:10].tolist()
            raise RuntimeError(f"The encoded {split_name} variable '{column}' contains "
                f"{invalid_n} codes outside [0, {cardinality - 1}]. Examples: {invalid_examples}")
        code_range_records.append({"split": split_name, "column": column, "cardinality": cardinality, 
            "observed_code_min": int(column_codes.min()), "observed_code_max": int(column_codes.max()),
            "observed_categories": int(np.unique(column_codes).size), "invalid_code_n": invalid_n})
    code_range_diagnostics = pd.DataFrame(code_range_records)
    # Validate the complete representation
    if not (len(num_model_data) == len(cat_model_data) == len(eval_data)):
        raise RuntimeError(f"The {split_name} numerical, categorical, and evaluation representations contain different row counts.")
    model_feature_count = (num_model_data.shape[1] + cat_model_data.shape[1])
    if model_feature_count != len(feature_order):
        raise RuntimeError("The number of LaDIS numerical and categorical inputs does not equal the definitive model-feature count.")
    representation_summary = pd.Series({"split": split_name, "rows": len(eval_data), "numerical_variables": num_model_data.shape[1],
        "categorical_variables": cat_model_data.shape[1], "raw_model_variables": model_feature_count,
        "evaluation_variables": eval_data.shape[1], "numerical_dtype": str(num_model_data.dtypes.iloc[0]),
        "categorical_dtype": str(cat_model_data.dtypes.iloc[0]), "numerical_missing_n": int(num_model_data.isna().sum().sum()),
        "numerical_infinite_n": int(np.isinf(num_vals).sum()), "categorical_missing_n": int(cat_model_data.isna().sum().sum()),
        "invalid_categorical_code_n": int(code_range_diagnostics["invalid_code_n"].sum()), "representation_valid": True}, name="value")
    diagnostics = {"port_tokenization": port_diagnostics, "numerical_transformation": num_diagnostics,
        "categorical_encoding": cat_diagnostics, "categorical_code_ranges": code_range_diagnostics}
    return (num_model_data, cat_model_data, eval_data, diagnostics, representation_summary)


def restrict_ladis_categories_to_training_support(*, cat_artifact: Mapping[str, Any], training_cat_data: pd.DataFrame, 
    cat_columns: Sequence[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Remove categorical states that were never observed in training.

    Multinomial diffusion can sample any state included in its configured
    cardinality. Predefined values such as ``missing`` or ``invalid`` must
    therefore not be included as valid generative states when they did not
    occur in the training partition.

    The returned artifact preserves the existing categorical-artifact
    interface, but its category lists, mappings, cardinalities, and unknown
    codes are rebuilt from observed training values only.
    """

    cat_columns = list(cat_columns)
    if (training_cat_data.columns.tolist() != cat_columns):
        raise RuntimeError("The training categorical table does not preserve the definitive LaDIS categorical order.")
    if training_cat_data.isna().any().any():
        affected_columns = training_cat_data.columns[training_cat_data.isna().any()].tolist() 
        raise RuntimeError(f"The LaDIS categorical training representation contains missing values in: {affected_columns}")
    rest_artifact = deepcopy(cat_artifact)
    rest_artifact["categorical_columns"] = list(cat_columns)
    column_info = rest_artifact["column_information"]
    rest_records: list[dict[str, Any]] = []
    for column in cat_columns:
        information = column_info[column]
        # Preserve the original category types stored during data preparation
        orig_cats = list(information["categories"])
        observed_cats = training_cat_data[column].dropna().unique().tolist()
        observed_cat_set = set(observed_cats)
        # Preserve the original artifact order wherever possible
        retained_cats = [category for category in orig_cats if category in observed_cat_set]
        retained_cat_set = set(retained_cats)
        # Include any observed category not represented in the original list
        retained_cats.extend(category for category in observed_cats if category not in retained_cat_set)
        removed_cats = [category for category in orig_cats if category not in observed_cat_set]
        if not retained_cats:
            raise RuntimeError(f"No observed training categories remain for '{column}'.")
        cat_to_code = {category: category_code for category_code, category in enumerate(retained_cats)}
        information["categories"] = retained_cats
        information["category_to_code"] = cat_to_code
        if "code_to_category" in information:
            information["code_to_category"] = {category_code: category for category, category_code in cat_to_code.items()}
        information["number_of_categories"] = len(retained_cats)
        information["number_of_observed_training_categories"] = len(retained_cats)
        # There are no unobserved categories in the generative state space
        information["unobserved_predefined_categories"] = []
        # Unknown values remain outside the valid generative code range
        information["unknown_code"] = len(retained_cats)
        # Preserve removed values for auditability
        information["excluded_generation_categories"] = removed_cats
        rest_records.append({"column": column, "original_cardinality": len(orig_cats), "observed_training_cardinality": len(retained_cats),
            "removed_unobserved_n": len(removed_cats), "removed_unobserved_categories": removed_cats})
    rest_summary = pd.DataFrame(rest_records)
    return (rest_artifact, rest_summary)


def fit_ladis_count_dequantization(*, train_num_data: pd.DataFrame, disc_count_columns: Sequence[str], random_state: int,
    gap_fraction: float = 0.10) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """
    Add deterministic local tie-breaking noise to count variables before
    fitting the Gaussian quantile transformer.

    Each observed count level receives a jitter interval based on the gaps to
    its own neighbouring levels. 
    Adjacent count-level intervals cannot overlap because ``gap_fraction`` is
    restricted to values below 0.5. Boundary levels are dequantized only
    toward the interior of the observed support.
    The noise is used only in LaDIS preprocessing. Count semantics are
    restored later by the existing inverse numerical transformation and
    rounding.
    """

    if train_num_data.empty:
        raise ValueError("`train_num_data` cannot be empty.")
    if not 0.0 < gap_fraction < 0.5:
        raise ValueError("`gap_fraction` must be inside (0, 0.5).")
    disc_count_columns = list(disc_count_columns)
    missing_columns = sorted(set(disc_count_columns).difference(train_num_data.columns))
    if missing_columns:
        raise RuntimeError(f"Some count variables are absent from the LaDIS numerical table: {missing_columns}")
    output_data = (train_num_data.reset_index(drop=True).astype(np.float64).copy())
    rng = np.random.default_rng(random_state)
    column_info: dict[str, dict[str, Any]] = {}
    diagnostic_records: list[dict[str, Any]] = []
    for column in disc_count_columns:
        values = pd.to_numeric(output_data[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Count variable '{column}' contains non-finite values before dequantization.")
        (unique_vals, inverse_indices, value_counts) = np.unique(values, return_inverse=True, return_counts=True)
        n_levels = len(unique_vals)
        # Store one train-fitted jitter interval per observed count level
        jitter_lows = np.zeros(n_levels, dtype=np.float64)
        jitter_highs = np.zeros(n_levels, dtype=np.float64)
        if n_levels == 1:
            # This branch keeps the helper robust to a count variable that becomes constant in a particular training partition
            scale = max(abs(float(unique_vals[0])), 1.0)
            constant_half_width = float(1e-6 * scale)
            jitter_lows[0] = -constant_half_width
            jitter_highs[0] = constant_half_width
            min_gap = np.nan
        else:
            gaps = np.diff(unique_vals)
            if (not np.isfinite(gaps).all() or np.any(gaps <= 0.0)):
                raise RuntimeError(f"Count variable '{column}' contains invalid ordered training gaps.")
            min_gap = float(gaps.min())
            # Lower boundary: move only upward
            jitter_highs[0] = float(gap_fraction)*gaps[0]
            # Upper boundary: move only downward
            jitter_lows[-1] = -float(gap_fraction)*gaps[-1]
            # Interior levels use their own lower and upper neighbouring gaps
            if n_levels > 2:
                jitter_lows[1:-1] = -float(gap_fraction)*gaps[:-1]
                jitter_highs[1:-1] = float(gap_fraction)*gaps[1:]
            # Explicitly confirm that adjacent dequantization intervals cannot overlap
            interval_upper = unique_vals[:-1] + jitter_highs[:-1]
            next_interval_lower = unique_vals[1:] + jitter_lows[1:]
            if np.any(interval_upper >= next_interval_lower):
                raise RuntimeError(f"Count dequantization intervals overlap for '{column}'.")
        # Convert the level-specific intervals into one interval per row
        row_jitter_lows = jitter_lows[inverse_indices]
        row_jitter_highs = jitter_highs[inverse_indices]
        uniform_draws = rng.random(len(values))
        noise = row_jitter_lows + uniform_draws*(row_jitter_highs - row_jitter_lows)
        dequantized_values = values + noise
        if not np.isfinite(dequantized_values).all():
            raise RuntimeError(f"Count dequantization produced non-finite values for '{column}'.")
        output_data[column] = dequantized_values
        interval_widths = jitter_highs - jitter_lows
        top_level_index = int(np.argmax(value_counts))
        # Store the complete train-fitted local dequantization interface
        column_info[column] = {"training_unique_values": int(n_levels), "training_minimum_gap": min_gap, "training_levels": unique_vals.tolist(),
            "training_level_counts": value_counts.astype(int).tolist(), "jitter_low": jitter_lows.tolist(), "jitter_high": jitter_highs.tolist(),
            "minimum_jitter_interval_width": float(interval_widths.min()), "maximum_jitter_interval_width": float(interval_widths.max()),
            "top_level": float(unique_vals[top_level_index]), "top_level_n": int(value_counts[top_level_index]),
            "top_level_pct": float(100.0*value_counts[top_level_index] / len(values))}
        diagnostic_records.append({"column": column, "training_unique_values": int(n_levels), "training_minimum_gap": min_gap,
            "minimum_jitter_interval_width": float(interval_widths.min()), "maximum_jitter_interval_width": float(interval_widths.max()),
            "top_level": float(unique_vals[top_level_index]), "top_level_n": int(value_counts[top_level_index]),
            "top_level_pct": float(100.0*value_counts[top_level_index] / len(values))})
    if not np.isfinite(output_data.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Count dequantization produced non-finite training values.")
    artifact = {"method": "local_gap_uniform_tie_breaking_before_quantile", "random_state": int(random_state), 
        "gap_fraction": float(gap_fraction), "columns": list(disc_count_columns), "column_information": column_info}
    diagnostics = pd.DataFrame(diagnostic_records)
    return output_data, artifact, diagnostics


def apply_ladis_count_dequantization(*, num_data: pd.DataFrame, dequantization_artifact: Mapping[str, Any], random_state: int,
    split_name: str) -> pd.DataFrame:
    """
    Apply train-fitted local count-dequantization intervals to one partition.

    Count levels observed during training use exactly the level-specific jitter
    intervals fitted on the training partition. Count levels that were not
    observed during training are left unchanged rather than assigning them an
    unsupported interval.
    """

    if num_data.empty:
        raise ValueError(f"The '{split_name}' numerical partition is empty.")
    output_data = (num_data.reset_index(drop=True).astype(np.float64).copy())
    columns = list(dequantization_artifact["columns"])
    column_info = dequantization_artifact["column_information"]
    missing_columns = sorted(set(columns).difference(output_data.columns))
    if missing_columns:
        raise RuntimeError(f"Split '{split_name}' is missing dequantized variables: {missing_columns}")
    rng = np.random.default_rng(random_state)
    for column in columns:
        values = pd.to_numeric(output_data[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Split '{split_name}' contains non-finite values in '{column}'.")
        info = column_info[column]
        training_levels = np.asarray(info["training_levels"], dtype=np.float64)
        jitter_lows = np.asarray(info["jitter_low"], dtype=np.float64)
        jitter_highs = np.asarray(info["jitter_high"], dtype=np.float64)
        if not (len(training_levels) == len(jitter_lows) == len(jitter_highs)):
            raise RuntimeError(f"The stored count-dequantization interface for '{column}' has inconsistent lengths.")
        if np.any(np.diff(training_levels) <= 0.0):
            raise RuntimeError(f"The stored training levels for '{column}' are not strictly increasing.")
        # Locate every current value in the sorted training-level array
        level_positions = np.searchsorted(training_levels, values)
        clipped_positions = np.clip(level_positions, 0, len(training_levels) - 1)
        # Only exact train-observed levels receive the fitted jitter
        matched_training_level = ((level_positions < len(training_levels)) & (training_levels[clipped_positions] == values))
        noise = np.zeros(len(values), dtype=np.float64)
        if matched_training_level.any():
            matched_positions = clipped_positions[matched_training_level]
            current_lows = jitter_lows[matched_positions]
            current_highs = jitter_highs[matched_positions]
            uniform_draws = rng.random(int(matched_training_level.sum()))
            noise[matched_training_level] = (current_lows + uniform_draws*(current_highs - current_lows))
        output_data[column] = (values + noise).astype(np.float64)
    if not np.isfinite(output_data.to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"Count dequantization produced non-finite values for split '{split_name}'.")
    return output_data


def fit_ladis_mode_dequantization(*, train_num_data: pd.DataFrame, excluded_columns: Sequence[str], random_state: int, 
    min_column_top_pct: float = 10.0, min_anchor_pct: float = 1.0, gap_fraction: float = 0.20, 
    boundary_jitter_width: float = 1e-3) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """
    Fit and apply reversible dequantization to dominant repeated numerical
    values that are not already handled as discrete counts.

    Repeated values at the empirical minimum or maximum are spread over a
    fixed one-sided interval outside the observed support. Repeated interior
    values are spread over a smaller interval bounded by their nearest
    neighbouring values.

    The fitted intervals are stored so that validation/test partitions can
    use the same transformation and generated values can later be snapped
    back to their original repeated modes.
    """

    if train_num_data.empty:
        raise ValueError("`train_num_data` cannot be empty.")
    if not 0.0 < min_column_top_pct <= 100.0:
        raise ValueError("`min_column_top_pct` must be inside (0, 100].")
    if not 0.0 < min_anchor_pct <= 100.0:
        raise ValueError("`min_anchor_pct` must be inside (0, 100].")
    if not 0.0 < gap_fraction < 0.5:
        raise ValueError("`gap_fraction` must be inside (0, 0.5).")
    if not 0.0 < boundary_jitter_width < 0.1:
        raise ValueError("`boundary_jitter_width` must be inside (0, 0.1).")
    num_columns = train_num_data.columns.tolist()
    excluded_column_set = set(excluded_columns)
    unexpected_excluded_columns = sorted(excluded_column_set.difference(num_columns))
    if unexpected_excluded_columns:
        raise RuntimeError(f"Some excluded mode-dequantization columns are not present in the numerical schema: {unexpected_excluded_columns}")
    training_vals = train_num_data.to_numpy(dtype=np.float64,)
    if not np.isfinite(training_vals).all():
        raise RuntimeError("The numerical training data contain non-finite values before mode dequantization.")
    dequantized_data = train_num_data.astype(np.float64).copy()
    random_gen = np.random.default_rng(random_state)
    column_info: dict[str, dict[str, Any]] = {}
    diagnostic_records: list[dict[str, Any]] = []
    n_rows = len(train_num_data)
    for column in num_columns:
        column_vals = pd.to_numeric(train_num_data[column], errors="raise").to_numpy(dtype=np.float64)
        val_counts = pd.Series(column_vals).value_counts(sort=True, dropna=False)
        unique_vals = np.sort(np.unique(column_vals))
        top_val = float(val_counts.index[0])
        top_val_n = int(val_counts.iloc[0])
        top_val_pct = float(100.0*top_val_n / n_rows)
        # Constant variables are not dequantized
        sel_column = bool(column not in excluded_column_set and top_val_pct >= min_column_top_pct and len(unique_vals) > 1)
        anchor_records: list[dict[str, Any]] = []
        dequantized_val_n = 0
        if sel_column:
            updated_vals = column_vals.copy()
            for (anchor_val_raw, anchor_count_raw) in val_counts.items():
                anchor_val = float(anchor_val_raw)
                anchor_count = int(anchor_count_raw)
                anchor_pct = float(100.0*anchor_count / n_rows)
                if anchor_pct < min_anchor_pct:
                    continue
                anchor_pos = int(np.searchsorted(unique_vals, anchor_val))
                is_lower_boundary = bool(anchor_pos == 0)
                is_upper_boundary = bool(anchor_pos == len(unique_vals) - 1)
                if is_lower_boundary:
                    # Spread a repeated empirical minimum below the observed support
                    jitter_low = -float(boundary_jitter_width)
                    jitter_high = 0.0
                    interval_type = "lower_boundary"
                elif is_upper_boundary:
                    # Symmetric treatment for a repeated empirical maximum
                    jitter_low = 0.0
                    jitter_high = float(boundary_jitter_width)
                    interval_type = "upper_boundary"
                else:
                    # Interior repeated values must remain between their two nearest genuine neighbours
                    lower_gap = float(anchor_val - unique_vals[anchor_pos - 1])
                    upper_gap = float(unique_vals[anchor_pos + 1] - anchor_val)
                    if (not np.isfinite(lower_gap) or not np.isfinite(upper_gap) or lower_gap <= 0 or upper_gap <= 0):
                        continue
                    half_width = float(gap_fraction*min(lower_gap, upper_gap))
                    if (not np.isfinite(half_width) or half_width <= 0):
                        continue
                    jitter_low = -half_width
                    jitter_high = half_width
                    interval_type = "interior"
                anchor_mask = column_vals == anchor_val
                current_anchor_n = int(anchor_mask.sum())
                if current_anchor_n == 0:
                    continue
                updated_vals[anchor_mask] += random_gen.uniform(low=jitter_low, high=jitter_high, size=current_anchor_n)
                dequantized_val_n += current_anchor_n
                anchor_records.append({"value": anchor_val, "training_n": current_anchor_n, "training_pct": anchor_pct, "interval_type": interval_type,
                    "jitter_low": float(jitter_low), "jitter_high": float(jitter_high), "interval_lower": float(anchor_val+ jitter_low),
                    "interval_upper": float(anchor_val + jitter_high)})
            dequantized_data[column] = updated_vals
        column_info[column] = {"selected": sel_column, "top_value": top_val, "top_value_n": top_val_n, "top_value_pct": top_val_pct, 
            "anchors": anchor_records}
        diagnostic_records.append({"column": column, "excluded_as_discrete_count": (column in excluded_column_set),
            "selected_for_mode_dequantization": sel_column, "training_unique_n_before": int(len(unique_vals)), "top_value": top_val, 
            "top_value_n": top_val_n, "top_value_pct": top_val_pct, "dequantized_anchor_n": len(anchor_records), 
            "dequantized_value_n": (dequantized_val_n), "dequantized_value_pct": float(100.0*dequantized_val_n / n_rows),
            "training_unique_n_after": int(dequantized_data[column].nunique(dropna=True))})
    dequantized_vals = dequantized_data.to_numpy(dtype=np.float64)
    if not np.isfinite(dequantized_vals).all():
        raise RuntimeError("Mode dequantization produced non-finite training values.")
    dequantization_artifact = {"numerical_columns": list(num_columns), "excluded_columns": sorted(excluded_column_set), 
        "random_state": int(random_state), "min_column_top_pct": float(min_column_top_pct), "min_anchor_pct": float(min_anchor_pct),
        "gap_fraction": float(gap_fraction), "boundary_jitter_width": float(boundary_jitter_width), "column_information": column_info}
    diagnostics = (pd.DataFrame(diagnostic_records).sort_values(["selected_for_mode_dequantization", "top_value_pct"],
        ascending=[False, False]).reset_index(drop=True))
    return (dequantized_data, dequantization_artifact, diagnostics)


def apply_ladis_mode_dequantization(*, num_data: pd.DataFrame, dequantization_artifact: Mapping[str, Any], random_state: int,
    split_name: str) -> pd.DataFrame:
    """
    Apply the train-fitted dominant-mode dequantization to one partition.
    """

    expected_columns = list(dequantization_artifact["numerical_columns"])
    if (num_data.columns.tolist() != expected_columns):
        raise RuntimeError(f"The numerical schema of split '{split_name}' differs from the mode-dequantization training schema.")
    dequantized_data = num_data.astype(np.float64).copy()
    random_gen = np.random.default_rng(random_state)
    column_info = dequantization_artifact["column_information"]
    for column in expected_columns:
        info = column_info[column]
        if not bool(info["selected"]):
            continue
        column_vals = pd.to_numeric(dequantized_data[column], errors="raise").to_numpy(dtype=np.float64)
        for anchor_info in info["anchors"]:
            anchor_val = float(anchor_info["value"])
            jitter_low = float(anchor_info["jitter_low"])
            jitter_high = float(anchor_info["jitter_high"])
            matching_tol = np.finfo(np.float64).eps*max(1.0, abs(anchor_val))*32.0
            anchor_mask = np.isclose(column_vals, anchor_val, atol=matching_tol, rtol=0.0)
            matching_n = int(anchor_mask.sum())
            if matching_n:
                column_vals[anchor_mask] += random_gen.uniform(low=jitter_low, high=jitter_high, size=matching_n)
        dequantized_data[column] = column_vals
    if not np.isfinite(dequantized_data.to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"Mode dequantization produced non-finite values for split '{split_name}'.")
    return dequantized_data


def inverse_ladis_mode_dequantization(*, prequantile_num_data: pd.DataFrame, dequantization_artifact: Mapping[str, Any]) -> pd.DataFrame:
    """
    Snap dequantized dominant-mode intervals back to their exact stored values.

    This operation must be performed after inverse Gaussian-quantile
    transformation and before the earlier numerical preprocessing is inverted.
    """

    expected_columns = list(dequantization_artifact["numerical_columns"])
    if (prequantile_num_data.columns.tolist() != expected_columns):
        raise RuntimeError("The inverse mode-dequantization schema differs from the training schema.")
    restored_data = prequantile_num_data.astype(np.float64).copy()
    column_info = (dequantization_artifact["column_information"])
    for column in expected_columns:
        information = column_info[column]
        if not bool(information["selected"]):
            continue
        column_vals = pd.to_numeric(restored_data[column], errors="raise").to_numpy(dtype=np.float64)
        for anchor_info in information["anchors"]:
            anchor_val = float(anchor_info["value"])
            interval_lower = float(anchor_info["interval_lower"])
            interval_upper = float(anchor_info["interval_upper"])
            interval_tol = np.finfo(np.float64).eps*max(1.0, abs(interval_lower), abs(interval_upper))*64.0
            anchor_interval_mask = ((column_vals >= interval_lower - interval_tol) & (column_vals <= interval_upper + interval_tol))
            column_vals[anchor_interval_mask] = anchor_val
        restored_data[column] = column_vals
    if not np.isfinite(restored_data.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Inverse mode dequantization produced non-finite values.")
    return restored_data


def fit_ladis_quantile_transformer(*, train_num_data: pd.DataFrame, random_state: int, n_quantiles: int = 1_000) -> tuple[QuantileTransformer,
    pd.DataFrame, pd.Series]:
    """
    Fit LaDIS's Gaussian quantile transformation on training data only.

    The existing numerical preprocessing removes strong skewness from selected
    variables and applies standardisation. Nevertheless, standardisation does
    not guarantee approximately Gaussian marginal distributions and may leave
    extreme values many standard deviations from zero.

    LaDIS uses a Gaussian quantile transformation as the final numerical
    model-space transformation. The transformation fitted here is placed
    after the existing reversible numerical preprocessing. Because the
    preceding transformations are monotonic, this preserves their ordering
    while producing the approximately Gaussian marginals required by the
    numerical diffusion process.

    Parameters
    ----------
    train_num_data: pandas.DataFrame
        Numerical training representation produced by the existing stored
        preprocessing artifact.
    random_state: int
        Reproducibility seed.
    n_quantiles: int
        Maximum number of quantiles used by the empirical transformation.

    Returns
    -------
    quantile_transformer: sklearn.preprocessing.QuantileTransformer
        Train-fitted Gaussian quantile transformation.
    transf_train_data: pandas.DataFrame
        Numerical training data in definitive LaDIS diffusion space.
    transf_summary: pandas.Series
        Structural and distributional diagnostics.
    """

    if train_num_data.empty:
        raise ValueError("`train_num_data` cannot be empty.")
    if n_quantiles <= 1:
        raise ValueError("`n_quantiles` must be greater than one.")
    num_columns = (train_num_data.columns.tolist())
    train_vals = train_num_data.to_numpy(dtype=np.float64)
    if not np.isfinite(train_vals).all():
        raise RuntimeError("The numerical training representation contains non-finite values before Gaussian quantile transformation.")
    effective_n_quantiles = min(int(n_quantiles), len(train_num_data))
    # Use every training observation when estimating the quantiles
    quantile_transf = QuantileTransformer(n_quantiles=effective_n_quantiles, output_distribution="normal", subsample=len(train_num_data),
        random_state=random_state, copy=True)
    transf_vals = (quantile_transf.fit_transform(train_vals).astype(np.float32, copy=False))
    if not np.isfinite(transf_vals).all():
        raise RuntimeError("The Gaussian quantile transformation produced non-finite training values.")
    transf_train_data = pd.DataFrame(transf_vals, columns=num_columns)
    transf_means = transf_train_data.mean()
    transf_std_devs = transf_train_data.std()
    transf_summary = pd.Series({"training_rows": len(transf_train_data), "numerical_variables": len(num_columns), 
        "requested_quantiles": int(n_quantiles), "effective_quantiles": effective_n_quantiles, "output_distribution": "normal",
        "maximum_absolute_value": float(np.abs(transf_vals).max()), "maximum_absolute_column_mean": float(transf_means.abs().max()),
        "median_column_standard_deviation": float(transf_std_devs.median()), "missing_values": int(transf_train_data.isna().sum().sum()),
        "infinite_values": int(np.isinf(transf_vals).sum())}, name="value")
    return (quantile_transf, transf_train_data, transf_summary)


def transform_ladis_quantile_space(*, num_data: pd.DataFrame, quantile_transf: QuantileTransformer, expected_columns: Sequence[str],
    split_name: str) -> pd.DataFrame:
    """
    Apply a train-fitted Gaussian quantile transformation to one partition.
    """

    expected_columns = list(expected_columns)
    if num_data.columns.tolist() != expected_columns:
        raise RuntimeError(f"The numerical schema of split '{split_name}' differs from the schema used to fit the LaDIS quantile transformation.")
    num_vals = num_data.to_numpy(dtype=np.float64)
    if not np.isfinite(num_vals).all():
        raise RuntimeError(f"The numerical representation of split '{split_name}' contains non-finite values before quantile transformation.")
    transf_vals = quantile_transf.transform(num_vals).astype(np.float32, copy=False)
    if not np.isfinite(transf_vals).all():
        raise RuntimeError(f"The quantile-transformed split '{split_name}' contains non-finite values.")
    return pd.DataFrame(transf_vals, columns=expected_columns)


def inverse_ladis_quantile_space(*, quantile_model_data: pd.DataFrame, quantile_transf: QuantileTransformer,
    expected_columns: Sequence[str]) -> pd.DataFrame:
    """
    Return generated numerical values from Gaussian-quantile space to the
    existing reversible numerical-preprocessing space.
    """

    expected_columns = list(expected_columns)
    if (quantile_model_data.columns.tolist() != expected_columns):
        raise RuntimeError("The generated Gaussian-quantile numerical schema differs from the definitive LaDIS numerical schema.")
    quantile_vals = quantile_model_data.to_numpy(dtype=np.float64)
    if not np.isfinite(quantile_vals).all():
        raise RuntimeError("The generated Gaussian-quantile representation contains non-finite values.")
    inv_vals = (quantile_transf.inverse_transform(quantile_vals))
    if not np.isfinite(inv_vals).all():
        raise RuntimeError("The inverse Gaussian-quantile transformation produced non-finite values.")
    return pd.DataFrame(inv_vals, columns=expected_columns)


def inverse_ladis_model_representation(*, num_model_data: pd.DataFrame, cat_model_data: pd.DataFrame, num_columns: Sequence[str],
    cat_columns: Sequence[str], feature_order: Sequence[str], num_artifact: Mapping[str, Any], cat_artifact: Mapping[str, Any],
    round_disc: bool = True, enforce_nonneg: bool = True) -> pd.DataFrame:
    """
    Return a LaDIS model-space table to the common evaluation space.

    Numerical variables are inverse-transformed to their original scale.
    Categorical integer codes are decoded into protocol labels and port
    tokens. Concrete ports are deliberately not restored here because model
    selection must use the same tokenised representation as CTGAN and TVAE.

    This function will also be reused after candidate sampling.
    """

    num_columns = list(num_columns)
    cat_columns = list(cat_columns)
    feature_order = list(feature_order)
    _validate_dataframe_columns(num_model_data, expected_columns=num_columns, df_name="TabDDPM numerical model data")
    _validate_dataframe_columns(cat_model_data, expected_columns=cat_columns, df_name="TabDDPM categorical model data")
    if len(num_model_data) != len(cat_model_data):
        raise ValueError("The numerical and categorical TabDDPM model tables contain different row counts.")
    inv_num_result = inverse_transform_numerical_features(num_model_data, num_artifact, round_disc=round_disc, 
        enforce_nonneg=enforce_nonneg)
    (restored_num_data, _) = _unpack_dataframe_result(inv_num_result, result_name="inverse_transform_numerical_features")
    restored_num_data = (restored_num_data.loc[:, num_columns].reset_index(drop=True))
    inv_cat_result = inverse_transform_categorical_features(cat_model_data, cat_artifact)
    (restored_cat_data, _) = _unpack_dataframe_result(inv_cat_result, result_name="inverse_transform_categorical_features")
    restored_cat_data = (restored_cat_data.loc[:, cat_columns].reset_index(drop=True))
    restored_eval_data = pd.concat([restored_num_data, restored_cat_data], axis=1)
    restored_eval_data = restored_eval_data.loc[:, feature_order]
    num_vals = restored_eval_data[num_columns].to_numpy(dtype=float)
    if np.isnan(num_vals).any():
        raise RuntimeError("The inverse-transformed TabDDPM numerical data contain missing values.")
    inf_mask = np.isinf(num_vals)
    if inf_mask.any():
        affected_columns = [num_columns[column_index] for column_index in np.flatnonzero(inf_mask.any(axis=0))]
        inf_counts = {column: int(inf_mask[:, num_columns.index(column)].sum()) for column in affected_columns}
        raise RuntimeError("The inverse-transformed TabDDPM numerical data contain infinite values." 
            f"Affected columns and counts: {inf_counts}")
    return restored_eval_data