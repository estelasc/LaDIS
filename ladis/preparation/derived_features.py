"""
Derived-feature detection and validation utilities for LaDIS preparation.

This module identifies variables that can be reconstructed deterministically
from other modelling variables.

Automatically suggested relationships are never removed solely because of
their names. A variable is removed from modelling only when its reconstruction
rule is explicitly configured or safely detected and the observed data satisfy
the relationship within the configured numerical and coverage thresholds.

The input DataFrame is never modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RATE_NAME_PATTERNS = (r"(?:^|_)(?:pkt|pkts|packet|packets|byte|bytes|bit|bits)_(?:s|sec|secs|second|seconds)$",
    r"(?:pkt|pkts|packet|packets|byte|bytes|bit|bits).*_per_(?:s|sec|secs|second|seconds)$",
    r"(?:^|_)rate(?:_|$)")


def _validate_reconstruction_thresholds(rtol: float, atol: float, min_match_pct: float, min_valid_coverage_pct: float) -> None:
    """
    Validate numerical thresholds used for deterministic reconstruction.
    """

    if rtol < 0:
        raise ValueError("`rtol` must be greater than or equal to zero.")
    if atol < 0:
        raise ValueError("`atol` must be greater than or equal to zero.")
    if not 0.0 <= min_match_pct <= 100.0:
        raise ValueError("`min_match_pct` must be between 0 and 100.")
    if not 0.0 <= min_valid_coverage_pct <= 100.0:
        raise ValueError("`min_valid_coverage_pct` must be between 0 and 100.")


def _validate_required_columns(df: pd.DataFrame, *, target_column: str, source_columns: Sequence[str], rule_name: str) -> None:
    """
    Validate that all variables required by a reconstruction rule exist.
    """

    required_columns = [target_column, *source_columns]
    missing_columns = sorted(set(required_columns).difference(df.columns))
    if missing_columns:
        raise ValueError(f"{rule_name} for '{target_column}' requires missing columns: {missing_columns}")


def identify_rate_candidates(df: pd.DataFrame, *, patterns: Sequence[str] = DEFAULT_RATE_NAME_PATTERNS) -> pd.DataFrame:
    """
    Identify numerical variables whose names suggest rate quantities.

    This function is diagnostic only. Candidate variables are never removed
    automatically and no mathematical relationship is assumed from their names.

    Parameters
    ----------
    df: pandas.DataFrame
        Feature-only DataFrame.
    patterns: sequence of str, optional
        Regular-expression patterns associated with rate-like variable names.

    Returns
    -------
    pandas.DataFrame
        Numerical rate candidates and the name patterns that identified them.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if isinstance(patterns, str):
        raise TypeError("`patterns` must be a sequence of regular-expression strings.")
    compiled_patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    candidate_records: list[dict] = []
    numeric_columns = df.select_dtypes(include=np.number).columns
    for column in numeric_columns:
        matched_patterns = [pattern.pattern for pattern in compiled_patterns if pattern.search(column)]
        if not matched_patterns:
            continue
        candidate_records.append({"column": column, "matched_pattern_n": len(matched_patterns),
            "matched_patterns": "; ".join(matched_patterns)})
    return pd.DataFrame(candidate_records, columns=["column", "matched_pattern_n", "matched_patterns"])


def validate_sum_reconstruction_rules(df: pd.DataFrame, rules: Mapping[str, Sequence[str]], *, rtol: float = 1e-5, atol: float = 1e-8, 
    min_match_pct: float = 99.9, min_valid_coverage_pct: float = 99.9) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Validate deterministic sum reconstruction rules.

    Each rule has the form:
        target = source_1 + source_2 + ... + source_n

    Parameters
    ----------
    df: pandas.DataFrame
        Feature-only DataFrame.
    rules: mapping of str to sequence of str
        Target variables mapped to the source variables whose sum should
        reconstruct them.
    rtol: float, default=1e-5
        Relative numerical tolerance used for row-wise comparison.
    atol: float, default=1e-8
        Absolute numerical tolerance used for row-wise comparison.
    min_match_pct: float, default=99.9
        Minimum percentage of valid observations that must satisfy the rule.
    min_valid_coverage_pct: float, default=99.9
        Minimum percentage of dataset rows on which the rule must be evaluable.

    Returns
    -------
    diagnostics: pandas.DataFrame
        Validation statistics for every configured sum rule.
    approved_rules: dict
        Sum rules approved for deterministic reconstruction.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if not isinstance(rules, Mapping):
        raise TypeError("`rules` must be a mapping from target variables to source-variable sequences.")
    _validate_reconstruction_thresholds(rtol, atol, min_match_pct, min_valid_coverage_pct)
    diagnostic_records: list[dict] = []
    approved_rules: dict[str, dict[str, Any]] = {}
    for target_column, source_columns in rules.items():
        if isinstance(source_columns, str) or not isinstance(source_columns, Sequence):
            raise TypeError(f"Sum rule for '{target_column}' must contain a sequence of source-variable names.")
        source_columns = list(source_columns)
        if not source_columns:
            raise ValueError(f"Sum rule for '{target_column}' must contain at least one source variable.")
        _validate_required_columns(df, target_column=target_column, source_columns=source_columns, rule_name="Sum rule")
        target_values = pd.to_numeric(df[target_column], errors="coerce")
        source_values = df[source_columns].apply(pd.to_numeric, errors="coerce")
        valid_mask = (target_values.notna() & np.isfinite(target_values)
            & source_values.notna().all(axis=1) & np.isfinite(source_values).all(axis=1))
        valid_count = int(valid_mask.sum())
        valid_coverage_pct = (100.0 * valid_count / len(df) if len(df) > 0 else np.nan)
        if valid_count == 0:
            match_pct = np.nan
            mean_abs_error = np.nan
            median_abs_error = np.nan
            q99_abs_error = np.nan
            max_abs_error = np.nan
            approved = False
        else:
            expected_values = source_values.loc[valid_mask].sum(axis=1)
            observed_values = target_values.loc[valid_mask]
            matches = np.isclose(observed_values, expected_values, rtol=rtol, atol=atol)
            absolute_error = (observed_values - expected_values).abs()
            match_pct = float(100.0 * np.mean(matches))
            mean_abs_error = float(absolute_error.mean())
            median_abs_error = float(absolute_error.median())
            q99_abs_error = float(absolute_error.quantile(0.99))
            max_abs_error = float(absolute_error.max())
            approved = bool(match_pct >= min_match_pct and valid_coverage_pct >= min_valid_coverage_pct)
        diagnostic_records.append({"target_column": target_column, "source_columns": source_columns,
            "valid_n": valid_count, "valid_coverage_pct": valid_coverage_pct, "match_pct": match_pct,
            "mean_abs_error": mean_abs_error, "median_abs_error": median_abs_error,
            "q99_abs_error": q99_abs_error, "max_abs_error": max_abs_error,
            "approved_for_reconstruction": approved})
        if approved:
            approved_rules[target_column] = {"operation": "sum", "source_columns": source_columns}
    diagnostics = pd.DataFrame(diagnostic_records, columns=["target_column", "source_columns", "valid_n",
        "valid_coverage_pct", "match_pct", "mean_abs_error", "median_abs_error", "q99_abs_error",
        "max_abs_error", "approved_for_reconstruction"])
    return (diagnostics, approved_rules)


def validate_ratio_reconstruction_rules(df: pd.DataFrame, rules: Mapping[str, Mapping[str, Any]], *, rtol: float = 1e-5, atol: float = 1e-8, 
    min_match_pct: float = 99.9, min_valid_coverage_pct: float = 99.9) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Validate deterministic ratio reconstruction rules.

    Each rule has the form:
        target = scale * numerator / denominator

    Parameters
    ----------
    df: pandas.DataFrame
        Feature-only DataFrame.
    rules: mapping
        Target variables mapped to rule dictionaries containing ``numerator``,
        ``denominator``, and optionally ``scale``.
    rtol: float, default=1e-5
        Relative numerical tolerance used for row-wise comparison.
    atol: float, default=1e-8
        Absolute numerical tolerance used for row-wise comparison.
    min_match_pct: float, default=99.9
        Minimum percentage of valid observations that must satisfy the rule.
    min_valid_coverage_pct: float, default=99.9
        Minimum percentage of dataset rows on which the rule must be evaluable.

    Returns
    -------
    diagnostics: pandas.DataFrame
        Validation statistics for every configured ratio rule.
    approved_rules: dict
        Ratio rules approved for deterministic reconstruction.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if not isinstance(rules, Mapping):
        raise TypeError("`rules` must be a mapping from target variables to ratio-rule mappings.")
    _validate_reconstruction_thresholds(rtol, atol, min_match_pct, min_valid_coverage_pct)
    diagnostic_records: list[dict] = []
    approved_rules: dict[str, dict[str, Any]] = {}
    for target_column, rule in rules.items():
        if not isinstance(rule, Mapping):
            raise TypeError(f"Ratio rule for '{target_column}' must be a mapping.")
        required_rule_fields = {"numerator", "denominator"}
        missing_rule_fields = sorted(required_rule_fields.difference(rule))
        if missing_rule_fields:
            raise ValueError(f"Ratio rule for '{target_column}' is missing required fields: {missing_rule_fields}")
        numerator_column = str(rule["numerator"])
        denominator_column = str(rule["denominator"])
        scale = float(rule.get("scale", 1.0))
        if not np.isfinite(scale):
            raise ValueError(f"Ratio-rule scale for '{target_column}' must be finite.")
        _validate_required_columns(df, target_column=target_column, source_columns=[numerator_column, denominator_column], rule_name="Ratio rule")
        target_values = pd.to_numeric(df[target_column], errors="coerce")
        numerator_values = pd.to_numeric(df[numerator_column], errors="coerce")
        denominator_values = pd.to_numeric(df[denominator_column], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            expected_values = scale * numerator_values / denominator_values
        valid_mask = (target_values.notna() & numerator_values.notna() & denominator_values.notna() & np.isfinite(target_values) 
            & np.isfinite(numerator_values) & np.isfinite(denominator_values) & denominator_values.ne(0) & np.isfinite(expected_values))
        valid_count = int(valid_mask.sum())
        valid_coverage_pct = (100.0 * valid_count / len(df) if len(df) > 0 else np.nan)
        if valid_count == 0:
            match_pct = np.nan
            mean_abs_error = np.nan
            median_abs_error = np.nan
            q99_abs_error = np.nan
            max_abs_error = np.nan
            approved = False
        else:
            observed_values = target_values.loc[valid_mask]
            reconstructed_values = expected_values.loc[valid_mask]
            matches = np.isclose(observed_values, reconstructed_values, rtol=rtol, atol=atol)
            absolute_error = (observed_values - reconstructed_values).abs()
            match_pct = float(100.0 * np.mean(matches))
            mean_abs_error = float(absolute_error.mean())
            median_abs_error = float(absolute_error.median())
            q99_abs_error = float(absolute_error.quantile(0.99))
            max_abs_error = float(absolute_error.max())
            approved = bool(match_pct >= min_match_pct and valid_coverage_pct >= min_valid_coverage_pct)
        diagnostic_records.append({"target_column": target_column, "numerator_column": numerator_column, "denominator_column": denominator_column, 
            "scale": scale, "valid_n": valid_count, "valid_coverage_pct": valid_coverage_pct, "match_pct": match_pct, "mean_abs_error": mean_abs_error, 
            "median_abs_error": median_abs_error, "q99_abs_error": q99_abs_error, "max_abs_error": max_abs_error, 
            "approved_for_reconstruction": approved})
        if approved:
            approved_rules[target_column] = {"operation": "ratio", "numerator": numerator_column, "denominator": denominator_column, "scale": scale}
    diagnostics = pd.DataFrame(diagnostic_records, columns=["target_column", "numerator_column", "denominator_column", "scale", "valid_n", 
        "valid_coverage_pct", "match_pct", "mean_abs_error", "median_abs_error", "q99_abs_error", "max_abs_error", "approved_for_reconstruction"])
    return (diagnostics, approved_rules)


def find_std_var_pairs(df: pd.DataFrame, *, std_suffix: str = "_std", var_suffix: str = "_var", rtol: float = 1e-5, atol: float = 1e-8, 
    min_match_pct: float = 99.9, min_valid_coverage_pct: float = 99.9) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Detect and validate standard-deviation / variance relationships.

    Candidate pairs are identified from configurable suffixes and validated
    against the deterministic relationship:
        variance = standard_deviation ** 2

    A matching name alone is never sufficient for approval.

    Parameters
    ----------
    df: pandas.DataFrame
        Feature-only DataFrame.
    std_suffix: str, default="_std"
        Suffix identifying standard-deviation variables.
    var_suffix: str, default="_var"
        Suffix identifying variance variables.
    rtol: float, default=1e-5
        Relative numerical tolerance used for row-wise comparison.
    atol: float, default=1e-8
        Absolute numerical tolerance used for row-wise comparison.
    min_match_pct: float, default=99.9
        Minimum percentage of valid observations that must satisfy the rule.
    min_valid_coverage_pct: float, default=99.9
        Minimum percentage of dataset rows on which the rule must be evaluable.

    Returns
    -------
    diagnostics: pandas.DataFrame
        Validation statistics for every detected standard-deviation / variance
        pair.
    approved_rules: dict
        Variance rules approved for deterministic reconstruction.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if not isinstance(std_suffix, str) or not std_suffix:
        raise ValueError("`std_suffix` must be a non-empty string.")
    if not isinstance(var_suffix, str) or not var_suffix:
        raise ValueError("`var_suffix` must be a non-empty string.")
    if std_suffix == var_suffix:
        raise ValueError("`std_suffix` and `var_suffix` must be different.")
    _validate_reconstruction_thresholds(rtol, atol, min_match_pct, min_valid_coverage_pct)
    diagnostic_records: list[dict] = []
    approved_rules: dict[str, dict[str, Any]] = {}
    std_columns = [column for column in df.columns if column.endswith(std_suffix)]
    for std_column in std_columns:
        base_name = std_column[:-len(std_suffix)]
        var_column = base_name + var_suffix
        if var_column not in df.columns:
            continue
        std_values = pd.to_numeric(df[std_column], errors="coerce")
        var_values = pd.to_numeric(df[var_column], errors="coerce")
        valid_mask = (std_values.notna() & var_values.notna() & np.isfinite(std_values) & np.isfinite(var_values))
        valid_count = int(valid_mask.sum())
        valid_coverage_pct = (100.0 * valid_count / len(df) if len(df) > 0 else np.nan)
        if valid_count == 0:
            match_pct = np.nan
            negative_variance_pct = np.nan
            mean_abs_error = np.nan
            median_abs_error = np.nan
            q99_abs_error = np.nan
            max_abs_error = np.nan
            approved = False
        else:
            expected_variance = std_values.loc[valid_mask] ** 2
            observed_variance = var_values.loc[valid_mask]
            matches = np.isclose(observed_variance, expected_variance, rtol=rtol, atol=atol)
            absolute_error = (observed_variance - expected_variance).abs()
            match_pct = float(100.0 * np.mean(matches))
            negative_variance_pct = float(100.0 * np.mean(observed_variance < 0))
            mean_abs_error = float(absolute_error.mean())
            median_abs_error = float(absolute_error.median())
            q99_abs_error = float(absolute_error.quantile(0.99))
            max_abs_error = float(absolute_error.max())
            approved = bool(match_pct >= min_match_pct and valid_coverage_pct >= min_valid_coverage_pct)
        diagnostic_records.append({"base_name": base_name, "std_column": std_column, "variance_column": var_column, "valid_n": valid_count, 
            "valid_coverage_pct": valid_coverage_pct, "match_pct": match_pct, "negative_variance_pct": negative_variance_pct,
            "mean_abs_error": mean_abs_error, "median_abs_error": median_abs_error, "q99_abs_error": q99_abs_error, "max_abs_error": max_abs_error,
            "approved_for_reconstruction": approved})
        if approved:
            approved_rules[var_column] = {"operation": "square", "source_column": std_column}
    diagnostics = pd.DataFrame(diagnostic_records, columns=["base_name", "std_column", "variance_column", "valid_n", "valid_coverage_pct", "match_pct", 
        "negative_variance_pct", "mean_abs_error", "median_abs_error", "q99_abs_error", "max_abs_error", "approved_for_reconstruction"])
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values(by=["match_pct", "median_abs_error"],
            ascending=[False, True]).reset_index(drop=True)
    return (diagnostics, approved_rules)


def _get_rule_dependencies(rule: Mapping[str, Any]) -> list[str]:
    """
    Extract source-variable dependencies from an approved reconstruction rule.
    """

    operation = rule["operation"]
    if operation == "sum":
        return list(rule["source_columns"])
    if operation == "ratio":
        return [rule["numerator"], rule["denominator"]]
    if operation == "square":
        return [rule["source_column"]]
    raise ValueError(f"Unsupported reconstruction operation: {operation}")


def _resolve_reconstruction_order(rules: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """
    Determine a valid deterministic reconstruction order.

    Dependencies between derived variables are resolved topologically so that a
    reconstructed variable can itself be used to reconstruct another variable.
    """

    target_columns = list(rules)
    target_set = set(target_columns)
    derived_dependencies = {target: set(_get_rule_dependencies(rule)).intersection(target_set)
        for target, rule in rules.items()}
    reconstruction_order: list[str] = []
    unresolved_targets = target_columns.copy()
    while unresolved_targets:
        ready_targets = [target for target in unresolved_targets if derived_dependencies[target].issubset(reconstruction_order)]
        if not ready_targets:
            cyclic_targets = {target: sorted(derived_dependencies[target]) for target in unresolved_targets}
            raise ValueError(f"Derived-feature reconstruction rules contain a cyclic dependency: {cyclic_targets}")
        for target in ready_targets:
            reconstruction_order.append(target)
            unresolved_targets.remove(target)
    return reconstruction_order


def build_derived_feature_set(df: pd.DataFrame, *, sum_rules: Mapping[str, Sequence[str]] | None = None, 
    rate_rules: Mapping[str, Mapping[str, Any]] | None = None, detect_std_var: bool = True, std_suffix: str = "_std", var_suffix: str = "_var", 
    rtol: float = 1e-5, atol: float = 1e-8, min_match_pct: float = 99.9, 
    min_valid_coverage_pct: float = 99.9) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Validate reconstructable derived variables and remove approved targets.

    Explicit sum and rate rules are validated alongside optional automatic
    standard-deviation / variance detection. Only relationships that satisfy
    the configured numerical and coverage thresholds are approved.

    Parameters
    ----------
    df: pandas.DataFrame
        Feature-only DataFrame.
    sum_rules: mapping, optional
        Explicit deterministic sum relationships.
    rate_rules: mapping, optional
        Explicit deterministic ratio relationships.
    detect_std_var: bool, default=True
        Whether to search for suffix-based standard-deviation / variance pairs.
    std_suffix: str, default="_std"
        Standard-deviation suffix used for automatic pair detection.
    var_suffix: str, default="_var"
        Variance suffix used for automatic pair detection.
    rtol: float, default=1e-5
        Relative numerical tolerance for deterministic validation.
    atol: float, default=1e-8
        Absolute numerical tolerance for deterministic validation.
    min_match_pct: float, default=99.9
        Minimum percentage of valid observations satisfying a rule.
    min_valid_coverage_pct: float, default=99.9
        Minimum percentage of dataset rows on which a rule must be evaluable.

    Returns
    -------
    feature_df: pandas.DataFrame
        Feature matrix after approved reconstructable variables are removed.
    derived_feature_report: pandas.DataFrame
        Combined diagnostics for all candidate reconstruction rules.
    derived_artifacts: dict
        Approved rules, removed variables, reconstruction order, and retained
        feature names required for reproducibility and later reconstruction.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if sum_rules is None:
        sum_rules = {}
    if rate_rules is None:
        rate_rules = {}
    if not isinstance(detect_std_var, bool):
        raise TypeError("`detect_std_var` must be Boolean.")
    sum_diagnostics, approved_sum_rules = validate_sum_reconstruction_rules(df, sum_rules, rtol=rtol, atol=atol, min_match_pct=min_match_pct, 
        min_valid_coverage_pct=min_valid_coverage_pct)
    rate_diagnostics, approved_rate_rules = validate_ratio_reconstruction_rules(df, rate_rules, rtol=rtol, atol=atol, min_match_pct=min_match_pct, 
        min_valid_coverage_pct=min_valid_coverage_pct)
    if detect_std_var:
        std_var_diagnostics, approved_std_var_rules = find_std_var_pairs(df, std_suffix=std_suffix, var_suffix=var_suffix, rtol=rtol, atol=atol, 
            min_match_pct=min_match_pct, min_valid_coverage_pct=min_valid_coverage_pct)
    else:
        std_var_diagnostics = pd.DataFrame()
        approved_std_var_rules = {}
    approved_rule_groups = {"sum": approved_sum_rules, "rate": approved_rate_rules, "std_var": approved_std_var_rules}
    duplicate_targets: set[str] = set()
    seen_targets: set[str] = set()
    for rule_group in approved_rule_groups.values():
        group_targets = set(rule_group)
        duplicate_targets.update(seen_targets.intersection(group_targets))
        seen_targets.update(group_targets)
    if duplicate_targets:
        raise ValueError(f"Multiple reconstruction rules were approved for the same target variables: {sorted(duplicate_targets)}")
    reconstruction_rules: dict[str, dict[str, Any]] = {}
    for rule_group in approved_rule_groups.values():
        reconstruction_rules.update(rule_group)
    reconstruction_order = _resolve_reconstruction_order(reconstruction_rules)
    report_frames: list[pd.DataFrame] = []
    for rule_family, diagnostics in [("sum", sum_diagnostics), ("rate", rate_diagnostics),
        ("std_var", std_var_diagnostics)]:
        if diagnostics.empty:
            continue
        diagnostics = diagnostics.copy()
        diagnostics.insert(0, "rule_family", rule_family)
        report_frames.append(diagnostics)
    if report_frames:
        derived_feature_report = pd.concat(report_frames, ignore_index=True, sort=False)
    else:
        derived_feature_report = pd.DataFrame()
    # Preserve the original feature order independently of reconstruction order
    removed_derived_columns = [column for column in df.columns if column in reconstruction_rules]
    retained_feature_columns = [column for column in df.columns if column not in reconstruction_rules]
    feature_df = df.loc[:, retained_feature_columns].copy()
    derived_artifacts = {"removed_derived_columns": removed_derived_columns, "retained_feature_columns": retained_feature_columns, 
        "reconstruction_rules": reconstruction_rules, "reconstruction_order": reconstruction_order, "approved_sum_rules": approved_sum_rules,
        "approved_rate_rules": approved_rate_rules, "approved_std_var_rules": approved_std_var_rules}
    return (feature_df, derived_feature_report, derived_artifacts)