"""
Train/validation/test splitting utilities for LaDIS preparation.

This module creates reproducible row partitions while optionally keeping exact
duplicate observations together to prevent direct duplication leakage across
splits.

Split assignment is positional and independent of the DataFrame index. The
input DataFrame is never modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "validation", "test")


def _validate_split_configuration(train_size: float, val_size: float, test_size: float, random_state: int, keep_exact_duplicates_together: bool,
    max_deviation_pct: float) -> None:
    """
    Validate train/validation/test splitting settings.
    """

    split_sizes = {"train_size": train_size, "validation_size": val_size, "test_size": test_size}
    for split_name, split_size in split_sizes.items():
        if not 0.0 < float(split_size) < 1.0:
            raise ValueError(f"`{split_name}` must be between 0 and 1.")
    if not np.isclose(sum(float(size) for size in split_sizes.values()), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("Train, validation, and test proportions must sum to 1.")
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise TypeError("`random_state` must be an integer.")
    if not isinstance(keep_exact_duplicates_together, bool):
        raise TypeError("`keep_exact_duplicates_together` must be Boolean.")
    if not 0.0 <= float(max_deviation_pct) <= 100.0:
        raise ValueError("`max_deviation_pct` must be between 0 and 100.")


def _allocate_target_rows(n_of_rows: int, split_fractions: np.ndarray) -> np.ndarray:
    """
    Convert fractional split sizes into integer target row counts.

    The largest-remainder method is used so that target counts sum exactly to
    the number of observations.
    """

    raw_targets = split_fractions * n_of_rows
    target_rows = np.floor(raw_targets).astype(int)
    rows_remaining = int(n_of_rows - target_rows.sum())
    if rows_remaining > 0:
        fractional_remainders = raw_targets - target_rows
        allocation_order = np.argsort(-fractional_remainders, kind="stable")
        target_rows[allocation_order[:rows_remaining]] += 1
    if np.any(target_rows == 0):
        raise ValueError("The dataset is too small for the requested split proportions to produce three non-empty partitions.")

    return target_rows


def _build_duplicate_groups(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign a positional group identifier to every exact row pattern.
    """

    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    duplicate_group_ids, _ = pd.factorize(row_hashes, sort=False)
    duplicate_group_sizes = np.bincount(duplicate_group_ids)
    # Verify repeated hash groups so that a rare hash collision can never be treated as an exact duplicate
    for group_id in np.flatnonzero(duplicate_group_sizes > 1):
        group_positions = np.flatnonzero(duplicate_group_ids == group_id)
        if len(df.iloc[group_positions].drop_duplicates()) != 1:
            raise RuntimeError("An unexpected row-hash collision was detected while identifying exact duplicates.")
    return (duplicate_group_ids.astype(int), duplicate_group_sizes.astype(int))


def _select_training_coverage_groups(df: pd.DataFrame, group_ids: np.ndarray, group_sizes: np.ndarray, coverage_columns: Sequence[str], *,
    random_state: int) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Select indivisible row groups required to cover categorical training states.

    At least one observation of every state present in each requested variable
    is forced into the training partition. A greedy set-cover strategy minimizes
    the number of required groups while respecting existing duplicate groups.
    """

    if isinstance(coverage_columns, str) or not isinstance(coverage_columns, Sequence):
        raise TypeError("`coverage_columns` must be a sequence of column names.")
    coverage_columns = list(coverage_columns)
    if len(coverage_columns) != len(set(coverage_columns)):
        raise ValueError("`coverage_columns` contains repeated variables.")
    unknown_columns = sorted(set(coverage_columns).difference(df.columns))
    if unknown_columns:
        raise ValueError(f"Training-coverage variables were not found in the DataFrame: {unknown_columns}")
    if not coverage_columns:
        return (np.array([], dtype=int), {"coverage_columns": [], "required_state_n": 0, "forced_group_n": 0, "forced_row_n": 0})
    missing_columns = [column for column in coverage_columns if df[column].isna().any()]
    if missing_columns:
        raise ValueError(f"Categorical training-coverage variables contain missing values: {missing_columns}")
    required_states: set[tuple[int, int]] = set()
    group_states: dict[int, set[tuple[int, int]]] = {group_id: set() for group_id in range(len(group_sizes))}
    # Represent every observed categorical state by a stable integer code
    for column_index, column in enumerate(coverage_columns):
        category_codes, category_values = pd.factorize(df[column], sort=False)
        for category_code in range(len(category_values)):
            required_states.add((column_index, category_code))
        for row_position, category_code in enumerate(category_codes):
            group_id = int(group_ids[row_position])
            group_states[group_id].add((column_index, int(category_code)))
    rng = np.random.default_rng(random_state)
    random_order = rng.permutation(len(group_sizes))
    random_rank = {int(group_id): rank for rank, group_id in enumerate(random_order)}
    uncovered_states = set(required_states)
    selected_groups: list[int] = []
    available_groups = set(range(len(group_sizes)))
    # Greedily favour groups covering many still-unrepresented states; smaller groups and seeded random order provide deterministic tie-breaking
    while uncovered_states:
        candidate_records: list[tuple[int, int, int, int]] = []
        for group_id in available_groups:
            covered_state_n = len(group_states[group_id].intersection(uncovered_states))
            if covered_state_n == 0:
                continue
            candidate_records.append((-covered_state_n, int(group_sizes[group_id]), random_rank[group_id], group_id))
        if not candidate_records:
            raise RuntimeError("Training categorical coverage could not be satisfied.")
        candidate_records.sort()
        selected_group = candidate_records[0][3]
        selected_groups.append(selected_group)
        available_groups.remove(selected_group)
        uncovered_states.difference_update(group_states[selected_group])
    selected_group_array = np.asarray(selected_groups, dtype=int)
    coverage_summary = {"coverage_columns": coverage_columns, "required_state_n": len(required_states), "forced_group_n": len(selected_groups),
        "forced_row_n": int(group_sizes[selected_group_array].sum())}

    return (selected_group_array, coverage_summary)


def _assign_atomic_groups(group_ids: np.ndarray, group_sizes: np.ndarray, target_rows: np.ndarray, *, random_state: int,
    forced_train_group_ids: Sequence[int] = ()) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign indivisible row groups while respecting required training groups.
    """

    rng = np.random.default_rng(random_state)
    forced_train_group_ids = np.asarray(list(forced_train_group_ids), dtype=int)
    if len(forced_train_group_ids) != len(np.unique(forced_train_group_ids)):
        raise ValueError("`forced_train_group_ids` contains repeated groups.")
    if (len(forced_train_group_ids) > 0 and (forced_train_group_ids.min() < 0 or forced_train_group_ids.max() >= len(group_sizes))):
        raise ValueError("`forced_train_group_ids` contains invalid group identifiers.")
    split_counts = np.zeros(len(SPLIT_NAMES), dtype=int)
    group_split_indices = np.full(len(group_sizes), -1, dtype=int)
    # Required categorical states are placed in train before allocating the remaining groups
    if len(forced_train_group_ids) > 0:
        group_split_indices[forced_train_group_ids] = 0
        split_counts[0] = int(group_sizes[forced_train_group_ids].sum())
    forced_group_set = set(forced_train_group_ids.tolist())
    group_order = np.asarray([group_id for group_id in range(len(group_sizes)) if group_id not in forced_group_set], dtype=int)
    rng.shuffle(group_order)
    # Large groups are assigned first; the prior shuffle randomizes ties
    group_order = group_order[np.argsort(-group_sizes[group_order], kind="stable")]
    normalization = np.maximum(target_rows, 1)
    for group_id in group_order:
        group_size = int(group_sizes[group_id])
        candidate_scores: list[float] = []
        for split_index in range(len(SPLIT_NAMES)):
            projected_counts = split_counts.copy()
            projected_counts[split_index] += group_size
            relative_deviation = (projected_counts - target_rows) / normalization
            candidate_scores.append(float(np.sum(relative_deviation**2)))
        selected_split = int(np.argmin(candidate_scores))
        group_split_indices[group_id] = selected_split
        split_counts[selected_split] += group_size
    row_split_indices = group_split_indices[group_ids]
    return (row_split_indices, split_counts)


def split_train_validation_test(df: pd.DataFrame, *, train_size: float = 0.80, val_size: float = 0.10, test_size: float = 0.10, random_state: int = 42,
    keep_exact_duplicates_together: bool = True, max_deviation_pct: float = 0.5, train_coverage_columns: Sequence[str] = ()) -> tuple[dict[str, pd.DataFrame],
    pd.DataFrame, dict[str, Any]]:
    """
    Split a modelling DataFrame into reproducible train, validation, and test sets.

    Exact duplicate observations can optionally be treated as indivisible
    groups so that identical feature vectors cannot appear in different splits.

    Parameters
    ----------
    df: pandas.DataFrame
        Final modelling feature matrix.
    train_size: float, default=0.80
        Requested fraction of observations assigned to training.
    val_size: float, default=0.10
        Requested fraction of observations assigned to validation.
    test_size: float, default=0.10
        Requested fraction of observations assigned to testing.
    random_state: int, default=42
        Seed controlling reproducible row/group assignment.
    keep_exact_duplicates_together: bool, default=True
        Whether exact duplicate feature vectors must remain in the same split.
    max_deviation_pct: float, default=0.5
        Maximum allowed absolute difference, in percentage points, between each
        requested split fraction and its realized row percentage.

    Returns
    -------
    data_splits: dict
        DataFrames keyed by ``train``, ``validation``, and ``test``.
    split_report: pandas.DataFrame
        Requested and realized split-size diagnostics.
    split_artifacts: dict
        Positional split assignment and duplicate-group diagnostics required for
        reproducibility and aligned splitting of other DataFrames.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("`df` must contain at least one observation.")
    _validate_split_configuration(train_size, val_size, test_size, random_state, keep_exact_duplicates_together, max_deviation_pct)
    split_fractions = np.array([train_size, val_size, test_size], dtype=float)
    target_rows = _allocate_target_rows(len(df), split_fractions)
    duplicate_group_ids, duplicate_group_sizes = _build_duplicate_groups(df)
    exact_duplicate_rows = int(np.sum(np.maximum(duplicate_group_sizes - 1, 0)))
    if keep_exact_duplicates_together:
        assignment_group_ids = duplicate_group_ids
        assignment_group_sizes = duplicate_group_sizes
    else:
        assignment_group_ids = np.arange(len(df), dtype=int)
        assignment_group_sizes = np.ones(len(df), dtype=int)
    forced_train_group_ids, coverage_summary = (_select_training_coverage_groups(df, assignment_group_ids, assignment_group_sizes, train_coverage_columns,
        random_state=random_state))
    row_split_indices, actual_rows = _assign_atomic_groups(assignment_group_ids, assignment_group_sizes, target_rows, random_state=random_state,
        forced_train_group_ids=forced_train_group_ids)
    if np.any(actual_rows == 0):
        raise ValueError("The requested split produced an empty partition. Adjust the split proportions or duplicate-group policy.")
    actual_percentages = 100.0 * actual_rows / len(df)
    target_percentages = 100.0 * split_fractions
    deviation_pct_points = np.abs(actual_percentages - target_percentages)
    excessive_deviation = deviation_pct_points > (float(max_deviation_pct) + 1e-12)
    if np.any(excessive_deviation):
        deviations = {SPLIT_NAMES[index]: float(deviation_pct_points[index])
            for index in np.flatnonzero(excessive_deviation)}
        raise ValueError("The realized split proportions exceed the configured maximum deviation in percentage points. "
            f"Observed deviations: {deviations}. This can occur when large duplicate groups must remain together.")
    split_row_positions = {split_name: np.flatnonzero(row_split_indices == split_index).astype(int).tolist()
        for split_index, split_name in enumerate(SPLIT_NAMES)}
    data_splits = {split_name: df.iloc[split_row_positions[split_name]].copy() for split_name in SPLIT_NAMES}
    # Verify that every configured categorical state is represented in train
    training_coverage_violations: dict[str, list[Any]] = {}
    for column in coverage_summary["coverage_columns"]:
        complete_support = set(df[column].drop_duplicates().tolist())
        training_support = set(data_splits["train"][column].drop_duplicates().tolist())
        missing_training_states = [value for value in complete_support if value not in training_support]
        if missing_training_states:
            training_coverage_violations[column] = missing_training_states
    if training_coverage_violations:
        raise RuntimeError(f"The training partition does not cover all required categorical states: {training_coverage_violations}")
    # Check whether any true duplicate pattern spans more than one partition
    duplicate_split_table = pd.DataFrame({"duplicate_group_id": duplicate_group_ids, "split_index": row_split_indices})
    group_split_counts = duplicate_split_table.groupby("duplicate_group_id", sort=False)["split_index"].nunique()
    duplicate_groups_across_splits = int((group_split_counts > 1).sum())
    if keep_exact_duplicates_together and duplicate_groups_across_splits > 0:
        raise RuntimeError("Exact duplicate observations were assigned to multiple splits despite duplicate grouping being enabled.")
    split_report = pd.DataFrame({"split": SPLIT_NAMES, "target_fraction": split_fractions, "target_pct": target_percentages, "target_n": target_rows, "actual_n": actual_rows,
        "actual_pct": actual_percentages, "deviation_pct_points": deviation_pct_points})
    row_split_assignment = [SPLIT_NAMES[split_index] for split_index in row_split_indices]
    duplicate_group_count = int(np.sum(duplicate_group_sizes > 1))
    largest_duplicate_group = int(duplicate_group_sizes.max()) if len(duplicate_group_sizes) > 0 else 0
    split_artifacts = {"source_row_n": int(len(df)), "split_names": list(SPLIT_NAMES),
        "split_fractions": {split_name: float(split_fractions[index]) for index, split_name in enumerate(SPLIT_NAMES)},
        "target_rows": {split_name: int(target_rows[index]) for index, split_name in enumerate(SPLIT_NAMES)},
        "actual_rows": {split_name: int(actual_rows[index]) for index, split_name in enumerate(SPLIT_NAMES)},
        "random_state": int(random_state), "keep_exact_duplicates_together": keep_exact_duplicates_together, "maximum_acceptable_deviation_pct": float(max_deviation_pct),
        "row_split_assignment": row_split_assignment, "split_row_positions": split_row_positions, "exact_duplicate_rows": exact_duplicate_rows,
        "exact_duplicate_group_n": duplicate_group_count, "largest_duplicate_group_n": largest_duplicate_group, "duplicate_groups_across_splits": duplicate_groups_across_splits,
        "train_coverage_columns": coverage_summary["coverage_columns"], "required_training_state_n": coverage_summary["required_state_n"],
        "forced_training_group_n": coverage_summary["forced_group_n"], "forced_training_row_n": coverage_summary["forced_row_n"]}
    return (data_splits, split_report, split_artifacts)


def apply_split_assignment(df: pd.DataFrame, split_artifacts: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """
    Apply an existing positional split assignment to an aligned DataFrame.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame with the same row order and row count as the data originally
        used to create the split.
    split_artifacts: mapping
        Artifacts returned by ``split_train_validation_test``.

    Returns
    -------
    dict
        DataFrames keyed by ``train``, ``validation``, and ``test``.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if not isinstance(split_artifacts, Mapping):
        raise TypeError("`split_artifacts` must be a mapping.")
    required_fields = {"source_row_n", "split_row_positions"}
    missing_fields = required_fields.difference(split_artifacts)
    if missing_fields:
        raise ValueError(f"Split artifacts are missing required fields: {sorted(missing_fields)}")
    source_row_count = int(split_artifacts["source_row_n"])
    if len(df) != source_row_count:
        raise ValueError(f"The aligned DataFrame contains {len(df)} rows, but the split assignment expects {source_row_count}.")
    split_row_positions = split_artifacts["split_row_positions"]
    if not isinstance(split_row_positions, Mapping):
        raise TypeError("`split_row_positions` in the split artifacts must be a mapping.")
    missing_splits = set(SPLIT_NAMES).difference(split_row_positions)
    if missing_splits:
        raise ValueError(f"Split artifacts are missing row positions for: {sorted(missing_splits)}")
    all_positions = [int(position) for split_name in SPLIT_NAMES for position in split_row_positions[split_name]]
    if len(all_positions) != source_row_count or len(set(all_positions)) != source_row_count:
        raise ValueError("Split row positions must assign every source observation exactly once.")
    if sorted(all_positions) != list(range(source_row_count)):
        raise ValueError("Split row positions must cover exactly the positional range of the source DataFrame.")
    data_splits = {split_name: df.iloc[list(split_row_positions[split_name])].copy() for split_name in SPLIT_NAMES}
    return data_splits