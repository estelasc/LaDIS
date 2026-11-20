"""
Storage utilities for reproducible LaDIS preparation bundles.

A preparation bundle stores the final train, validation, and test feature
matrices together with the artifacts, diagnostic reports, and metadata required
to reproduce and reverse dataset preparation.

Tabular partitions are stored as Parquet files. Python preparation artifacts
are serialized with Joblib. A human-readable JSON manifest records bundle
structure, dimensions, data types, and SHA-256 checksums.

Only trusted preparation bundles should be loaded because Joblib serialization
uses Python pickle internally.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .splitting import SPLIT_NAMES


BUNDLE_FORMAT = "ladis_preparation"


def _validate_storage_settings(parquet_compression: str | None, joblib_compression: int, bundle_version: str) -> None:
    """
    Validate preparation-bundle storage settings.
    """

    if parquet_compression is not None and (not isinstance(parquet_compression, str) or not parquet_compression):
        raise ValueError("`parquet_compression` must be a non-empty string or None.")
    if isinstance(joblib_compression, bool) or not isinstance(joblib_compression, (int, np.integer)):
        raise TypeError("`joblib_compression` must be an integer.")
    if not 0 <= int(joblib_compression) <= 9:
        raise ValueError("`joblib_compression` must be between 0 and 9.")
    if not isinstance(bundle_version, str) or not bundle_version:
        raise ValueError("`bundle_version` must be a non-empty string.")


def _validate_data_splits(data_splits: Mapping[str, pd.DataFrame]) -> None:
    """
    Validate prepared train, validation, and test partitions before storage.
    """

    if not isinstance(data_splits, Mapping):
        raise TypeError("`data_splits` must be a mapping.")
    missing_splits = set(SPLIT_NAMES).difference(data_splits)
    unexpected_splits = set(data_splits).difference(SPLIT_NAMES)
    if missing_splits:
        raise ValueError(f"`data_splits` is missing required partitions: {sorted(missing_splits)}")
    if unexpected_splits:
        raise ValueError(f"`data_splits` contains unexpected partitions: {sorted(unexpected_splits)}")
    reference_columns = list(data_splits["train"].columns)
    reference_dtypes = [str(dtype) for dtype in data_splits["train"].dtypes]
    if len(reference_columns) != len(set(reference_columns)):
        raise ValueError("Prepared feature columns must be unique.")
    if any(not isinstance(column, str) for column in reference_columns):
        raise TypeError("Prepared feature column names must be strings.")
    for split_name in SPLIT_NAMES:
        split_data = data_splits[split_name]
        if not isinstance(split_data, pd.DataFrame):
            raise TypeError(f"The '{split_name}' partition must be a pandas DataFrame.")
        if split_data.empty:
            raise ValueError(f"The '{split_name}' partition must not be empty.")
        if list(split_data.columns) != reference_columns:
            raise ValueError(f"The '{split_name}' partition does not have the same feature columns and order as train.")
        split_dtypes = [str(dtype) for dtype in split_data.dtypes]
        if split_dtypes != reference_dtypes:
            raise ValueError(f"The '{split_name}' partition does not have the same feature data types as train.")


def _sha256_file(path: Path) -> str:
    """
    Compute the SHA-256 digest of a file.
    """

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_bundle_file(bundle_path: Path, relative_path: str) -> Path:
    """
    Resolve a manifest file path while preventing directory traversal.
    """

    resolved_bundle_path = bundle_path.resolve()
    resolved_file_path = (bundle_path / relative_path).resolve()
    if resolved_bundle_path != resolved_file_path and resolved_bundle_path not in resolved_file_path.parents:
        raise ValueError(f"Bundle manifest contains an invalid file path: {relative_path}")
    return resolved_file_path


def save_preparation_bundle(output_path: str | Path, data_splits: Mapping[str, pd.DataFrame], *, artifacts: Mapping[str, Any] | None = None,
    reports: Mapping[str, Any] | None = None, metadata: Mapping[str, Any] | None = None, parquet_compression: str | None = "zstd",
    joblib_compression: int = 3, bundle_version: str = "1.0", overwrite: bool = False) -> dict[str, Any]:
    """
    Store prepared LaDIS data and associated reproducibility information.

    The bundle is written first to a temporary directory so that incomplete
    files are not exposed as a valid preparation bundle.

    Parameters
    ----------
    output_path: str or pathlib.Path
        Destination directory for the preparation bundle.
    data_splits: mapping
        Prepared train, validation, and test DataFrames.
    artifacts: mapping, optional
        Preparation artifacts required for reconstruction and reproducibility.
    reports: mapping, optional
        Diagnostic tables and summaries produced during preparation.
    metadata: mapping, optional
        Dataset, configuration, and execution metadata.
    parquet_compression: str or None, default="zstd"
        Compression algorithm used for Parquet partitions.
    joblib_compression: int, default=3
        Joblib compression level between 0 and 9.
    bundle_version: str, default="1.0"
        Preparation-bundle format version.
    overwrite: bool, default=False
        Whether an existing destination bundle may be replaced.

    Returns
    -------
    dict
        Manifest describing the stored preparation bundle.
    """

    _validate_data_splits(data_splits)
    _validate_storage_settings(parquet_compression, joblib_compression, bundle_version)
    if artifacts is None:
        artifacts = {}
    if reports is None:
        reports = {}
    if metadata is None:
        metadata = {}
    if not isinstance(artifacts, Mapping):
        raise TypeError("`artifacts` must be a mapping.")
    if not isinstance(reports, Mapping):
        raise TypeError("`reports` must be a mapping.")
    if not isinstance(metadata, Mapping):
        raise TypeError("`metadata` must be a mapping.")
    if not isinstance(overwrite, bool):
        raise TypeError("`overwrite` must be Boolean.")
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Preparation bundle already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    try:
        split_directory = temporary_path / "splits"
        split_directory.mkdir()
        split_manifest: dict[str, dict[str, Any]] = {}
        # Store final prepared partitions independently for efficient later access
        for split_name in SPLIT_NAMES:
            split_data = data_splits[split_name]
            relative_path = Path("splits") / f"{split_name}.parquet"
            split_path = temporary_path / relative_path
            split_data.to_parquet(split_path, index=True, compression=parquet_compression)
            split_manifest[split_name] = {"file": relative_path.as_posix(), "rows": int(split_data.shape[0]), "columns": int(split_data.shape[1])}
        artifact_path = temporary_path / "artifacts.joblib"
        report_path = temporary_path / "reports.joblib"
        metadata_path = temporary_path / "metadata.joblib"
        joblib.dump(dict(artifacts), artifact_path, compress=int(joblib_compression))
        joblib.dump(dict(reports), report_path, compress=int(joblib_compression))
        joblib.dump(dict(metadata), metadata_path, compress=int(joblib_compression))
        stored_files = [*(split_manifest[split_name]["file"] for split_name in SPLIT_NAMES), "artifacts.joblib", "reports.joblib", "metadata.joblib"]
        checksums = {relative_path: _sha256_file(temporary_path / relative_path) for relative_path in stored_files}
        train_data = data_splits["train"]
        manifest = {"bundle_format": BUNDLE_FORMAT, "bundle_version": bundle_version, "split_names": list(SPLIT_NAMES), "feature_columns": list(train_data.columns),
            "feature_dtypes": {column: str(train_data[column].dtype) for column in train_data.columns}, "splits": split_manifest, "artifacts_file": "artifacts.joblib",
            "reports_file": "reports.joblib", "metadata_file": "metadata.joblib", "files": {relative_path: {"sha256": checksum} for relative_path, checksum in checksums.items()}}
        manifest_path = temporary_path / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        temporary_path.rename(output_path)
    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise
    return manifest


def load_preparation_bundle(bundle_path: str | Path, *, verify_checksums: bool = True, expected_bundle_version: str | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, Any],
    dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Load and validate a stored LaDIS preparation bundle.

    Parameters
    ----------
    bundle_path: str or pathlib.Path
        Preparation-bundle directory.
    verify_checksums: bool, default=True
        Whether stored SHA-256 checksums must be verified before loading.
    expected_bundle_version: str, optional
        Bundle version required by the caller.

    Returns
    -------
    data_splits: dict
        Loaded train, validation, and test DataFrames.
    artifacts: dict
        Stored preparation artifacts.
    reports: dict
        Stored diagnostic reports and summaries.
    metadata: dict
        Stored preparation metadata.
    manifest: dict
        Validated human-readable bundle manifest.
    """

    if not isinstance(verify_checksums, bool):
        raise TypeError("`verify_checksums` must be Boolean.")
    bundle_path = Path(bundle_path)
    if not bundle_path.is_dir():
        raise FileNotFoundError(f"Preparation bundle was not found: {bundle_path}")
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Preparation bundle manifest was not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    required_manifest_fields = {"bundle_format", "bundle_version", "split_names", "feature_columns", "feature_dtypes", "splits", "artifacts_file", "reports_file",
        "metadata_file", "files"}
    missing_manifest_fields = required_manifest_fields.difference(manifest)
    if missing_manifest_fields:
        raise ValueError(f"Preparation bundle manifest is missing required fields: {sorted(missing_manifest_fields)}")
    if manifest["bundle_format"] != BUNDLE_FORMAT:
        raise ValueError(f"Unsupported preparation bundle format: {manifest['bundle_format']}")
    if expected_bundle_version is not None and manifest["bundle_version"] != expected_bundle_version:
        raise ValueError(f"Expected preparation bundle version '{expected_bundle_version}', but found "
            f"'{manifest['bundle_version']}'.")
    if list(manifest["split_names"]) != list(SPLIT_NAMES):
        raise ValueError("Preparation bundle does not contain the expected train/validation/test split structure.")
    # Verify all stored files before deserializing Python objects
    if verify_checksums:
        for relative_path, file_info in manifest["files"].items():
            file_path = _resolve_bundle_file(bundle_path, relative_path)
            if not file_path.is_file():
                raise FileNotFoundError(f"Preparation bundle file is missing: {file_path}")
            observed_checksum = _sha256_file(file_path)
            expected_checksum = file_info["sha256"]
            if observed_checksum != expected_checksum:
                raise ValueError(f"Checksum verification failed for preparation-bundle file: {relative_path}")
    data_splits: dict[str, pd.DataFrame] = {}
    for split_name in SPLIT_NAMES:
        split_info = manifest["splits"].get(split_name)
        if split_info is None:
            raise ValueError(f"Preparation bundle manifest is missing the '{split_name}' partition.")
        split_path = _resolve_bundle_file(bundle_path, split_info["file"])
        if not split_path.is_file():
            raise FileNotFoundError(f"Stored '{split_name}' partition was not found: {split_path}")
        split_data = pd.read_parquet(split_path)
        expected_shape = (int(split_info["rows"]), int(split_info["columns"]))
        if split_data.shape != expected_shape:
            raise ValueError(f"Stored '{split_name}' partition has shape {split_data.shape}, but the manifest "
                f"expects {expected_shape}.")
        if list(split_data.columns) != list(manifest["feature_columns"]):
            raise ValueError(f"Stored '{split_name}' partition does not match the feature columns recorded in the manifest.")
        data_splits[split_name] = split_data
    _validate_data_splits(data_splits)
    expected_dtypes = manifest["feature_dtypes"]
    observed_dtypes = {column: str(data_splits["train"][column].dtype) for column in data_splits["train"].columns}
    if observed_dtypes != expected_dtypes:
        raise ValueError("Stored feature data types do not match the preparation-bundle manifest.")
    artifact_path = _resolve_bundle_file(bundle_path, manifest["artifacts_file"])
    report_path = _resolve_bundle_file(bundle_path, manifest["reports_file"])
    metadata_path = _resolve_bundle_file(bundle_path, manifest["metadata_file"])
    artifacts = joblib.load(artifact_path)
    reports = joblib.load(report_path)
    metadata = joblib.load(metadata_path)
    if not isinstance(artifacts, dict):
        raise TypeError("Stored preparation artifacts must contain a dictionary.")
    if not isinstance(reports, dict):
        raise TypeError("Stored preparation reports must contain a dictionary.")
    if not isinstance(metadata, dict):
        raise TypeError("Stored preparation metadata must contain a dictionary.")
    return (data_splits, artifacts, reports, metadata, manifest)