"""
Persistence and checkpoint-restoration utilities for LaDIS.

The portable LaDIS checkpoint stores everything required to restore the fitted
mixed-type autoencoder, the selected clean-z0 latent diffusion model, and the
train-fitted preprocessing state used during generation.

The richer generation artifact additionally stores model-selection histories,
validation diagnostics, reconstruction information, and metadata required to
reproduce a paper experiment. Saving these diagnostics is optional for users
who only need a reusable trained generator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from .models import CleanLatentDenoiser, CleanLatentGaussianDiffusion, MixedTypeAutoencoder

LADIS_MODEL_CHECKPOINT_KEYS = ["representation_model", "latent_diffusion_model", "model_feature_order", "preprocessing"]
LADIS_GEN_ARTIFACT_KEYS = ["software_versions", "source_database", "traffic_category", "generative_model", "random_state", "model",
    "training_rows", "validation_rows", "test_rows_used", "tuning", "final_training", "validation", "reconstruction_corrections",
    "reconstruction_validation_summary", "derived_feature_validation", "dtype_validation", "model_feature_order", "discrete_columns", "port_columns",
    "standardized_column_order", "standardized_dtypes", "excluded_identifier_columns", "recoverable_constant_columns", "recoverable_constant_values",
    "provenance_columns", "provenance_values", "generated_data_path", "synthetic_rows", "synthetic_columns", "synthetic_dtypes"]


# Environment and checkpoint helpers
def _safe_package_version(package_name: str) -> str | None:
    """Return one installed package version, or ``None`` when unavailable."""

    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def collect_ladis_software_versions() -> dict[str, Any]:
    """Record the principal software versions used by a LaDIS experiment."""

    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "joblib": _safe_package_version("joblib"), "scikit_learn": _safe_package_version("scikit-learn"),
        "sdmetrics": _safe_package_version("sdmetrics")}


def _state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Create an independent CPU copy of one fitted PyTorch state dictionary."""

    state_dict = {parameter_name: parameter_value.detach().cpu().clone() for parameter_name, parameter_value in model.state_dict().items()}
    if not state_dict:
        raise RuntimeError("The fitted model produced an empty state dictionary.")
    return state_dict


def _validate_finite_state_dict(state_dict: Mapping[str, torch.Tensor], *, checkpoint_name: str) -> None:
    """Reject checkpoints containing non-finite floating-point tensors."""

    if not state_dict:
        raise RuntimeError(f"The LaDIS {checkpoint_name} checkpoint is empty.")
    nonfinite_tensors = [parameter_name for parameter_name, parameter_value in state_dict.items() if torch.is_floating_point(parameter_value)
        and not bool(torch.isfinite(parameter_value).all())]
    if nonfinite_tensors:
        raise RuntimeError(f"The LaDIS {checkpoint_name} checkpoint contains non-finite tensors: {nonfinite_tensors}")


# Portable LaDIS model checkpoint
def build_ladis_model_checkpoint(*, autoencoder: MixedTypeAutoencoder, diffusion_model: CleanLatentGaussianDiffusion, ae_conf: Mapping[str, Any],
    diffusion_conf: Mapping[str, Any], selected_diffusion_cand: str, latent_dim: int, cat_cards: Mapping[str, int], num_columns: Sequence[str],
    cat_columns: Sequence[str], model_feature_order: Sequence[str], num_artifact: Mapping[str, Any], num_quantile_transformer: Any,
    cat_gen_artifact: Mapping[str, Any], port_artifact: Mapping[str, Any], count_dequantization_artifact: Mapping[str, Any],
    mode_dequantization_artifact: Mapping[str, Any], latent_standardization_artifact: Mapping[str, Any], binary_columns: Sequence[str] = (),
    binary_val_enabled: bool = False, binary_allowed_vals: Mapping[str, Sequence[Any]] | None = None,
    count_dequantization_diagnostics: pd.DataFrame | None = None, mode_dequantization_diagnostics: pd.DataFrame | None = None,
    quantile_fit_summary: pd.Series | pd.DataFrame | None = None, cat_restriction_summary: pd.DataFrame | None = None,
    latent_standardization_diagnostics: pd.DataFrame | None = None, latent_standardization_summary: pd.Series | None = None,
    latent_standardization_roundtrip: pd.Series | None = None, reconstruction: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """
    Build a portable checkpoint for the definitive fitted LaDIS generator.

    The checkpoint preserves the same trained model components and preprocessing
    state used in the definitive experiments. ``reconstruction`` is optional;
    when supplied, it may contain dataset-specific metadata required to restore
    generated observations beyond the common modeled feature representation.
    """

    num_columns = list(num_columns)
    cat_columns = list(cat_columns)
    feature_order = list(model_feature_order)
    binary_columns = list(binary_columns)
    cat_cards = {column: int(cat_cards[column]) for column in cat_columns}
    binary_allowed_vals = ({} if binary_allowed_vals is None else {str(column): list(values) for column, values in binary_allowed_vals.items()})
    if latent_dim <= 0:
        raise ValueError("`latent_dim` must be positive.")
    if int(autoencoder.latent_dim) != int(latent_dim):
        raise RuntimeError("The autoencoder does not preserve the supplied latent dimension.")
    if int(diffusion_model.latent_dim) != int(latent_dim):
        raise RuntimeError("The diffusion model does not preserve the supplied latent dimension.")
    if set(cat_cards) != set(cat_columns):
        raise RuntimeError("Categorical-cardinality keys do not match the categorical model variables.")
    if set(num_columns).intersection(cat_columns):
        raise RuntimeError("Numerical and categorical model-variable definitions overlap.")
    if set(num_columns).union(cat_columns) != set(feature_order):
        raise RuntimeError("Numerical and categorical variables do not jointly cover the model feature schema.")
    ae_state_dict = _state_dict_to_cpu(autoencoder)
    diffusion_state_dict = _state_dict_to_cpu(diffusion_model)
    _validate_finite_state_dict(ae_state_dict, checkpoint_name="autoencoder")
    _validate_finite_state_dict(diffusion_state_dict, checkpoint_name="latent diffusion")
    latent_columns = [f"z_{dimension:02d}" for dimension in range(int(latent_dim))]
    preprocessing = {"numerical_artifact": num_artifact, "numerical_quantile_transformer": num_quantile_transformer,
        "categorical_generation_artifact": cat_gen_artifact, "port_artifact": port_artifact, 
        "count_dequantization_artifact": count_dequantization_artifact,
        "count_dequantization_diagnostics": (None if count_dequantization_diagnostics is None else count_dequantization_diagnostics.copy()),
        "mode_dequantization_artifact": mode_dequantization_artifact,
        "mode_dequantization_diagnostics": (None if mode_dequantization_diagnostics is None else mode_dequantization_diagnostics.copy()),
        "quantile_fit_summary": (None if quantile_fit_summary is None else quantile_fit_summary.copy()),
        "categorical_restriction_summary": (None if cat_restriction_summary is None else cat_restriction_summary.copy()),
        "latent_standardization_artifact": latent_standardization_artifact,
        "latent_standardization_diagnostics": (None if latent_standardization_diagnostics is None else latent_standardization_diagnostics.copy()),
        "latent_standardization_summary": (None if latent_standardization_summary is None else latent_standardization_summary.copy()),
        "latent_standardization_roundtrip": (None if latent_standardization_roundtrip is None else latent_standardization_roundtrip.copy())}
    checkpoint: dict[str, Any] = {"representation_model": {"model_class": type(autoencoder).__name__, "state_dict": ae_state_dict,
            "configuration": dict(ae_conf), "latent_dimension": int(latent_dim), "categorical_cardinalities": cat_cards, 
            "numerical_columns": num_columns, "categorical_columns": cat_columns, "binary_columns": binary_columns, 
            "binary_validation_enabled": bool(binary_val_enabled), "binary_allowed_values": binary_allowed_vals, "checkpoint_device": "cpu"},
        "latent_diffusion_model": {"model_class": type(diffusion_model).__name__, "state_dict": diffusion_state_dict, 
            "configuration": dict(diffusion_conf), "selected_candidate": str(selected_diffusion_cand), "latent_dimension": int(latent_dim),
            "latent_columns": latent_columns, "checkpoint_device": "cpu"}, "model_feature_order": feature_order, "preprocessing": preprocessing}
    if reconstruction is not None:
        checkpoint["reconstruction"] = dict(reconstruction)
    validate_ladis_model_checkpoint(checkpoint)
    return checkpoint


def validate_ladis_model_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Validate the structural interface of one portable LaDIS checkpoint."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("`checkpoint` must be a mapping.")
    missing_keys = [key for key in LADIS_MODEL_CHECKPOINT_KEYS if key not in checkpoint]
    if missing_keys:
        raise RuntimeError(f"The LaDIS checkpoint is missing required keys: {missing_keys}")
    rep_checkpoint = checkpoint["representation_model"]
    diffusion_checkpoint = checkpoint["latent_diffusion_model"]
    preprocessing = checkpoint["preprocessing"]
    for name, value in {"representation_model": rep_checkpoint, "latent_diffusion_model": diffusion_checkpoint, "preprocessing": preprocessing}.items():
        if not isinstance(value, Mapping):
            raise TypeError(f"Checkpoint entry '{name}' must be a mapping.")
    req_rep_keys = {"state_dict", "configuration", "latent_dimension", "categorical_cardinalities", "numerical_columns", "categorical_columns"}
    req_diffusion_keys = {"state_dict", "configuration", "latent_dimension", "latent_columns"}
    req_preprocessing_keys = {"numerical_artifact", "numerical_quantile_transformer", "categorical_generation_artifact", "port_artifact",
        "count_dequantization_artifact", "mode_dequantization_artifact", "latent_standardization_artifact"}
    missing_rep = sorted(req_rep_keys.difference(rep_checkpoint))
    missing_diff = sorted(req_diffusion_keys.difference(diffusion_checkpoint))
    missing_preproc = sorted(req_preprocessing_keys.difference(preprocessing))
    if missing_rep:
        raise RuntimeError(f"The representation-model checkpoint is incomplete: {missing_rep}")
    if missing_diff:
        raise RuntimeError(f"The latent-diffusion checkpoint is incomplete: {missing_diff}")
    if missing_preproc:
        raise RuntimeError(f"The LaDIS preprocessing checkpoint is incomplete: {missing_preproc}")
    rep_latent_dim = int(rep_checkpoint["latent_dimension"])
    diffusion_latent_dim = int(diffusion_checkpoint["latent_dimension"])
    if rep_latent_dim != diffusion_latent_dim:
        raise RuntimeError("Representation and diffusion checkpoints use different latent dimensions.")
    latent_columns = list(diffusion_checkpoint["latent_columns"])
    expected_latent_columns = [f"z_{dimension:02d}" for dimension in range(rep_latent_dim)]
    if latent_columns != expected_latent_columns:
        raise RuntimeError("The stored latent-column order does not match the checkpoint latent dimension.")
    _validate_finite_state_dict(rep_checkpoint["state_dict"], checkpoint_name="autoencoder")
    _validate_finite_state_dict(diffusion_checkpoint["state_dict"], checkpoint_name="latent diffusion")


def restore_ladis_models(checkpoint: Mapping[str, Any], *, device: torch.device | str) -> tuple[MixedTypeAutoencoder, CleanLatentGaussianDiffusion,
    list[str], list[str], dict[str, int], list[str]]:
    """
    Restore the definitive LaDIS autoencoder and latent diffusion model.

    This is the public equivalent of the checkpoint-restoration logic used in
    the computational-efficiency experiments. The fitted autoencoder is frozen
    after loading because it is used only as a decoder during generation.
    """

    validate_ladis_model_checkpoint(checkpoint)
    device = torch.device(device)
    rep_checkpoint = checkpoint["representation_model"]
    diffusion_checkpoint = checkpoint["latent_diffusion_model"]
    num_columns = list(rep_checkpoint["numerical_columns"])
    cat_columns = list(rep_checkpoint["categorical_columns"])
    stored_cards = rep_checkpoint["categorical_cardinalities"]
    cat_cards = {column: int(stored_cards[column]) for column in cat_columns}
    # Reconstruct the frozen mixed-type autoencoder
    rep_conf = dict(rep_checkpoint["configuration"])
    emb_dims = dict(rep_conf["embedding_dimensions"])
    rep_model = MixedTypeAutoencoder(num_features=len(num_columns), cat_cards=[cat_cards[column] for column in cat_columns],
        cat_emb_dims=[int(emb_dims[column]) for column in cat_columns], latent_dim=int(rep_checkpoint["latent_dimension"]),
        enc_hidden_dims=tuple(int(value) for value in rep_conf["encoder_hidden_dimensions"]),
        dec_hidden_dims=tuple(int(value) for value in rep_conf["decoder_hidden_dimensions"]), dropout=float(rep_conf.get("dropout", 0.0))).to(device)
    rep_model.load_state_dict(rep_checkpoint["state_dict"], strict=True)
    for parameter in rep_model.parameters():
        parameter.requires_grad_(False)
    rep_model.eval()
    # Reconstruct the selected clean-z0 denoiser and Gaussian latent diffusion
    diffusion_conf = dict(diffusion_checkpoint["configuration"])
    latent_denoiser = CleanLatentDenoiser(latent_dim=int(diffusion_checkpoint["latent_dimension"]),
        hidden_dims=tuple(int(value) for value in diffusion_conf["hidden_dimensions"]), 
        timestep_emb_dim=int(diffusion_conf["timestep_embedding_dimension"]), dropout=float(diffusion_conf.get("dropout", 0.0)))
    diffusion_model = CleanLatentGaussianDiffusion(denoiser=latent_denoiser, num_timesteps=int(diffusion_conf["num_timesteps"]),
        scheduler=str(diffusion_conf["scheduler"])).to(device)
    diffusion_model.load_state_dict(diffusion_checkpoint["state_dict"], strict=True)
    diffusion_model.eval()
    latent_columns = list(diffusion_checkpoint["latent_columns"])
    return (rep_model, diffusion_model, num_columns, cat_columns, cat_cards, latent_columns)


def save_ladis_model_checkpoint(checkpoint: Mapping[str, Any], path: str | Path, *, compress: int = 3) -> Path:
    """Validate and save one portable LaDIS model checkpoint with Joblib."""

    validate_ladis_model_checkpoint(checkpoint)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(checkpoint), output_path, compress=compress)
    if not output_path.is_file():
        raise FileNotFoundError(f"The LaDIS checkpoint was not created: {output_path}")
    return output_path


def load_ladis_model_checkpoint(path: str | Path) -> dict[str, Any]:
    """
    Load a portable LaDIS checkpoint.

    For convenience, this function also accepts a full generation-artifact file
    and automatically extracts its ``model`` entry.
    """

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"The LaDIS checkpoint was not found: {input_path}")
    loaded = joblib.load(input_path)
    if not isinstance(loaded, Mapping):
        raise TypeError("The loaded LaDIS checkpoint must be a mapping.")
    checkpoint = loaded["model"] if "model" in loaded else loaded
    validate_ladis_model_checkpoint(checkpoint)
    return dict(checkpoint)


# Complete experimental generation artifact
def build_ladis_generation_artifact(*, model_checkpoint: Mapping[str, Any], source_database: str, traffic_cat: str, random_state: int,
    training_rows: int, val_rows: int, tuning: Mapping[str, Any], final_training: Mapping[str, Any], validation: Mapping[str, Any],
    reconstruction_corrections: pd.DataFrame, reconstruction_val_summary: pd.Series, derived_feature_val: pd.DataFrame, dtype_val: pd.DataFrame,
    discrete_columns: Sequence[str], port_columns: Sequence[str], standardized_column_order: Sequence[str], standardized_dtypes: Mapping[str, str],
    excluded_identifier_columns: Sequence[str], recoverable_constant_columns: Sequence[str], recoverable_constant_vals: Mapping[str, Any],
    provenance_columns: Sequence[str], provenance_vals: Mapping[str, Any], gen_data_path: str | Path, synth_data: pd.DataFrame, test_rows_used: int = 0,
    gen_model: str = "LaDIS", software_versions: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the complete paper-style LaDIS generation artifact."""

    validate_ladis_model_checkpoint(model_checkpoint)
    if training_rows < 0 or val_rows < 0 or test_rows_used < 0:
        raise ValueError("Stored partition row counts cannot be negative.")
    if not isinstance(synth_data, pd.DataFrame):
        raise TypeError("`synthetic_data` must be a pandas DataFrame.")
    artifact = {"software_versions": (collect_ladis_software_versions() if software_versions is None else dict(software_versions)),
        "source_database": str(source_database), "traffic_category": str(traffic_cat), "generative_model": str(gen_model), 
        "random_state": int(random_state), "model": dict(model_checkpoint), "training_rows": int(training_rows), "validation_rows": int(val_rows),
        "test_rows_used": int(test_rows_used), "tuning": dict(tuning), "final_training": dict(final_training), "validation": dict(validation),
        "reconstruction_corrections": reconstruction_corrections.copy(), "reconstruction_validation_summary": reconstruction_val_summary.copy(),
        "derived_feature_validation": derived_feature_val.copy(), "dtype_validation": dtype_val.copy(), 
        "model_feature_order": list(model_checkpoint["model_feature_order"]), "discrete_columns": list(discrete_columns), 
        "port_columns": list(port_columns), "standardized_column_order": list(standardized_column_order),
        "standardized_dtypes": dict(standardized_dtypes), "excluded_identifier_columns": list(excluded_identifier_columns),
        "recoverable_constant_columns": list(recoverable_constant_columns), "recoverable_constant_values": dict(recoverable_constant_vals),
        "provenance_columns": list(provenance_columns), "provenance_values": dict(provenance_vals), "generated_data_path": str(gen_data_path),
        "synthetic_rows": len(synth_data), "synthetic_columns": synth_data.columns.tolist(),
        "synthetic_dtypes": {column: str(dtype) for column, dtype in synth_data.dtypes.items()}}
    validate_ladis_generation_artifact(artifact)
    return artifact


def validate_ladis_generation_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the canonical top-level interface of a LaDIS generation artifact."""

    if not isinstance(artifact, Mapping):
        raise TypeError("`artifact` must be a mapping.")
    observed_keys = list(artifact.keys())
    if observed_keys != LADIS_GEN_ARTIFACT_KEYS:
        raise RuntimeError("The LaDIS generation artifact does not preserve the canonical top-level interface.\n"
            f"Expected: {LADIS_GEN_ARTIFACT_KEYS}\n Observed: {observed_keys}")
    validate_ladis_model_checkpoint(artifact["model"])


def save_ladis_generation_artifact(artifact: Mapping[str, Any], path: str | Path, *, compress: int = 3) -> Path:
    """Validate and save one complete LaDIS generation artifact."""

    validate_ladis_generation_artifact(artifact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(artifact), output_path, compress=compress)
    if not output_path.is_file():
        raise FileNotFoundError(f"The LaDIS generation artifact was not created: {output_path}")
    return output_path


def load_ladis_generation_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate one complete LaDIS generation artifact."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"The LaDIS generation artifact was not found: {input_path}")
    artifact = joblib.load(input_path)
    validate_ladis_generation_artifact(artifact)
    return dict(artifact)


def load_ladis_for_generation(path: str | Path, *, device: torch.device | str) -> tuple[dict[str, Any], MixedTypeAutoencoder,
    CleanLatentGaussianDiffusion, list[str], list[str], dict[str, int], list[str]]:
    """
    Load one saved LaDIS checkpoint or generation artifact and restore its models.

    Returns the portable checkpoint followed by the restored autoencoder,
    diffusion model, numerical columns, categorical columns, categorical
    cardinalities, and latent-column order.
    """

    checkpoint = load_ladis_model_checkpoint(path)
    restored = restore_ladis_models(checkpoint, device=device)
    return (checkpoint, *restored)


# Synthetic-data file I/O
def save_ladis_synthetic_data(synth_data: pd.DataFrame, path: str | Path, *, compression: str = "zstd") -> Path:
    """Save a definitive LaDIS synthetic table as Parquet."""

    if not isinstance(synth_data, pd.DataFrame):
        raise TypeError("`synthetic_data` must be a pandas DataFrame.")
    if synth_data.empty:
        raise ValueError("`synthetic_data` is empty.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synth_data.to_parquet(output_path, index=False, compression=compression)
    if not output_path.is_file():
        raise FileNotFoundError(f"The synthetic LaDIS dataset was not created: {output_path}")
    return output_path


def load_ladis_synthetic_data(path: str | Path) -> pd.DataFrame:
    """Load a synthetic LaDIS Parquet dataset."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"The synthetic LaDIS dataset was not found: {input_path}")
    data = pd.read_parquet(input_path)
    if data.empty:
        raise RuntimeError("The loaded synthetic LaDIS dataset is empty.")
    return data
