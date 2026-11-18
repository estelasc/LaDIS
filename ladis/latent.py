"""
Latent-space utilities for LaDIS.

This module freezes the trained autoencoder, encodes mixed-type observations
into the continuous latent space, and fits, applies, and reverses the
training-based latent standardization used by the diffusion model.
"""

from __future__ import annotations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .utils import _validate_ladis_mixed_type_data, _validate_ladis_latent_dataframe
from .models import MixedTypeAutoencoder


def freeze_ladis_autoencoder(*, model: MixedTypeAutoencoder) -> tuple[MixedTypeAutoencoder, pd.Series]:
    """
    Freeze the selected autoencoder before latent diffusion modelling.

    Freezing the autoencoder prevents its parameters from changing during
    diffusion training. Gradients can still propagate through the decoder
    with respect to its latent input, which will later allow decoder-aware
    denoising while keeping the decoder parameters fixed.
    """

    total_params = int(sum(param.numel() for param in model.parameters()))
    trainable_params_before = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    trainable_params_after = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    if trainable_params_after != 0:
        raise RuntimeError("The selected autoencoder still exposes trainable parameters after freezing.")
    freeze_summary = pd.Series({"total_parameters": total_params, "trainable_parameters_before_freezing": trainable_params_before,
        "trainable_parameters_after_freezing": trainable_params_after, "model_in_evaluation_mode": bool(not model.training),
        "autoencoder_frozen": True}, name="value")
    return (model, freeze_summary)


def encode_ladis_partition(*, model: MixedTypeAutoencoder, num_data: pd.DataFrame, cat_data: pd.DataFrame, num_columns: Sequence[str],
    cat_columns: Sequence[str], cat_cards: Mapping[str, int], batch_size: int, device: torch.device, data_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Encode one mixed-type partition into the raw continuous latent space.
    """

    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")
    (num_vals, cat_vals) = _validate_ladis_mixed_type_data(num_data=num_data, cat_data=cat_data, num_columns=num_columns, cat_columns=cat_columns,
        cat_cards=cat_cards, data_name=data_name)
    dataset = TensorDataset(torch.from_numpy(num_vals), torch.from_numpy(cat_vals))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=(device.type == "cuda"))
    model.to(device)
    model.eval()
    latent_batches = []
    with torch.no_grad():
        for (num_batch, cat_batch) in loader:
            num_batch = num_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            cat_batch = cat_batch.to(device=device, dtype=torch.long, non_blocking=True)
            latent_batch = model.encode(num_batch, cat_batch)
            if not torch.isfinite(latent_batch).all().item():
                raise RuntimeError(f"The autoencoder produced non-finite latent values while encoding '{data_name}'.")
            latent_batches.append(latent_batch.detach().cpu().numpy().astype(np.float32, copy=False))
    latent_vals = np.concatenate(latent_batches, axis=0)
    expected_shape = (len(num_data), int(model.latent_dim))
    if latent_vals.shape != expected_shape:
        raise RuntimeError(f"The encoded latent representation of '{data_name}' has shape {latent_vals.shape}, but {expected_shape} was expected.")
    latent_columns = [f"z_{dimension:02d}" for dimension in range(model.latent_dim)]
    latent_data = pd.DataFrame(latent_vals, columns=latent_columns).astype(np.float32)
    latent_stds = np.std(latent_vals.astype(np.float64), axis=0, ddof=0)
    encoding_summary = pd.Series({"partition": data_name, "rows": len(latent_data), "latent_dimension": latent_data.shape[1],
        "latent_dtype": str(latent_data.dtypes.iloc[0]), "minimum_coordinate_std": float(latent_stds.min()), 
        "median_coordinate_std": float(np.median(latent_stds)), "maximum_coordinate_std": float(latent_stds.max()),
        "maximum_absolute_value": float(np.abs(latent_vals).max()), "latent_values_are_finite": True, "encoding_valid": True}, name="value")
    return (latent_data, encoding_summary)


def fit_ladis_latent_standardization(*, train_latent_data: pd.DataFrame, 
    min_coordinate_std: float = 1e-6) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.Series]:
    """
    Fit coordinate-wise latent standardization using training data only.

    The transformation removes the training mean and divides by the training
    standard deviation independently for each latent coordinate. It does not
    whiten, rotate, Gaussianize, or clip the learned latent distribution.
    """

    if min_coordinate_std <= 0.0:
        raise ValueError("`min_coordinate_std` must be positive.")
    latent_columns = train_latent_data.columns.tolist()
    train_vals = _validate_ladis_latent_dataframe(latent_data=train_latent_data, expected_columns=latent_columns, data_name="raw latent training data")
    training_means = np.mean(train_vals, axis=0)
    training_stds = np.std(train_vals, axis=0, ddof=0)
    collapsed_mask = training_stds <= float(min_coordinate_std)
    if collapsed_mask.any():
        collapsed_columns = [latent_columns[column_index] for column_index in np.flatnonzero(collapsed_mask)]
        raise RuntimeError(f"At least one latent coordinate is effectively collapsed before diffusion modelling: {collapsed_columns}")
    standardized_vals = (train_vals - training_means[np.newaxis, :]) / training_stds[np.newaxis, :]
    if not np.isfinite(standardized_vals).all():
        raise RuntimeError("Training latent standardization produced non-finite values.")
    standardized_train_data = pd.DataFrame(standardized_vals.astype(np.float32), columns=latent_columns)
    standardized_check_vals = standardized_train_data.to_numpy(dtype=np.float64)
    standardized_means = np.mean(standardized_check_vals, axis=0)
    standardized_stds = np.std(standardized_check_vals, axis=0, ddof=0)
    diagnostic_records = []
    for column_index, column in enumerate(latent_columns):
        raw_column = train_vals[:, column_index]
        standardized_column = standardized_check_vals[:, column_index]
        diagnostic_records.append({"latent_coordinate": column, "raw_mean": float(training_means[column_index]),
            "raw_std": float(training_stds[column_index]), "raw_min": float(raw_column.min()), "raw_q001": float(np.quantile(raw_column, 0.001)),
            "raw_median": float(np.median(raw_column)), "raw_q999": float(np.quantile(raw_column, 0.999)), "raw_max": float(raw_column.max()),
            "standardized_mean": float(standardized_means[column_index]), "standardized_std": float(standardized_stds[column_index]),
            "standardized_abs_q999": float(np.quantile(np.abs(standardized_column), 0.999)), 
            "standardized_max_abs": float(np.abs(standardized_column).max())})
    diagnostics = pd.DataFrame(diagnostic_records)
    standardization_artifact = {"latent_columns": list(latent_columns), "training_mean": training_means.astype(np.float64),
        "training_std": training_stds.astype(np.float64), "minimum_coordinate_std": float(min_coordinate_std), 
        "fitted_rows": int(len(train_latent_data)), "standardization_type": "coordinatewise_training_mean_std"}
    standardization_summary = pd.Series({"training_rows": len(train_latent_data), "latent_dimension": len(latent_columns),
        "raw_minimum_coordinate_std": float(training_stds.min()), "raw_median_coordinate_std": float(np.median(training_stds)),
        "raw_maximum_coordinate_std": float(training_stds.max()), "standardized_max_abs_coordinate_mean": float(np.abs(standardized_means).max()),
        "standardized_minimum_coordinate_std": float(standardized_stds.min()), "standardized_maximum_coordinate_std": float(standardized_stds.max()),
        "collapsed_coordinates": int(collapsed_mask.sum()), "standardization_fitted_on_training_only": True, "latent_standardization_valid": True},
        name="value")
    return (standardized_train_data, standardization_artifact, diagnostics, standardization_summary)


def transform_ladis_latent_space(*, latent_data: pd.DataFrame, standardization_artifact: Mapping[str, Any], 
    split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply the training-fitted latent standardization to another partition.
    """

    latent_columns = list(standardization_artifact["latent_columns"])
    latent_vals =  _validate_ladis_latent_dataframe(latent_data=latent_data, expected_columns=latent_columns, 
        data_name=(f"{split_name} raw latent data"))
    training_means = np.asarray(standardization_artifact["training_mean"], dtype=np.float64)
    training_stds = np.asarray(standardization_artifact["training_std"], dtype=np.float64)
    expected_shape = (len(latent_columns),)
    if (training_means.shape != expected_shape or training_stds.shape != expected_shape):
        raise RuntimeError("The stored latent-standardization parameters have unexpected shapes.")
    if (not np.isfinite(training_means).all() or not np.isfinite(training_stds).all() or np.any(training_stds <= 0.0)):
        raise RuntimeError("The stored latent-standardization parameters are invalid.")
    standardized_vals = (latent_vals - training_means[np.newaxis, :]) / training_stds[np.newaxis, :]
    if not np.isfinite(standardized_vals).all():
        raise RuntimeError(f"Latent standardization produced non-finite values for '{split_name}'.")
    standardized_data = pd.DataFrame(standardized_vals.astype(np.float32), columns=latent_columns)
    check_vals = standardized_data.to_numpy(dtype=np.float64)
    coordinate_means = np.mean(check_vals, axis=0)
    coordinate_stds = np.std(check_vals, axis=0, ddof=0)
    transf_summary = pd.Series({"partition": split_name, "rows": len(standardized_data), "latent_dimension": len(latent_columns),
        "maximum_absolute_coordinate_mean": float(np.abs(coordinate_means).max()), "minimum_coordinate_std": float(coordinate_stds.min()),
        "median_coordinate_std": float(np.median(coordinate_stds)), "maximum_coordinate_std": float(coordinate_stds.max()),
        "absolute_q999": float(np.quantile(np.abs(check_vals), 0.999)), "maximum_absolute_value": float(np.abs(check_vals).max()),
        "latent_values_are_finite": True, "transformation_valid": True}, name="value")
    return (standardized_data, transf_summary)


def inverse_ladis_latent_standardization(*, standardized_latent_data: pd.DataFrame, standardization_artifact: Mapping[str, Any]) -> pd.DataFrame:
    """
    Return standardized latent vectors to the raw autoencoder latent space.
    """

    latent_columns = list(standardization_artifact["latent_columns"])
    standardized_vals = _validate_ladis_latent_dataframe(latent_data=standardized_latent_data, expected_columns=latent_columns, 
        data_name="standardized latent data")
    training_means = np.asarray(standardization_artifact["training_mean"], dtype=np.float64)
    training_stds = np.asarray(standardization_artifact["training_std"], dtype=np.float64)
    raw_vals = standardized_vals*training_stds[np.newaxis, :] + training_means[np.newaxis, :]
    if not np.isfinite(raw_vals).all():
        raise RuntimeError("Inverse latent standardization produced non-finite values.")
    return pd.DataFrame(raw_vals.astype(np.float32), columns=latent_columns)