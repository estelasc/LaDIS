"""
Evaluation and diagnostic utilities for LaDIS.

This module evaluates synthetic-data quality and model behavior throughout the
LaDIS pipeline, including autoencoder reconstruction, latent space structure,
diffusion denoising, preprocessing validity, and SDMetrics-based fidelity.
"""

from __future__ import annotations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset
from sdmetrics.reports import QualityReport
from sklearn.preprocessing import QuantileTransformer

from .utils import _validate_ladis_mixed_type_data, _validate_ladis_latent_dataframe
from .models import MixedTypeAutoencoder, CleanLatentGaussianDiffusion
from .latent import inverse_ladis_latent_standardization
from .preprocessing import inverse_transform_numerical_features, inverse_transform_categorical_features, inverse_ladis_quantile_space


# Synthetic-data evaluation
def build_sdmetrics_metadata(*, feature_order: Sequence[str], cat_columns: Sequence[str]) -> dict:
    """
    Build SDMetrics single-table metadata for the model representation.

    Parameters
    ----------
    feature_order: sequence of str
        Complete feature-column order.
    cat_columns: sequence of str
        Variables evaluated as categorical.

    Returns
    -------
    dict
        SDMetrics-compatible metadata dictionary.
    """

    feature_order = list(feature_order)
    cat_columns = set(cat_columns)
    return {"columns": {column: {"sdtype": ("categorical" if column in cat_columns else "numerical")} for column in feature_order}}


def evaluate_sample(*, real_val_data: pd.DataFrame, synth_data: pd.DataFrame, fitted_training_data: pd.DataFrame,
    metadata: dict, cat_columns: Sequence[str], candidate_name: str) -> tuple[pd.Series, pd.DataFrame, QualityReport]:
    """
    Evaluate a sample against held-out validation data.

    The primary model-selection criterion is the SDMetrics Quality Report.
    Additional diagnostics detect:
        - invalid categorical levels
        - missing or infinite values
        - synthetic duplicate rows
        - exact row matches with fitted training data
        - numerical values outside the fitted training range

    Returns
    -------
    summary: pandas.Series
        Candidate-level evaluation summary.
    property_scores: pandas.DataFrame
        SDMetrics property scores.
    report: sdmetrics QualityReport
        Generated report object.
    """

    expected_columns = (real_val_data.columns.tolist())
    if synth_data.columns.tolist() != expected_columns:
        raise ValueError("The synthetic CTGAN schema differs from the validation schema.")
    if len(synth_data) != len(real_val_data):
        raise ValueError("CTGAN evaluation requires equal-sized real and synthetic validation datasets.")
    cat_columns = list(cat_columns)
    num_columns = [column for column in expected_columns if column not in cat_columns]
    # SDMetrics statistical-fidelity evaluation
    quality_report = QualityReport()
    quality_report.generate(real_data=real_val_data, synthetic_data=synth_data, metadata=metadata, verbose=False)
    overall_quality_score = float(quality_report.get_score())
    property_scores = (quality_report.get_properties().copy())
    property_score_lookup = (property_scores.set_index("Property")["Score"].to_dict())
    # Structural validity
    missing_n = int(synth_data.isna().sum().sum())
    synth_num_vals = (synth_data[num_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
    inf_n = int(np.isinf(synth_num_vals).sum())
    invalid_cat_n = 0
    for column in cat_columns:
        known_cats = set(fitted_training_data[column].unique())
        invalid_cat_n += int((~synth_data[column].isin(known_cats)).sum())
    # Duplication and exact-training-match diagnostics
    synth_duplicate_n = int(synth_data.duplicated().sum())
    synth_duplicate_pct = (100.0*synth_duplicate_n / max(len(synth_data), 1))
    training_hashes = np.unique(pd.util.hash_pandas_object(fitted_training_data, index=False).to_numpy())
    synth_hashes = (pd.util.hash_pandas_object(synth_data, index=False).to_numpy())
    exact_training_match_n = int(np.isin(synth_hashes, training_hashes).sum())
    exact_training_match_pct = (100.0*exact_training_match_n / max(len(synth_data), 1))
    # Numerical range diagnostics
    out_of_training_range_n = 0
    for column in num_columns:
        training_vals = pd.to_numeric(fitted_training_data[column], errors="coerce")
        synth_vals = pd.to_numeric(synth_data[column], errors="coerce")
        training_min = training_vals.min()
        training_max = training_vals.max()
        out_of_training_range_n += int(((synth_vals < training_min) | (synth_vals > training_max)).sum())
    summary = pd.Series({"candidate": candidate_name, "quality_score": overall_quality_score, 
        "column_shapes_score": property_score_lookup.get("Column Shapes", np.nan), 
        "column_pair_trends_score": property_score_lookup.get("Column Pair Trends", np.nan), "evaluation_rows": len(synth_data), "missing_n": missing_n,
        "infinite_n": inf_n, "invalid_categorical_n": invalid_cat_n, "synthetic_duplicate_n": synth_duplicate_n, 
        "synthetic_duplicate_pct": synth_duplicate_pct, "exact_training_match_n": exact_training_match_n,
        "exact_training_match_pct": exact_training_match_pct, "out_of_training_range_n": out_of_training_range_n}, name="value")
    return (summary, property_scores, quality_report)


# Autoencoder evaluation
def reconstruct_ladis_autoencoder_partition(*, model: MixedTypeAutoencoder, num_data: pd.DataFrame, cat_data: pd.DataFrame, num_columns: Sequence[str],
    cat_columns: Sequence[str], cat_cards: Mapping[str, int], batch_size: int, device: torch.device, 
    data_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct one partition in autoencoder numerical/code space.
    """

    (num_vals, cat_vals) = _validate_ladis_mixed_type_data(num_data=num_data, cat_data=cat_data, num_columns=num_columns, cat_columns=cat_columns,
        cat_cards=cat_cards, data_name=data_name)
    dataset = TensorDataset(torch.from_numpy(num_vals), torch.from_numpy(cat_vals))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=(device.type == "cuda"))
    num_batches = []
    cat_batches = []
    model.eval()
    with torch.no_grad():
        for (num_batch, cat_batch) in loader:
            num_batch = num_batch.to(device=device, dtype=torch.float32)
            cat_batch = cat_batch.to(device=device, dtype=torch.long)
            (_, num_rec, cat_logits) = model(num_batch, cat_batch)
            num_batches.append(num_rec.cpu().numpy())
            cat_preds = torch.stack([logits.argmax(dim=1) for logits in cat_logits], dim=1)
            cat_batches.append(cat_preds.cpu().numpy())
    rec_num_data = pd.DataFrame(np.concatenate(num_batches, axis=0), columns=list(num_columns)).astype(np.float32)
    rec_cat_data = pd.DataFrame(np.concatenate(cat_batches, axis=0), columns=list(cat_columns)).astype(np.int64)
    return (rec_num_data, rec_cat_data)


def evaluate_ladis_autoencoder_reconstruction(*, model: MixedTypeAutoencoder, num_data: pd.DataFrame, cat_data: pd.DataFrame, 
    num_columns: Sequence[str], cat_columns: Sequence[str], cat_cards: Mapping[str, int], batch_size: int, device: torch.device, 
    split_name: str) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate numerical and categorical autoencoder reconstruction fidelity.
    """

    (rec_num_data, rec_cat_data) = reconstruct_ladis_autoencoder_partition(model=model, num_data=num_data, cat_data=cat_data, num_columns=num_columns,
        cat_columns=cat_columns, cat_cards=cat_cards, batch_size=batch_size, device=device, data_name=split_name)
    ref_num = num_data.to_numpy(dtype=np.float64)
    rec_num = rec_num_data.to_numpy(dtype=np.float64)
    errors = rec_num - ref_num
    num_records = []
    for column_index, column in enumerate(num_columns):
        column_error = errors[:, column_index]
        ref_column = ref_num[:, column_index]
        rec_column = rec_num[:, column_index]
        mse = float(np.mean(np.square(column_error)))
        ref_std = float(np.std(ref_column))
        rec_std = float(np.std(rec_column))
        if (ref_std > 0.0 and rec_std > 0.0):
            correlation = float(np.corrcoef(ref_column, rec_column)[0, 1])
        else:
            correlation = np.nan
        num_records.append({"split": split_name, "column": column, "mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(column_error))),
            "mean_error": float(np.mean(column_error)), "reference_std": ref_std, "reconstructed_std": rec_std, "correlation": correlation})
    num_diagnostics = pd.DataFrame(num_records)
    ref_cat = cat_data.to_numpy(dtype=np.int64)
    rec_cat = rec_cat_data.to_numpy(dtype=np.int64)
    cat_records = []
    for column_index, column in enumerate(cat_columns):
        ref_codes = ref_cat[:, column_index]
        pred_codes = rec_cat[:, column_index]
        cat_records.append({"split": split_name, "column": column, "cardinality": int(cat_cards[column]), 
            "accuracy": float(np.mean(pred_codes == ref_codes)), "reference_observed_categories": int(np.unique(ref_codes).size),
            "predicted_observed_categories": int(np.unique(pred_codes).size)})
    cat_diagnostics = pd.DataFrame(cat_records)
    finite_corrs = num_diagnostics["correlation"].replace([np.inf, -np.inf], np.nan).dropna()
    global_mse = float(np.mean(np.square(errors)))
    summary = pd.Series({"split": split_name, "rows": len(num_data), "numerical_global_mse": global_mse, 
        "numerical_global_rmse": float(np.sqrt(global_mse)), "numerical_global_mae": float(np.mean(np.abs(errors))),
        "numerical_median_column_rmse": float(num_diagnostics["rmse"].median()), "numerical_maximum_column_rmse": float(num_diagnostics["rmse"].max()),
        "numerical_median_correlation": float(finite_corrs.median()), "categorical_macro_accuracy": float(cat_diagnostics["accuracy"].mean()),
        "categorical_minimum_accuracy": float(cat_diagnostics["accuracy"].min()), 
        "categorical_exact_row_accuracy": float(np.mean(np.all(rec_cat == ref_cat, axis=1))),
        "reconstructed_numerical_values_are_finite": bool(np.isfinite(rec_num).all()), "reconstruction_valid": True}, name="value")
    return (summary, num_diagnostics, cat_diagnostics)


# Latent space evaluation
def validate_ladis_latent_standardization_roundtrip(*, raw_latent_data: pd.DataFrame, standardized_latent_data: pd.DataFrame,
    standardization_artifact: Mapping[str, Any], split_name: str, abs_tol: float = 1e-5, rel_tol: float = 1e-5) -> pd.Series:
    """
    Validate standardized-latent -> raw-latent inversion.
    """

    if abs_tol < 0.0:
        raise ValueError("`abs_tol` cannot be negative.")
    if rel_tol < 0.0:
        raise ValueError("`rel_tol` cannot be negative.")
    latent_columns = list(standardization_artifact["latent_columns"])
    ref_vals = _validate_ladis_latent_dataframe(latent_data=raw_latent_data, expected_columns=latent_columns, 
        data_name=(f"{split_name} raw latent reference"))
    recovered_data = inverse_ladis_latent_standardization(standardized_latent_data=standardized_latent_data, 
        standardization_artifact=standardization_artifact)
    recovered_vals = recovered_data.to_numpy(dtype=np.float64)
    abs_errs = np.abs(recovered_vals - ref_vals)
    close_mask = np.isclose(recovered_vals, ref_vals, atol=abs_tol, rtol=rel_tol, equal_nan=False)
    mismatch_n = int((~close_mask).sum())
    roundtrip_valid = bool(mismatch_n == 0)
    if not roundtrip_valid:
        raise RuntimeError(f"The latent-standardization round trip failed for '{split_name}' with {mismatch_n} mismatched values.")
    return pd.Series({"partition": split_name, "rows": len(raw_latent_data), "latent_dimension": len(latent_columns), "mismatch_values": mismatch_n,
        "maximum_absolute_error": float(abs_errs.max()), "mean_absolute_error": float(abs_errs.mean()), "roundtrip_valid": roundtrip_valid},
        name="value")


def diagnose_ladis_latent_space(*, train_latent_data: pd.DataFrame, val_latent_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Diagnose the standardized train and validation latent distributions.

    The diagnostics assess coordinate scale, validation shift, latent
    dependence, and effective dimensionality without forcing the learned
    latent distribution to be Gaussian or independent.
    """

    latent_columns = train_latent_data.columns.tolist()
    train_vals = _validate_ladis_latent_dataframe(latent_data=train_latent_data, expected_columns=latent_columns, 
        data_name="standardized training latent data")
    val_vals = _validate_ladis_latent_dataframe(latent_data=val_latent_data, expected_columns=latent_columns, 
        data_name="standardized validation latent data")
    train_means = np.mean(train_vals, axis=0)
    train_stds = np.std(train_vals, axis=0, ddof=0)
    val_means = np.mean(val_vals, axis=0)
    val_stds = np.std(val_vals, axis=0, ddof=0)
    if np.any(val_stds <= 0.0):
        raise RuntimeError("At least one validation latent coordinate is constant.")
    train_lower_bounds = np.min(train_vals, axis=0)
    train_upper_bounds = np.max(train_vals, axis=0)
    val_outside_training_support = ((val_vals < train_lower_bounds[np.newaxis, :]) | (val_vals > train_upper_bounds[np.newaxis, :]))
    coordinate_records = []
    for column_index, column in enumerate(latent_columns):
        train_column = train_vals[:, column_index]
        val_column = val_vals[:, column_index]
        coordinate_records.append({"latent_coordinate": column, "training_mean": float(train_means[column_index]), 
            "training_std": float(train_stds[column_index]), "validation_mean": float(val_means[column_index]), 
            "validation_std": float(val_stds[column_index]), 
            "validation_to_training_std_ratio": float(val_stds[column_index] / train_stds[column_index]), 
            "training_q001": float(np.quantile(train_column, 0.001)), "training_q999": float(np.quantile(train_column, 0.999)),
            "validation_q001": float(np.quantile(val_column, 0.001)), "validation_q999": float(np.quantile(val_column, 0.999)),
            "validation_outside_training_range_pct": float(100.0*val_outside_training_support[:, column_index].mean())})
    coordinate_diagnostics = (pd.DataFrame(coordinate_records))
    corr_matrix_vals = np.corrcoef(train_vals, rowvar=False)
    if not np.isfinite(corr_matrix_vals).all():
        raise RuntimeError("The standardized training latent correlation matrix contains non-finite values.")
    corr_matrix = pd.DataFrame(corr_matrix_vals, index=latent_columns, columns=latent_columns)
    off_diagonal_mask = ~np.eye(len(latent_columns), dtype=bool)
    abs_off_diagonal_corrs = np.abs(corr_matrix_vals[off_diagonal_mask])
    cov_matrix = np.cov(train_vals, rowvar=False, bias=True)
    cov_eigenvalues = np.linalg.eigvalsh(cov_matrix)
    cov_eigenvalues = np.clip(cov_eigenvalues, a_min=0.0, a_max=None)
    total_eigenvalue = float(cov_eigenvalues.sum())
    if total_eigenvalue <= 0.0:
        raise RuntimeError("The standardized latent covariance matrix has no positive variance.")
    norm_eigenvalues = cov_eigenvalues / total_eigenvalue
    pos_prob_mask = norm_eigenvalues > 0.0
    effective_rank = float(np.exp(-np.sum(norm_eigenvalues[pos_prob_mask]*np.log(norm_eigenvalues[pos_prob_mask]))))
    pos_eigenvalues = cov_eigenvalues[cov_eigenvalues > (cov_eigenvalues.max()*1e-12)]
    if pos_eigenvalues.size:
        cov_condition_num = float(pos_eigenvalues.max() / pos_eigenvalues.min())
    else:
        cov_condition_num = np.inf
    max_abs_training_mean = float(np.abs(train_means).max())
    max_training_std_error = float(np.abs(train_stds - 1.0).max())
    # Training standardization should be numerically exact up to float32 conversion tolerance
    if max_abs_training_mean > 1e-4:
        raise RuntimeError("The standardized training latent coordinates are not centered sufficiently close to zero.")
    if max_training_std_error > 1e-4:
        raise RuntimeError("The standardized training latent coordinates do not have unit variance within the expected numerical tolerance.")
    latent_summary = pd.Series({"training_rows": len(train_latent_data), "validation_rows": len(val_latent_data), 
        "latent_dimension": len(latent_columns), "training_max_abs_coordinate_mean": max_abs_training_mean,
        "training_max_abs_std_error_from_one": max_training_std_error, "validation_max_abs_coordinate_mean": float(np.abs(val_means).max()),
        "validation_minimum_coordinate_std": float(val_stds.min()), "validation_median_coordinate_std": float(np.median(val_stds)),
        "validation_maximum_coordinate_std": float(val_stds.max()), 
        "validation_outside_training_range_pct": float(100.0*val_outside_training_support.mean()), 
        "maximum_absolute_off_diagonal_correlation": float(abs_off_diagonal_corrs.max()), 
        "median_absolute_off_diagonal_correlation": float(np.median(abs_off_diagonal_corrs)), "effective_rank": effective_rank,
        "effective_rank_fraction": float(effective_rank / len(latent_columns)), "covariance_condition_number": cov_condition_num,
        "training_latent_values_are_finite": True, "validation_latent_values_are_finite": True, "latent_space_structurally_valid": True}, name="value")
    return (coordinate_diagnostics, corr_matrix, latent_summary)


# Diffusion evaluation
def evaluate_ladis_clean_z0_denoising(*, diffusion_model: CleanLatentGaussianDiffusion, val_latent_data: pd.DataFrame, timesteps: Sequence[int],
    eval_rows: int, random_state: int, batch_size: int, device: torch.device) -> pd.DataFrame:
    """
    Measure direct-z0 prediction quality at fixed diffusion timesteps.
    """

    latent_columns = val_latent_data.columns.tolist()
    val_vals = _validate_ladis_latent_dataframe(latent_data=val_latent_data, expected_columns=latent_columns, 
        data_name="clean-z0 denoising validation").astype(np.float32)
    eval_rows = min(int(eval_rows), len(val_vals))
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(val_vals), size=eval_rows, replace=False)
    ref_z0 = val_vals[indices]
    records = []
    diffusion_model.eval()
    for timestep in timesteps:
        timestep = int(timestep)
        if not (0 <= timestep < diffusion_model.num_timesteps):
            raise ValueError(f"Invalid diffusion timestep: {timestep}")
        timestep_rng = np.random.default_rng(random_state + timestep + 1)
        noise = timestep_rng.standard_normal(size=ref_z0.shape).astype(np.float32)
        pred_batches = []
        with torch.no_grad():
            for start_index in range(0, eval_rows, batch_size):
                end_index = min(start_index + batch_size, eval_rows)
                z0_batch = torch.from_numpy(ref_z0[start_index:end_index]).to(device=device, dtype=torch.float32)
                noise_batch = torch.from_numpy(noise[start_index:end_index]).to(device=device, dtype=torch.float32)
                timestep_batch = torch.full((len(z0_batch),), timestep, device=device, dtype=torch.long)
                zt_batch = diffusion_model.q_sample(z0=z0_batch, timesteps=timestep_batch, noise=noise_batch)
                pred_batch = diffusion_model.predict_z0(zt=zt_batch, timesteps=timestep_batch)
                pred_batches.append(pred_batch.cpu().numpy())
        pred_z0 = np.concatenate(pred_batches, axis=0).astype(np.float64)
        ref_vals = ref_z0.astype(np.float64)
        error = pred_z0 - ref_vals
        correlations = []
        for column_index in range(ref_vals.shape[1]):
            ref_column = ref_vals[:, column_index]
            pred_column = pred_z0[:, column_index]
            if (np.std(ref_column) > 0.0 and np.std(pred_column) > 0.0):
                correlations.append(float(np.corrcoef(ref_column, pred_column)[0, 1]))
        ref_stds = np.std(ref_vals, axis=0, ddof=0)
        pred_stds = np.std(pred_z0, axis=0, ddof=0)
        std_ratios = pred_stds / ref_stds
        alpha_bar = float(diffusion_model.alphas_cumprod[timestep].detach().cpu().item())
        snr = float(alpha_bar / max(1.0 - alpha_bar, 1e-12))
        records.append({"timestep": timestep, "alpha_bar": alpha_bar, "snr": snr, "z0_mse": float(np.mean(error**2)), 
            "z0_rmse": float(np.sqrt(np.mean(error**2))), "z0_mae": float(np.mean(np.abs(error))), 
            "median_coordinate_correlation": float(np.median(correlations)), "median_predicted_to_reference_std_ratio": float(np.median(std_ratios)),
            "maximum_absolute_predicted_z0": float(np.abs(pred_z0).max())})
    return pd.DataFrame(records)


def validate_preprocessing_round_trip(df_raw: pd.DataFrame, transf_num: pd.DataFrame, transf_cat: pd.DataFrame, *,
    num_artifact: dict, cat_artifact: dict, split_name: str, rtol: float = 1e-7, atol: float = 1e-8) -> pd.DataFrame:
    """
    Validate transformation and inverse transformation on real observations.

    Continuous variables are compared using numerical tolerance. Discrete and
    categorical variables are expected to be reconstructed exactly.
    """

    val_records: list[dict] = []
    restored_num = inverse_transform_numerical_features(transformed_df=transf_num, num_artifact=num_artifact, round_disc=True, enforce_nonneg=True)
    disc_columns = set(num_artifact["discrete_columns"])
    for column in num_artifact["numerical_columns"]:
        ori_vals = pd.to_numeric(df_raw[column], errors="coerce").to_numpy(dtype=float)
        restored_vals = pd.to_numeric(restored_num[column], errors="coerce").to_numpy(dtype=float)
        abs_errors = np.abs(ori_vals - restored_vals)
        if column in disc_columns:
            successful = bool(np.array_equal(ori_vals, restored_vals))
            mismatch_n = int(np.sum(ori_vals != restored_vals))
        else:
            successful = bool(np.allclose(ori_vals, restored_vals, rtol=rtol, atol=atol))
            mismatch_n = int(np.sum(~np.isclose(ori_vals, restored_vals, rtol=rtol, atol=atol)))
        val_records.append({"split": split_name, "column": column, "variable_group": ("discrete" if column in disc_columns else "continuous"),
            "round_trip_successful": successful, "mismatch_n": mismatch_n, 
            "maximum_absolute_error": (float(abs_errors.max()) if len(abs_errors) else np.nan)})
    restored_cat = (inverse_transform_categorical_features(encoded_df=transf_cat, cat_artifact=cat_artifact))
    for column in cat_artifact["categorical_columns"]:
        ori_vals = (df_raw[column].astype("object").to_numpy())
        restored_vals = (restored_cat[column].astype("object").to_numpy())
        equality_mask = np.array([original == restored for original, restored in zip(ori_vals, restored_vals)])
        val_records.append({"split": split_name, "column": column, "variable_group": "categorical", "round_trip_successful": bool(equality_mask.all()),
            "mismatch_n": int((~equality_mask).sum()), "maximum_absolute_error": np.nan})
    return pd.DataFrame(val_records)


def validate_ladis_definitive_preprocessing(*, quantile_num_data: Mapping[str, pd.DataFrame], prequantile_num_data: Mapping[str, pd.DataFrame],
    cat_model_data: Mapping[str, pd.DataFrame], cat_ref_data: Mapping[str, pd.DataFrame], quantile_transf: QuantileTransformer,
    cat_artifact: Mapping[str, Any], num_columns: Sequence[str], cat_columns: Sequence[str], train_split_name: str = "train",
    max_train_prequantile_inv_err: float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate the definitive numerical and categorical LaDIS interfaces.

    Gaussian quantile transformation is not treated as an exactly reversible
    preprocessing operation. It estimates each marginal distribution using a
    finite number of quantile landmarks. In addition, held-out values outside
    the fitted training range are mapped to the boundaries of the transformed
    distribution. Consequently, this validation checks numerical stability and 
    approximate training inversion rather than demanding exact original-scale 
    recovery.

    The categorical representation remains exactly reversible and is checked
    strictly.

    Parameters
    ----------
    quantile_num_data
        Numerical model-space tables after Gaussian quantile transformation,
        indexed by split name.
    prequantile_num_data
        Numerical tables immediately before Gaussian quantile transformation,
        indexed by split name.
    cat_model_data
        Restricted categorical integer-code tables, indexed by split name.
    cat_ref_data
        Expected decoded categorical values, indexed by split name.
    quantile_transf
        Gaussian quantile transformer fitted only on the training partition.
    cat_artifact
        Restricted train-observed categorical artifact used to encode and
        decode LaDIS categorical states.
    num_columns
        Definitive numerical-variable order.
    cat_columns
        Definitive categorical-variable order.
    train_split_name
        Name assigned to the training partition.
    max_train_prequantile_inv_err
        Maximum accepted absolute inverse-quantile error in pre-quantile
        training space. This is a numerical-stability safeguard rather than
        an exact-reconstruction tolerance.

    Returns
    -------
    split_summary
        One validation record per partition.
    num_diagnostics
        One numerical diagnostic record per variable and partition.
    cat_diagnostics
        One categorical diagnostic record per variable and partition.
    """

    num_columns = list(num_columns)
    cat_columns = list(cat_columns)
    split_names = set(quantile_num_data)
    input_split_sets = {"prequantile_num_data": set(prequantile_num_data),
        "categorical_model_data": set(cat_model_data), "categorical_reference_data": set(cat_ref_data)}
    for input_name, input_splits in input_split_sets.items():
        if input_splits != split_names:
            raise ValueError(f"`{input_name}` does not contain the same partitions as `quantile_numerical_data`.")
    if train_split_name not in split_names:
        raise ValueError(f"Training split '{train_split_name}' was not provided.")
    if max_train_prequantile_inv_err <= 0:
        raise ValueError("`max_train_prequantile_inverse_error` must be positive.")
    # The first and final fitted quantiles define the empirical numerical  support learned exclusively from training data
    fitted_quantiles = np.asarray(quantile_transf.quantiles_, dtype=np.float64)
    if fitted_quantiles.shape[1] != len(num_columns):
        raise RuntimeError("The fitted quantile-transformer dimension differs from the definitive numerical schema.")
    prequantile_lower_bounds = fitted_quantiles[0, :]
    prequantile_upper_bounds = fitted_quantiles[-1, :]
    # The transformed training table defines the numerical output support that should also contain transformed validation and test values
    train_quantile_vals = (quantile_num_data[train_split_name].loc[:, num_columns].to_numpy(dtype=np.float64))
    quantile_lower_bounds = np.min(train_quantile_vals, axis=0)
    quantile_upper_bounds = np.max(train_quantile_vals, axis=0)
    split_records: list[dict[str, Any]] = []
    num_records: list[dict[str, Any]] = []
    cat_records: list[dict[str, Any]] = []
    for split_name in quantile_num_data:
        quantile_data = (quantile_num_data[split_name].reset_index(drop=True))
        prequantile_data = (prequantile_num_data[split_name].reset_index(drop=True))
        cat_codes = (cat_model_data[split_name].reset_index(drop=True))
        cat_ref = (cat_ref_data[split_name].reset_index(drop=True))
        num_schema_matches = bool(quantile_data.columns.tolist() == num_columns and prequantile_data.columns.tolist() == num_columns)
        cat_schema_matches = bool(cat_codes.columns.tolist() == cat_columns and cat_ref.columns.tolist() == cat_columns)
        if not num_schema_matches:
            raise RuntimeError(f"The numerical schema differs for split '{split_name}'.")
        if not cat_schema_matches:
            raise RuntimeError(f"The categorical schema differs for split '{split_name}'.")
        row_counts_match = bool(len(quantile_data) == len(prequantile_data) == len(cat_codes) == len(cat_ref))
        if not row_counts_match:
            raise RuntimeError(f"The definitive representations contain different row counts for split '{split_name}'.")
        quantile_vals = quantile_data.to_numpy(dtype=np.float64)
        prequantile_vals = prequantile_data.to_numpy(dtype=np.float64)
        quantile_vals_are_finite = bool(np.isfinite(quantile_vals).all())
        prequantile_vals_are_finite = bool(np.isfinite(prequantile_vals).all())
        if not quantile_vals_are_finite:
            raise RuntimeError(f"Split '{split_name}' contains non-finite Gaussian-quantile values.")
        if not prequantile_vals_are_finite:
            raise RuntimeError(f"Split '{split_name}' contains non-finite pre-quantile values.")
        # The quantile inverse is assessed in pre-quantile model space
        rec_prequantile_data = inverse_ladis_quantile_space(quantile_model_data=quantile_data, quantile_transf=quantile_transf,
                expected_columns=num_columns)
        rec_prequantile_vals = (rec_prequantile_data.to_numpy(dtype=np.float64))
        inv_vals_are_finite = bool(np.isfinite(rec_prequantile_vals).all())
        inv_abs_errs = np.abs(rec_prequantile_vals - prequantile_vals)
        max_inv_err = float(inv_abs_errs.max())
        mean_inv_err = float(inv_abs_errs.mean())
        median_inv_err = float(np.median(inv_abs_errs))
        # Held-out values outside the fitted training range are expected to be mapped to quantile boundaries
        below_fitted_support = (prequantile_vals < prequantile_lower_bounds[np.newaxis, :])
        above_fitted_support = (prequantile_vals > prequantile_upper_bounds[np.newaxis, :])
        outside_fitted_support = (below_fitted_support | above_fitted_support)
        # After transformation, all values should lie inside the Gaussian output support observed in the transformed training partition
        below_quantile_support = (quantile_vals < (quantile_lower_bounds[np.newaxis, :] - 1e-6))
        above_quantile_support = (quantile_vals > (quantile_upper_bounds[np.newaxis, :] + 1e-6))
        outside_quantile_support = (below_quantile_support | above_quantile_support)
        # Validate restricted categorical integer codes
        total_invalid_cat_codes = 0
        for column in cat_columns:
            column_info = (cat_artifact["column_information"][column])
            cardinality = int(column_info["number_of_categories"])
            codes = pd.to_numeric(cat_codes[column], errors="raise").to_numpy(dtype=np.float64)
            rounded_codes = np.rint(codes)
            nonint_mask = ~np.isclose(codes, rounded_codes, atol=1e-8, rtol=0.0)
            int_codes = rounded_codes.astype(np.int64)
            invalid_code_mask = (nonint_mask | (int_codes < 0) | (int_codes >= cardinality))
            invalid_code_n = int(invalid_code_mask.sum())
            total_invalid_cat_codes += invalid_code_n
            cat_records.append({"split": split_name, "column": column, "cardinality": cardinality, "observed_code_min": int(int_codes.min()),
                "observed_code_max": int(int_codes.max()), "invalid_code_n": invalid_code_n})
        # Decode using the same restricted artifact used during encoding
        rec_cat_data = inverse_transform_categorical_features(cat_codes, cat_artifact).loc[:, cat_columns].reset_index(drop=True)
        rec_cat_strings = (rec_cat_data.astype("string"))
        expected_cat_strings = (cat_ref.astype("string"))
        cat_mismatch_mask = (rec_cat_strings != expected_cat_strings)
        cat_mismatch_n = int(cat_mismatch_mask.to_numpy().sum())
        # Only the training partition is expected to be approximately reversible through the fitted quantile estimator
        train_inv_err_valid = bool(split_name != train_split_name or max_inv_err <= max_train_prequantile_inv_err)
        split_is_valid = bool(num_schema_matches and cat_schema_matches and row_counts_match and quantile_vals_are_finite
            and prequantile_vals_are_finite and inv_vals_are_finite and not outside_quantile_support.any() and total_invalid_cat_codes == 0
            and cat_mismatch_n == 0 and train_inv_err_valid)
        total_num_vals = int(prequantile_vals.size)
        outside_fitted_support_n = int(outside_fitted_support.sum())
        split_records.append({"split": split_name, "rows": len(quantile_data), "numerical_schema_matches": num_schema_matches,
            "categorical_schema_matches": cat_schema_matches, "quantile_values_are_finite": quantile_vals_are_finite,
            "inverse_quantile_values_are_finite": inv_vals_are_finite, "maximum_prequantile_inverse_error": max_inv_err,
            "mean_prequantile_inverse_error": mean_inv_err, "median_prequantile_inverse_error": median_inv_err,
            "values_outside_fitted_training_support_n": outside_fitted_support_n,
            "values_outside_fitted_training_support_pct": float(100.0*outside_fitted_support_n / total_num_vals),
            "quantile_values_outside_training_output_support_n": int(outside_quantile_support.sum()),
            "invalid_categorical_code_n": total_invalid_cat_codes, "categorical_mismatch_n": cat_mismatch_n, 
            "train_inverse_error_valid": train_inv_err_valid, "interface_valid": split_is_valid})
        # Per-variable numerical diagnostics
        for column_index, column in enumerate(num_columns):
            num_records.append({"split": split_name, "column": column, "fitted_prequantile_min": float(prequantile_lower_bounds[column_index]),
                "fitted_prequantile_max": float(prequantile_upper_bounds[column_index]),  
                "below_fitted_support_n": int(below_fitted_support[:, column_index].sum()), 
                "above_fitted_support_n": int(above_fitted_support[:, column_index].sum()),
                "maximum_prequantile_inverse_error": float(inv_abs_errs[:, column_index].max()),
                "mean_prequantile_inverse_error": float(inv_abs_errs[:, column_index].mean())})
    split_summary = pd.DataFrame(split_records)
    num_diagnostics = pd.DataFrame(num_records)
    cat_diagnostics = pd.DataFrame(cat_records)
    return (split_summary, num_diagnostics, cat_diagnostics)