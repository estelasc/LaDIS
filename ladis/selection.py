"""
Model-selection utilities for LaDIS.

This module implements the two-stage selection procedure used by LaDIS:
1. latent-dimensionality selection, where complete autoencoder--diffusion
   pipelines are compared using synthetic-data quality;
2. clean-z0 latent-diffusion selection, where the reference denoiser and the
   controlled learning-rate/width variants are compared using the deterministic
   hierarchy reported in the paper.

Only training and validation data are used by these functions. Test data must
remain excluded from all model-selection decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import warnings

import numpy as np
import pandas as pd
import torch

from .evaluation import (build_sdmetrics_metadata, evaluate_sample, evaluate_ladis_autoencoder_reconstruction, evaluate_ladis_clean_z0_denoising,
    reconstruct_ladis_autoencoder_partition)
from .latent import (encode_ladis_partition, fit_ladis_latent_standardization, freeze_ladis_autoencoder, inverse_ladis_latent_standardization,
    transform_ladis_latent_space)
from .preprocessing import inverse_ladis_mode_dequantization, inverse_ladis_model_representation, inverse_ladis_quantile_space
from .sampling import decode_ladis_latent_partition, sample_ladis_standardized_latents
from .training import fit_ladis_autoencoder, fit_ladis_clean_z0_diffusion


# Shared selection helpers
def _summarize_ladis_generated_latent(*, gen_latent_data: pd.DataFrame, training_latent_data: pd.DataFrame) -> pd.Series:
    """
    Compare one generated standardized latent sample with the training latent space.

    The returned quantities are the same latent diagnostics used during the
    clean-z0 denoiser selection in the LaDIS experiments.
    """

    if gen_latent_data.columns.tolist() != training_latent_data.columns.tolist():
        raise RuntimeError("Generated and training latent data do not preserve the same coordinate order.")
    if gen_latent_data.empty or training_latent_data.empty:
        raise ValueError("Generated and training latent data must be non-empty.")
    gen_vals = gen_latent_data.to_numpy(dtype=np.float64)
    train_vals = training_latent_data.to_numpy(dtype=np.float64)
    if not np.isfinite(gen_vals).all():
        raise RuntimeError("Generated standardized latent data contain non-finite values.")
    if not np.isfinite(train_vals).all():
        raise RuntimeError("Training standardized latent data contain non-finite values.")
    gen_coordinate_means = np.mean(gen_vals, axis=0)
    gen_coordinate_stds = np.std(gen_vals, axis=0, ddof=0)
    train_coordinate_stds = np.std(train_vals, axis=0, ddof=0)
    if np.any(train_coordinate_stds <= 0.0):
        raise RuntimeError("At least one training latent coordinate has zero variance.")
    gen_to_training_std_ratio = gen_coordinate_stds / train_coordinate_stds
    gen_latent_corr = np.corrcoef(gen_vals, rowvar=False)
    train_latent_corr = np.corrcoef(train_vals, rowvar=False)
    offdiag_mask = ~np.eye(train_vals.shape[1], dtype=bool)
    return pd.Series({"maximum_absolute_generated_coordinate_mean": float(np.abs(gen_coordinate_means).max()),
        "median_generated_to_training_std_ratio": float(np.median(gen_to_training_std_ratio)), 
        "minimum_generated_to_training_std_ratio": float(gen_to_training_std_ratio.min()),
        "maximum_generated_to_training_std_ratio": float(gen_to_training_std_ratio.max()),
        "mean_absolute_latent_correlation_error": float(np.mean(np.abs(gen_latent_corr[offdiag_mask] - train_latent_corr[offdiag_mask]))),
        "maximum_absolute_generated_latent": float(np.abs(gen_vals).max()), "generated_latent_values_are_finite": True}, name="value")


def _evaluate_ladis_diffusion_candidate(*, cand_name: str, diffusion_model: torch.nn.Module, autoencoder: torch.nn.Module,
    training_latent_data: pd.DataFrame, real_val_data: pd.DataFrame, fitted_training_data: pd.DataFrame, metadata: Mapping[str, Any],
    latent_standardization_artifact: Mapping[str, Any], num_quantile_transformer: Any, mode_dequantization_artifact: Mapping[str, Any],
    num_artifact: Mapping[str, Any], cat_artifact: Mapping[str, Any], feature_order: Sequence[str], num_columns: Sequence[str], 
    cat_columns: Sequence[str], cat_cards: Mapping[str, int], sample_rows: int, batch_size: int, random_state: int, 
    device: torch.device) -> dict[str, Any]:
    """
    Generate and evaluate one fitted clean-z0 latent-diffusion candidate.

    Every candidate is sampled with the same number of observations and the
    same random seeds so that the comparison differs only because of the fitted
    candidate itself.
    """

    feature_order = list(feature_order)
    num_columns = list(num_columns)
    cat_columns = list(cat_columns)
    latent_columns = training_latent_data.columns.tolist()
    if sample_rows <= 0:
        raise ValueError("`sample_rows` must be positive.")
    if len(real_val_data) != sample_rows:
        raise ValueError("The real validation reference must contain exactly `sample_rows` observations.")
    # Generate standardized latent observations
    synth_z0, synth_z0_summary = sample_ladis_standardized_latents(diffusion_model=diffusion_model, num_rows=sample_rows, latent_columns=latent_columns,
        batch_size=batch_size, random_state=random_state + 30_000, device=device)
    # Compare generated and training latent distributions
    latent_summary = _summarize_ladis_generated_latent(gen_latent_data=synth_z0, training_latent_data=training_latent_data)
    # Return generated standardized latents to the raw autoencoder latent space
    synth_z_raw = inverse_ladis_latent_standardization(standardized_latent_data=synth_z0, standardization_artifact=latent_standardization_artifact)
    # Decode through the frozen autoencoder. Categorical states are sampled from the decoder softmax distributions
    synth_num_quantile, synth_cat_codes, decode_summary = decode_ladis_latent_partition(model=autoencoder, raw_latent_data=synth_z_raw,
        num_columns=num_columns, cat_columns=cat_columns, cat_cards=cat_cards, batch_size=batch_size, device=device, cat_sampling="sample",
        random_state=random_state + 40_000)
    # Invert the numerical representation used by the autoencoder
    synth_num_prequantile = inverse_ladis_quantile_space(quantile_model_data=synth_num_quantile, quantile_transf=num_quantile_transformer,
        expected_columns=num_columns)
    synth_num_preproc = inverse_ladis_mode_dequantization(prequantile_num_data=synth_num_prequantile,
        dequantization_artifact=mode_dequantization_artifact)
    synth_eval = inverse_ladis_model_representation(num_model_data=synth_num_preproc, cat_model_data=synth_cat_codes, num_columns=num_columns,
        cat_columns=cat_columns, feature_order=feature_order, num_artifact=num_artifact, cat_artifact=cat_artifact, round_disc=True, 
        enforce_nonneg=True)
    # Candidate validity checks used before statistical evaluation.
    if synth_eval.columns.tolist() != feature_order:
        raise RuntimeError(f"Candidate '{cand_name}' does not preserve the common evaluation schema.")
    if len(synth_eval) != sample_rows:
        raise RuntimeError(f"Candidate '{cand_name}' does not preserve the requested sample size.")
    if synth_eval.isna().any().any():
        raise RuntimeError(f"Candidate '{cand_name}' contains missing values.")
    if np.isinf(synth_eval[num_columns].to_numpy(dtype=np.float64)).any():
        raise RuntimeError(f"Candidate '{cand_name}' contains infinite numerical values.")
    quality_summary, property_scores, quality_report = evaluate_sample(real_val_data=real_val_data, synth_data=synth_eval,
        fitted_training_data=fitted_training_data, metadata=dict(metadata), cat_columns=cat_columns, candidate_name=cand_name)
    return {"synth_z0": synth_z0, "synth_z0_summary": synth_z0_summary, "latent_summary": latent_summary, "synth_z_raw": synth_z_raw,
        "decode_summary": decode_summary, "synth_eval": synth_eval, "quality_summary": quality_summary, "property_scores": property_scores,
        "quality_report": quality_report}


# latent-dimensionality selection
def rank_ladis_latent_dimensions(latent_dim_study: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Rank latent dimensions using the LaDIS generative mean-rank criterion.

    Overall Quality Score, Column Shapes, and Column Pair Trends are ranked
    independently in descending order. The candidate with the lowest mean rank
    is selected; overall synthetic quality is the deterministic tie-breaker.
    """

    required_columns = {"synthetic_quality", "synthetic_shapes", "synthetic_pair_trends"}
    missing_columns = sorted(required_columns.difference(latent_dim_study.columns))
    if missing_columns:
        raise ValueError(f"Latent-dimension results are missing required columns: {missing_columns}")
    if latent_dim_study.empty:
        raise ValueError("The latent-dimension study is empty.")
    ranking = latent_dim_study.copy()
    selection_metrics = {"synthetic_quality": "max", "synthetic_shapes": "max", "synthetic_pair_trends": "max"}
    rank_columns: list[str] = []
    for metric, direction in selection_metrics.items():
        rank_column = f"{metric}_rank"
        ranking[rank_column] = ranking[metric].rank(ascending=(direction == "min"), method="average")
        rank_columns.append(rank_column)
    ranking["generative_mean_rank"] = ranking[rank_columns].mean(axis=1)
    ranking = ranking.sort_values(["generative_mean_rank", "synthetic_quality"], ascending=[True, False])
    selected_dimension = int(ranking.index[0])
    return ranking, selected_dimension


def run_ladis_latent_dimension_selection(*, train_num_data: pd.DataFrame, train_cat_data: pd.DataFrame, val_num_data: pd.DataFrame,
    val_cat_data: pd.DataFrame, train_eval_data: pd.DataFrame, val_eval_data: pd.DataFrame, feature_order: Sequence[str], num_columns: Sequence[str],
    cat_columns: Sequence[str], cat_cards: Mapping[str, int], num_artifact: Mapping[str, Any], cat_artifact: Mapping[str, Any],
    num_quantile_transformer: Any, mode_dequantization_artifact: Mapping[str, Any], latent_dim_cands: Sequence[int], ae_base_conf: Mapping[str, Any],
    diffusion_base_conf: Mapping[str, Any], random_state: int, device: torch.device, evaluation_rows: int = 10_000, batch_size: int = 4_096,
    verbose: bool = True) -> dict[str, Any]:
    """
    Run the complete LaDIS latent-dimensionality selection experiment.

    For every candidate dimension a complete autoencoder--diffusion pipeline is
    trained while all other architecture and optimization settings remain
    fixed. Each candidate generates the same number of validation observations.
    The final dimension is selected from downstream synthetic-data quality, not
    from autoencoder reconstruction error.
    """

    feature_order = list(feature_order)
    num_columns = list(num_columns)
    cat_columns = list(cat_columns)
    cand_dims = [int(value) for value in latent_dim_cands]
    if not cand_dims:
        raise ValueError("At least one latent-dimension candidate is required.")
    if len(cand_dims) != len(set(cand_dims)):
        raise ValueError("`latent_dimension_candidates` contains duplicated values.")
    if any(value <= 0 for value in cand_dims):
        raise ValueError("Every latent-dimension candidate must be positive.")
    if evaluation_rows <= 0:
        raise ValueError("`evaluation_rows` must be positive.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")
    effective_eval_rows = min(int(evaluation_rows), len(val_eval_data))
    if effective_eval_rows <= 0:
        raise ValueError("The validation evaluation data are empty.")
    # Use the same held-out validation observations for every candidate
    eval_indices = val_eval_data.sample(n=effective_eval_rows, random_state=random_state, replace=False).index
    real_val_ref = val_eval_data.loc[eval_indices, feature_order].reset_index(drop=True)
    metadata = build_sdmetrics_metadata(feature_order=feature_order, cat_columns=cat_columns)
    result_records: list[dict[str, Any]] = []
    ae_histories: dict[int, pd.DataFrame] = {}
    diffusion_histories: dict[int, pd.DataFrame] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=(r"ks_2samp: Exact calculation unsuccessful\. "
                r"Switching to method=asymp\."), category=RuntimeWarning)
        for latent_dim in cand_dims:
            candidate_name = f"latent_dim_{latent_dim}"
            if verbose:
                print("\n" + "=" * 80)
                print(f"LATENT DIMENSION: {latent_dim}")
                print("=" * 80)
            # Train the mixed-type autoencoder with only the latent dimension changed
            ae_conf = dict(ae_base_conf)
            ae_conf["name"] = f"ae_latent{latent_dim}"
            ae_conf["latent_dimension"] = int(latent_dim)
            candidate_ae, ae_history, _ = fit_ladis_autoencoder(train_num_data=train_num_data, train_cat_data=train_cat_data, val_num_data=val_num_data,
                val_cat_data=val_cat_data, num_columns=num_columns, cat_columns=cat_columns, cat_cards=cat_cards, conf=ae_conf,
                random_state=random_state, device=device, verbose=False)
            ae_histories[latent_dim] = ae_history.copy()
            # Direct autoencoder reconstruction diagnostics
            val_ae_summary, _, _ = evaluate_ladis_autoencoder_reconstruction(model=candidate_ae, num_data=val_num_data, cat_data=val_cat_data,
                num_columns=num_columns, cat_columns=cat_columns, cat_cards=cat_cards, batch_size=batch_size, device=device, 
                split_name=f"validation_latent{latent_dim}")
            # Empirical autoencoder reconstruction ceiling in common evaluation space
            rec_num_quantile, rec_cat_codes = reconstruct_ladis_autoencoder_partition(model=candidate_ae, num_data=val_num_data, cat_data=val_cat_data,
                num_columns=num_columns, cat_columns=cat_columns, cat_cards=cat_cards, batch_size=batch_size, device=device,
                data_name=f"validation_reconstruction_latent{latent_dim}")
            rec_num_prequantile = inverse_ladis_quantile_space(quantile_model_data=rec_num_quantile, quantile_transf=num_quantile_transformer,
                expected_columns=num_columns)
            rec_num_preproc = inverse_ladis_mode_dequantization(prequantile_num_data=rec_num_prequantile,
                dequantization_artifact=mode_dequantization_artifact)
            rec_eval = inverse_ladis_model_representation(num_model_data=rec_num_preproc, cat_model_data=rec_cat_codes, num_columns=num_columns,
                cat_columns=cat_columns, feature_order=feature_order, num_artifact=num_artifact, cat_artifact=cat_artifact, round_disc=True,
                enforce_nonneg=True)
            ceiling_rec = rec_eval.loc[eval_indices, feature_order].reset_index(drop=True)
            ceiling_summary, ceiling_properties, _ = evaluate_sample(real_val_data=real_val_ref, synth_data=ceiling_rec,
                fitted_training_data=train_eval_data, metadata=metadata, cat_columns=cat_columns, candidate_name=f"ae_ceiling_latent{latent_dim}")
            ceiling_lookup = ceiling_properties.set_index("Property")["Score"].to_dict()
            # Construct the train-fitted standardized latent representation
            candidate_ae, _ = freeze_ladis_autoencoder(model=candidate_ae)
            train_z_raw, _ = encode_ladis_partition(model=candidate_ae, num_data=train_num_data, cat_data=train_cat_data, num_columns=num_columns,
                cat_columns=cat_columns, cat_cards=cat_cards, batch_size=batch_size, device=device, data_name=f"train_latent{latent_dim}")
            val_z_raw, _ = encode_ladis_partition(model=candidate_ae, num_data=val_num_data, cat_data=val_cat_data, num_columns=num_columns,
                cat_columns=cat_columns, cat_cards=cat_cards, batch_size=batch_size, device=device, data_name=f"val_latent{latent_dim}")
            train_z0, latent_standardization_artifact, _, _ = fit_ladis_latent_standardization(train_latent_data=train_z_raw, min_coordinate_std=1e-6)
            val_z0, _ = transform_ladis_latent_space(latent_data=val_z_raw, standardization_artifact=latent_standardization_artifact, split_name="val")
            # Train the same clean-z0 denoiser architecture for every dimension
            diffusion_conf = dict(diffusion_base_conf)
            diffusion_conf["name"] = f"clean_z0_latent{latent_dim}"
            diffusion_conf["latent_dimension"] = int(latent_dim)
            candidate_diffusion, diffusion_history, diffusion_training_summary = fit_ladis_clean_z0_diffusion(train_latent_data=train_z0,
                val_latent_data=val_z0, conf=diffusion_conf, random_state=random_state, device=device, verbose=False)
            diffusion_histories[latent_dim] = diffusion_history.copy()
            # Generate and decode the same number of observations for every dimension
            latent_columns = [f"z_{dimension:02d}" for dimension in range(latent_dim)]
            synth_z0, _ = sample_ladis_standardized_latents(diffusion_model=candidate_diffusion, num_rows=effective_eval_rows,
                latent_columns=latent_columns, batch_size=batch_size, random_state=random_state + 30_000, device=device)
            synth_z_raw = inverse_ladis_latent_standardization(standardized_latent_data=synth_z0,
                standardization_artifact=latent_standardization_artifact)
            synth_num_quantile, synth_cat_codes, _ = decode_ladis_latent_partition(model=candidate_ae, raw_latent_data=synth_z_raw, 
                num_columns=num_columns, cat_columns=cat_columns, cat_cards=cat_cards, batch_size=batch_size, device=device, cat_sampling="sample",
                random_state=random_state + 40_000)
            synth_num_prequantile = inverse_ladis_quantile_space(quantile_model_data=synth_num_quantile, quantile_transf=num_quantile_transformer,
                expected_columns=num_columns)
            synth_num_preproc = inverse_ladis_mode_dequantization(prequantile_num_data=synth_num_prequantile,
                dequantization_artifact=mode_dequantization_artifact)
            synth_eval = inverse_ladis_model_representation(num_model_data=synth_num_preproc, cat_model_data=synth_cat_codes, num_columns=num_columns,
                cat_columns=cat_columns, feature_order=feature_order, num_artifact=num_artifact, cat_artifact=cat_artifact, round_disc=True,
                enforce_nonneg=True)
            synth_quality, synth_properties, _ = evaluate_sample(real_val_data=real_val_ref, synth_data=synth_eval, fitted_training_data=train_eval_data,
                metadata=metadata, cat_columns=cat_columns, candidate_name=candidate_name)
            synth_lookup = synth_properties.set_index("Property")["Score"].to_dict()
            result_records.append({"latent_dimension": int(latent_dim), "ae_validation_rmse": float(val_ae_summary["numerical_global_rmse"]),
                    "ae_validation_median_correlation": float(val_ae_summary["numerical_median_correlation"]),
                    "ae_categorical_macro_accuracy": float(val_ae_summary["categorical_macro_accuracy"]),
                    "ae_ceiling_quality": float(ceiling_summary["quality_score"]), "ae_ceiling_shapes": float(ceiling_lookup["Column Shapes"]),
                    "ae_ceiling_pair_trends": float(ceiling_lookup["Column Pair Trends"]),
                    "diffusion_best_validation_z0_mse": float(diffusion_training_summary["best_validation_z0_mse"]),
                    "synthetic_quality": float(synth_quality["quality_score"]), "synthetic_shapes": float(synth_lookup["Column Shapes"]),
                    "synthetic_pair_trends": float(synth_lookup["Column Pair Trends"])})
            if verbose:
                print(f"\nCompleted latent dimension {latent_dim}.")
    study = pd.DataFrame(result_records).sort_values("latent_dimension").set_index("latent_dimension")
    ranking, selected_dimension = rank_ladis_latent_dimensions(study)
    return {"selected_latent_dimension": selected_dimension, "study": study, "ranking": ranking, "autoencoder_histories": ae_histories,
        "diffusion_histories": diffusion_histories, "evaluation_rows": effective_eval_rows, "evaluation_indices": eval_indices.copy(),
        "real_validation_reference": real_val_ref, "sdmetrics_metadata": metadata}


# Stage 2: clean-z0 denoiser selection
def build_ladis_clean_z0_candidate_configurations(ref_conf: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Build the three controlled clean-z0 optimization candidates used by LaDIS.

    Starting from the reference configuration, the candidates use:

    - a lower learning rate of 2.5e-4;
    - a wider denoiser (512, 1024, 1024, 512);
    - the wider denoiser together with the lower learning rate.
    """

    reference = dict(ref_conf)
    if "name" not in reference:
        raise ValueError("The reference diffusion configuration must contain a `name`.")
    lower_lr = dict(reference)
    lower_lr["name"] = "lower_lr_clean_z0_ladis"
    lower_lr["learning_rate"] = 2.5e-4
    wide = dict(reference)
    wide["name"] = "wide_clean_z0_ladis"
    wide["hidden_dimensions"] = (512, 1024, 1024, 512)
    wide_lower_lr = dict(reference)
    wide_lower_lr["name"] = "wide_lower_lr_clean_z0_ladis"
    wide_lower_lr["hidden_dimensions"] = (512, 1024, 1024, 512)
    wide_lower_lr["learning_rate"] = 2.5e-4
    return [lower_lr, wide, wide_lower_lr]


def run_ladis_clean_z0_selection(*, train_latent_data: pd.DataFrame, val_latent_data: pd.DataFrame, autoencoder: torch.nn.Module,
    ref_conf: Mapping[str, Any], real_val_data: pd.DataFrame, fitted_training_data: pd.DataFrame, metadata: Mapping[str, Any],
    latent_standardization_artifact: Mapping[str, Any], num_quantile_transformer: Any, mode_dequantization_artifact: Mapping[str, Any],
    num_artifact: Mapping[str, Any], cat_artifact: Mapping[str, Any], feature_order: Sequence[str], num_columns: Sequence[str], 
    cat_columns: Sequence[str], cat_cards: Mapping[str, int], random_state: int, device: torch.device, batch_size: int = 4_096,
    diagnostic_timesteps: Sequence[int] = (0, 50, 125, 250, 375, 499), diagnostic_rows: int = 10_000, verbose: bool = True) -> dict[str, Any]:
    """
    Train, evaluate, rank, and select the definitive clean-z0 LaDIS denoiser.

    The reference configuration and the three controlled optimization variants
    are trained on the same latent training data using the same random seed.
    Every fitted candidate is evaluated with the same generated sample size and
    validation reference. Selection follows the deterministic hierarchy used in
    the paper: higher Quality Score, lower latent correlation error, smaller
    deviation of the median latent standard-deviation ratio from one, and fewer
    model parameters.

    The selected fitted checkpoint is returned directly; no additional refit is
    performed after selection.
    """

    ref_conf = dict(ref_conf)
    if "name" not in ref_conf:
        raise ValueError("The reference diffusion configuration must contain a `name`.")
    if int(ref_conf["latent_dimension"]) != train_latent_data.shape[1]:
        raise RuntimeError("The reference diffusion configuration does not match the selected latent dimension.")
    if train_latent_data.columns.tolist() != val_latent_data.columns.tolist():
        raise RuntimeError("Training and validation latent data do not preserve the same coordinate order.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")
    sample_rows = len(real_val_data)
    if sample_rows <= 0:
        raise ValueError("The real validation reference is empty.")
    optimization_confs = build_ladis_clean_z0_candidate_configurations(ref_conf)
    all_confs = [ref_conf, *optimization_confs]
    models: dict[str, torch.nn.Module] = {}
    histories: dict[str, pd.DataFrame] = {}
    training_summaries: dict[str, pd.Series] = {}
    denoising_diagnostics: dict[str, pd.DataFrame] = {}
    results: dict[str, dict[str, Any]] = {}
    for conf in all_confs:
        candidate_name = str(conf["name"])
        if verbose:
            print("\n" + "=" * 80)
            print(f"Training candidate: {candidate_name}")
            print("=" * 80)
        model, history, training_summary = fit_ladis_clean_z0_diffusion(train_latent_data=train_latent_data, val_latent_data=val_latent_data,
            conf=conf, random_state=random_state, device=device, verbose=verbose)
        models[candidate_name] = model
        histories[candidate_name] = history
        training_summaries[candidate_name] = training_summary
        denoising_diagnostics[candidate_name] = evaluate_ladis_clean_z0_denoising(diffusion_model=model, val_latent_data=val_latent_data,
            timesteps=diagnostic_timesteps, eval_rows=min(int(diagnostic_rows), len(val_latent_data)), random_state=random_state + 20_000,
            batch_size=batch_size, device=device)
        results[candidate_name] = _evaluate_ladis_diffusion_candidate(cand_name=candidate_name, diffusion_model=model, autoencoder=autoencoder,
            training_latent_data=train_latent_data, real_val_data=real_val_data, fitted_training_data=fitted_training_data, metadata=metadata,
            latent_standardization_artifact=latent_standardization_artifact, num_quantile_transformer=num_quantile_transformer,
            mode_dequantization_artifact=mode_dequantization_artifact, num_artifact=num_artifact, cat_artifact=cat_artifact, 
            feature_order=feature_order, num_columns=num_columns, cat_columns=cat_columns, cat_cards=cat_cards, sample_rows=sample_rows,
            batch_size=batch_size, random_state=random_state, device=device)
    # Construct candidate-level comparison
    comparison_records: list[dict[str, Any]] = []
    reference_name = str(ref_conf["name"])
    for conf in all_confs:
        candidate_name = str(conf["name"])
        training_summary = training_summaries[candidate_name]
        candidate_result = results[candidate_name]
        property_lookup = candidate_result["property_scores"].set_index("Property")["Score"].to_dict()
        if candidate_name == reference_name:
            changed_factor = "reference"
        elif candidate_name == "lower_lr_clean_z0_ladis":
            changed_factor = "learning_rate"
        elif candidate_name == "wide_clean_z0_ladis":
            changed_factor = "denoiser_width"
        else:
            changed_factor = "denoiser_width_and_learning_rate"
        comparison_records.append({"candidate": candidate_name, "changed_factor": changed_factor, "hidden_dimensions": str(conf["hidden_dimensions"]),
            "learning_rate": float(conf["learning_rate"]), "model_parameters": int(training_summary["model_parameters"]),
            "best_step": int(training_summary["best_step"]), "best_validation_z0_mse": float(training_summary["best_validation_z0_mse"]),
            "quality_score": float(candidate_result["quality_summary"]["quality_score"]), "column_shapes_score": float(property_lookup["Column Shapes"]),
            "column_pair_trends_score": float(property_lookup["Column Pair Trends"]),
            "latent_correlation_error": float(candidate_result["latent_summary"]["mean_absolute_latent_correlation_error"]),
            "median_generated_latent_std_ratio": float(candidate_result["latent_summary"]["median_generated_to_training_std_ratio"])})
    comparison = pd.DataFrame(comparison_records).sort_values("quality_score", ascending=False).reset_index(drop=True)
    # Deterministic final hierarchy used by LaDIS
    selection = comparison.copy()
    selection["latent_std_ratio_error"] = selection["median_generated_latent_std_ratio"].sub(1.0).abs()
    selection = selection.sort_values(by=["quality_score", "latent_correlation_error", "latent_std_ratio_error", "model_parameters"],
        ascending=[False, True, True, True]).reset_index(drop=True)
    selected_name = str(selection.loc[0, "candidate"])
    conf_registry = {str(conf["name"]): dict(conf) for conf in all_confs}
    return {"selected_candidate": selected_name, "selected_configuration": conf_registry[selected_name], "selected_model": models[selected_name],
        "selected_history": histories[selected_name], "selected_training_summary": training_summaries[selected_name], 
        "selected_results": results[selected_name], "comparison": comparison, "selection": selection, "candidate_configurations": all_confs,
        "models": models, "histories": histories, "training_summaries": training_summaries, "denoising_diagnostics": denoising_diagnostics,
        "results": results, "refit_after_selection": False}
