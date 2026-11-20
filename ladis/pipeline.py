"""
High-level training and generation pipeline for LaDIS.

The public pipeline composes the validated LaDIS building blocks without
changing the methodology used in the paper. Model selection uses training and
validation data only; the test partition is never used for fitting or selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .evaluation import (build_sdmetrics_metadata, diagnose_ladis_latent_space, evaluate_ladis_autoencoder_reconstruction,
    validate_ladis_latent_standardization_roundtrip)
from .io import build_ladis_model_checkpoint, load_ladis_for_generation, save_ladis_model_checkpoint
from .latent import (encode_ladis_partition, fit_ladis_latent_standardization, freeze_ladis_autoencoder, inverse_ladis_latent_standardization,
    transform_ladis_latent_space)
from .models import derive_ladis_embedding_dimensions
from .preparation import inverse_transform_port_tokens, prepare_dataset
from .preprocessing import (apply_ladis_count_dequantization, apply_ladis_mode_dequantization, fit_ladis_count_dequantization, fit_ladis_mode_dequantization,
    fit_ladis_quantile_transformer, fit_ordinal_categorical_encoder, fit_reversible_numerical_preprocessor, inverse_ladis_mode_dequantization,
    inverse_ladis_model_representation, inverse_ladis_quantile_space, prepare_ladis_partition, restrict_ladis_categories_to_training_support,
    transform_ladis_quantile_space)
from .reconstruction import reconstruct_to_original_space, validate_reconstructed_synthetic_data
from .sampling import decode_ladis_latent_partition, sample_ladis_standardized_latents
from .selection import run_ladis_clean_z0_selection, run_ladis_latent_dimension_selection
from .training import fit_ladis_autoencoder


PAPER_LADIS_CONFIG: dict[str, Any] = {"latent_dimension_candidates": (4, 8, 12, 16, 20, 24, 28, 32), "embedding_min_dimension": 2, "embedding_max_dimension": 16,
    "preprocessing": {"count_gap_fraction": 0.10, "mode_min_column_top_pct": 10.0, "mode_min_anchor_pct": 1.0, "mode_gap_fraction": 0.20,
    "mode_boundary_jitter_width": 1e-3, "quantile_count": 1_000}, "autoencoder": { "encoder_hidden_dimensions": (256, 128, 64),
    "decoder_hidden_dimensions": (64, 128, 256), "dropout": 0.0, "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 4_096, "maximum_epochs": 100,
    "early_stopping_patience": 10, "minimum_improvement": 1e-4, "gradient_clip_norm": 1.0, "numerical_loss_weight": 1.0, "variance_loss_weight": 0.0,
    "variance_loss_epsilon": 1e-6, "marginal_loss_weight": 0.0, "categorical_loss_weight": 1.0, "normalize_categorical_loss": True},
    "diffusion": {"hidden_dimensions": (256, 512, 512, 256), "timestep_embedding_dimension": 128, "dropout": 0.0, "num_timesteps": 500, "scheduler": "cosine",
    "prediction_target": "z0", "bound_predicted_z0": False, "learning_rate": 5e-4, "weight_decay": 1e-5, "batch_size": 4_096, "training_steps": 10_000,
    "gradient_clip_norm": 1.0, "validation_interval": 500, "validation_rows": 10_000, "minimum_improvement": 1e-6, "decoder_aware_loss_weight": 0.0},
    "selection": {"evaluation_rows": 10_000, "batch_size": 4_096, "diagnostic_timesteps": (0, 50, 125, 250, 375, 499), "diagnostic_rows": 10_000}}


def _deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive copy of ``base`` updated with user overrides."""

    result = deepcopy(dict(base))
    for key, value in updates.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def build_ladis_smoke_test_config() -> dict[str, Any]:
    """
    Return a deliberately reduced configuration for integration testing only.

    This configuration is not intended for scientific experiments or for
    reproducing paper results. It preserves the complete pipeline structure
    while drastically reducing training budgets.
    """

    return _deep_update(PAPER_LADIS_CONFIG, {"latent_dimension_candidates": (4, 8), "autoencoder": {"maximum_epochs": 3, "early_stopping_patience": 2},
        "diffusion": {"num_timesteps": 50, "training_steps": 20, "validation_interval": 10, "validation_rows": 64}, "selection": {"evaluation_rows": 64,
        "batch_size": 256, "diagnostic_timesteps": (0, 10, 25, 49), "diagnostic_rows": 64}})


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _validate_model_config(config: Mapping[str, Any]) -> None:
    latent_dims = tuple(int(value) for value in config["latent_dimension_candidates"])
    if not latent_dims or any(value <= 0 for value in latent_dims):
        raise ValueError("`latent_dimension_candidates` must contain positive dimensions.")
    if len(latent_dims) != len(set(latent_dims)):
        raise ValueError("`latent_dimension_candidates` contains duplicated dimensions.")
    if int(config["embedding_min_dimension"]) <= 0:
        raise ValueError("`embedding_min_dimension` must be positive.")
    if int(config["embedding_max_dimension"]) < int(config["embedding_min_dimension"]):
        raise ValueError("`embedding_max_dimension` cannot be smaller than `embedding_min_dimension`.")
    diffusion_conf = config["diffusion"]
    if str(diffusion_conf["prediction_target"]) != "z0":
        raise ValueError("The reference LaDIS pipeline requires direct clean-z0 prediction.")
    if bool(diffusion_conf["bound_predicted_z0"]):
        raise ValueError("The reference LaDIS pipeline requires unbounded clean-z0 predictions.")
    num_timesteps = int(diffusion_conf["num_timesteps"])
    if num_timesteps <= 1:
        raise ValueError("`num_timesteps` must be greater than one.")
    diagnostic_timesteps = tuple(int(value) for value in config["selection"]["diagnostic_timesteps"])
    invalid_timesteps = [value for value in diagnostic_timesteps if not 0 <= value < num_timesteps]
    if invalid_timesteps:
        raise ValueError(f"Diagnostic timesteps are outside the diffusion range: {invalid_timesteps}")


def _first_observed_values(series: pd.Series) -> list[Any]:
    values: list[Any] = []
    for value in pd.unique(series.dropna()):
        if isinstance(value, np.generic):
            value = value.item()
        values.append(value)
    return values


class LaDIS:
    """
    High-level LaDIS estimator and synthetic-data generator.

    The default model configuration reproduces the methodology reported in the
    paper. Dataset-specific semantic decisions remain in the preparation YAML.
    """

    def __init__(self, *, preparation_config_path: str | Path | None = None, model_config: Mapping[str, Any] | None = None, random_state: int | None = None,
        device: torch.device | str | None = None, include_provenance: bool = True, verbose: bool = True) -> None:

        if random_state is not None and (isinstance(random_state, bool) or not isinstance(random_state, int)):
            raise TypeError("`random_state` must be an integer or None.")
        if not isinstance(include_provenance, bool):
            raise TypeError("`include_provenance` must be Boolean.")
        if not isinstance(verbose, bool):
            raise TypeError("`verbose` must be Boolean.")
        self.preparation_config_path = preparation_config_path
        self.model_config = _deep_update(PAPER_LADIS_CONFIG, {} if model_config is None else model_config)
        _validate_model_config(self.model_config)
        self.random_state = random_state
        self.device = _resolve_device(device)
        self.include_provenance = include_provenance
        self.verbose = verbose
        self.is_fitted_ = False

    def fit(self, source: str | Path | pd.DataFrame) -> "LaDIS":
        """Fit the complete two-stage LaDIS pipeline using train/validation only."""

        preparation_result = prepare_dataset(source, config_path=self.preparation_config_path)
        prepared_splits = preparation_result["data_splits"]
        preparation_artifacts = preparation_result["artifacts"]
        preparation_config = preparation_result["metadata"]["preparation_config"]
        random_state = int(preparation_config["split"]["random_state"] if self.random_state is None else self.random_state)
        feature_order = prepared_splits["train"].columns.tolist()
        model_types = preparation_artifacts["variable_types"]["model"]
        numerical_set = set(model_types["continuous"] + model_types["discrete_count"])
        categorical_set = set(model_types["binary"] + model_types["categorical"])
        num_columns = [column for column in feature_order if column in numerical_set]
        cat_columns = [column for column in feature_order if column in categorical_set]
        if set(num_columns).intersection(cat_columns):
            raise RuntimeError("Numerical and categorical LaDIS variables overlap.")
        if set(num_columns).union(cat_columns) != set(feature_order):
            raise RuntimeError("Numerical and categorical LaDIS variables do not cover the model schema.")
        if self.verbose:
            print("Preparing train-fitted LaDIS model representation...")
        num_artifact = fit_reversible_numerical_preprocessor(prepared_splits["train"], cont_columns=model_types["continuous"],
            disc_columns=model_types["discrete_count"], skewness_log_thr=float(preparation_config["model_preprocessing"]["skewness_log_threshold"]))
        port_artifact = preparation_artifacts["ports"]
        port_columns = list(port_artifact["port_columns"])
        predefined_port_categories = {column: list(port_artifact["column_artifacts"][column]["token_categories"]) for column in port_columns}
        cat_artifact = fit_ordinal_categorical_encoder(prepared_splits["train"], cat_columns=cat_columns, predefined_cats=predefined_port_categories)
        training_cat_data = prepared_splits["train"].loc[:, cat_columns].reset_index(drop=True)
        cat_artifact, cat_restriction_summary = restrict_ladis_categories_to_training_support(cat_artifact=cat_artifact, training_cat_data=training_cat_data,
            cat_columns=cat_columns)
        cat_cards = {column: int(cat_artifact["column_information"][column]["number_of_categories"]) for column in cat_columns}
        partition_outputs: dict[str, tuple[Any, ...]] = {}
        for split_name in ("train", "validation", "test"):
            partition_outputs[split_name] = prepare_ladis_partition(df_prepared=prepared_splits[split_name], feature_order=feature_order, num_columns=num_columns,
                cat_columns=cat_columns, cat_cards=cat_cards, num_artifact=num_artifact, cat_artifact=cat_artifact, split_name=split_name)
        train_num, train_cat, train_eval, train_rep_diag, train_rep_summary = partition_outputs["train"]
        val_num, val_cat, val_eval, val_rep_diag, val_rep_summary = partition_outputs["validation"]
        test_num, test_cat, test_eval, test_rep_diag, test_rep_summary = partition_outputs["test"]
        prep_conf = self.model_config["preprocessing"]
        train_count, count_artifact, count_diagnostics = fit_ladis_count_dequantization(train_num_data=train_num,
            disc_count_columns=model_types["discrete_count"], random_state=random_state, gap_fraction=float(prep_conf["count_gap_fraction"]))
        val_count = apply_ladis_count_dequantization(num_data=val_num, dequantization_artifact=count_artifact, random_state=random_state + 1,
            split_name="validation")
        test_count = apply_ladis_count_dequantization(num_data=test_num, dequantization_artifact=count_artifact, random_state=random_state + 2, split_name="test")
        train_mode, mode_artifact, mode_diagnostics = fit_ladis_mode_dequantization(train_num_data=train_count, excluded_columns=model_types["discrete_count"],
            random_state=random_state + 10, min_column_top_pct=float(prep_conf["mode_min_column_top_pct"]),
            min_anchor_pct=float(prep_conf["mode_min_anchor_pct"]), gap_fraction=float(prep_conf["mode_gap_fraction"]),
            boundary_jitter_width=float(prep_conf["mode_boundary_jitter_width"]))
        val_mode = apply_ladis_mode_dequantization(num_data=val_count, dequantization_artifact=mode_artifact, random_state=random_state + 11,
            split_name="validation")
        test_mode = apply_ladis_mode_dequantization(num_data=test_count, dequantization_artifact=mode_artifact, random_state=random_state + 12, split_name="test")
        quantile_transformer, train_quantile, quantile_summary = fit_ladis_quantile_transformer(train_num_data=train_mode, random_state=random_state,
            n_quantiles=int(prep_conf["quantile_count"]))
        val_quantile = transform_ladis_quantile_space(num_data=val_mode, quantile_transf=quantile_transformer, expected_columns=num_columns,
            split_name="validation")
        test_quantile = transform_ladis_quantile_space(num_data=test_mode, quantile_transf=quantile_transformer, expected_columns=num_columns, split_name="test")
        embedding_dimensions = derive_ladis_embedding_dimensions(cat_columns=cat_columns, cat_cards=cat_cards,
            min_dim=int(self.model_config["embedding_min_dimension"]), max_dim=int(self.model_config["embedding_max_dimension"]))
        ae_base_conf = deepcopy(self.model_config["autoencoder"])
        ae_base_conf["embedding_dimensions"] = embedding_dimensions
        diffusion_base_conf = deepcopy(self.model_config["diffusion"])
        selection_conf = self.model_config["selection"]
        if self.verbose:
            print("Selecting the LaDIS latent dimension...")
        latent_selection = run_ladis_latent_dimension_selection(train_num_data=train_quantile, train_cat_data=train_cat, val_num_data=val_quantile,
            val_cat_data=val_cat, train_eval_data=train_eval, val_eval_data=val_eval, feature_order=feature_order, num_columns=num_columns,
            cat_columns=cat_columns, cat_cards=cat_cards, num_artifact=num_artifact, cat_artifact=cat_artifact, num_quantile_transformer=quantile_transformer,
            mode_dequantization_artifact=mode_artifact, latent_dim_cands=self.model_config["latent_dimension_candidates"], ae_base_conf=ae_base_conf,
            diffusion_base_conf=diffusion_base_conf, random_state=random_state, device=self.device, evaluation_rows=int(selection_conf["evaluation_rows"]),
            batch_size=int(selection_conf["batch_size"]), verbose=self.verbose)
        selected_latent_dim = int(latent_selection["selected_latent_dimension"])
        if self.verbose:
            print(f"Retraining the definitive autoencoder at latent dimension {selected_latent_dim}...")
        definitive_ae_conf = deepcopy(ae_base_conf)
        definitive_ae_conf["name"] = f"reference_ae_latent{selected_latent_dim}"
        definitive_ae_conf["latent_dimension"] = selected_latent_dim
        autoencoder, ae_history, ae_training_summary = fit_ladis_autoencoder(train_num_data=train_quantile, train_cat_data=train_cat, val_num_data=val_quantile,
            val_cat_data=val_cat, num_columns=num_columns, cat_columns=cat_columns, cat_cards=cat_cards, conf=definitive_ae_conf, random_state=random_state,
            device=self.device, verbose=self.verbose)
        train_ae_summary, train_ae_num_diag, train_ae_cat_diag = evaluate_ladis_autoencoder_reconstruction(model=autoencoder, num_data=train_quantile,
            cat_data=train_cat, num_columns=num_columns, cat_columns=cat_columns, cat_cards=cat_cards, batch_size=int(ae_base_conf["batch_size"]),
            device=self.device, split_name="train")
        val_ae_summary, val_ae_num_diag, val_ae_cat_diag = evaluate_ladis_autoencoder_reconstruction(model=autoencoder, num_data=val_quantile, cat_data=val_cat,
            num_columns=num_columns, cat_columns=cat_columns, cat_cards=cat_cards, batch_size=int(ae_base_conf["batch_size"]), device=self.device,
            split_name="validation")
        autoencoder, ae_freeze_summary = freeze_ladis_autoencoder(model=autoencoder)
        train_z_raw, train_z_raw_summary = encode_ladis_partition(model=autoencoder, num_data=train_quantile, cat_data=train_cat, num_columns=num_columns,
            cat_columns=cat_columns, cat_cards=cat_cards, batch_size=int(selection_conf["batch_size"]), device=self.device, data_name="train")
        val_z_raw, val_z_raw_summary = encode_ladis_partition(model=autoencoder, num_data=val_quantile, cat_data=val_cat, num_columns=num_columns,
            cat_columns=cat_columns, cat_cards=cat_cards, batch_size=int(selection_conf["batch_size"]), device=self.device, data_name="validation")
        train_z0, latent_standardization_artifact, latent_standardization_diagnostics, latent_standardization_summary = (
            fit_ladis_latent_standardization(train_latent_data=train_z_raw, min_coordinate_std=1e-6))
        val_z0, val_z0_summary = transform_ladis_latent_space(latent_data=val_z_raw, standardization_artifact=latent_standardization_artifact,
            split_name="validation")
        latent_roundtrip_summary = validate_ladis_latent_standardization_roundtrip(raw_latent_data=train_z_raw, standardized_latent_data=train_z0,
            standardization_artifact=latent_standardization_artifact, split_name="train")
        latent_coordinate_diag, latent_corr_matrix, latent_space_summary = diagnose_ladis_latent_space(train_latent_data=train_z0, val_latent_data=val_z0)
        reference_diffusion_conf = deepcopy(diffusion_base_conf)
        reference_diffusion_conf["name"] = "reference_clean_z0_ladis"
        reference_diffusion_conf["latent_dimension"] = selected_latent_dim
        reference_diffusion_conf["validation_rows"] = min(int(reference_diffusion_conf.get("validation_rows", len(val_z0))), len(val_z0))
        if self.verbose:
            print("Selecting the definitive clean-z0 denoiser...")
        denoiser_selection = run_ladis_clean_z0_selection(train_latent_data=train_z0, val_latent_data=val_z0, autoencoder=autoencoder,
            ref_conf=reference_diffusion_conf, real_val_data=latent_selection["real_validation_reference"], fitted_training_data=train_eval,
            metadata=latent_selection.get("sdmetrics_metadata", build_sdmetrics_metadata(feature_order=feature_order, cat_columns=cat_columns)),
            latent_standardization_artifact=latent_standardization_artifact, num_quantile_transformer=quantile_transformer,
            mode_dequantization_artifact=mode_artifact, num_artifact=num_artifact, cat_artifact=cat_artifact, feature_order=feature_order,
            num_columns=num_columns, cat_columns=cat_columns, cat_cards=cat_cards, random_state=random_state, device=self.device,
            batch_size=int(selection_conf["batch_size"]), diagnostic_timesteps=selection_conf["diagnostic_timesteps"],
            diagnostic_rows=int(selection_conf["diagnostic_rows"]), verbose=self.verbose)
        diffusion_model = denoiser_selection["selected_model"]
        selected_diffusion_conf = dict(denoiser_selection["selected_configuration"])
        selected_diffusion_name = str(denoiser_selection["selected_candidate"])
        standardized_column_order = preparation_artifacts.get("standardized_column_order")
        standardized_dtypes = preparation_artifacts.get("standardized_dtypes")
        if standardized_column_order is None or standardized_dtypes is None:
            raise RuntimeError("The preparation artifacts do not contain `standardized_column_order` and `standardized_dtypes`. "
                "Use the updated preparation pipeline before fitting the high-level LaDIS API.")
        feature_selection_artifact = preparation_artifacts["feature_selection"]
        derived_feature_artifact = preparation_artifacts["derived_features"]
        recov_cte_vals = dict(feature_selection_artifact["rec_cte_vals"])
        prov_columns = list(feature_selection_artifact["prov_columns"])
        prov_vals = dict(feature_selection_artifact["prov_vals"])
        omitted_columns = list(dict.fromkeys(list(feature_selection_artifact["excluded_id_columns"])
            + list(feature_selection_artifact.get("excluded_nonrec_cte_columns", []))))
        strictly_positive_minimums: dict[str, float] = {}
        for rule in derived_feature_artifact["reconstruction_rules"].values():
            if rule["operation"] != "ratio":
                continue
            denominator = str(rule["denominator"])
            if denominator in strictly_positive_minimums:
                continue
            denominator_values = pd.to_numeric(prepared_splits["train"][denominator], errors="coerce").to_numpy(dtype=float)
            positive_values = denominator_values[np.isfinite(denominator_values) & (denominator_values > 0.0)]
            if positive_values.size == 0:
                raise RuntimeError(f"Rate denominator '{denominator}' has no positive training value for reconstruction.")
            strictly_positive_minimums[denominator] = float(positive_values.min())
        binary_allowed_values = {column: _first_observed_values(prepared_splits["train"][column]) for column in model_types["binary"]}
        validate_binary = bool(binary_allowed_values) and all(len(values) == 2 for values in binary_allowed_values.values())
        if "protocol" in train_eval.columns:
            allowed_protocol_values = _first_observed_values(train_eval["protocol"])
        elif "protocol" in recov_cte_vals:
            allowed_protocol_values = [recov_cte_vals["protocol"]]
        else:
            allowed_protocol_values = []
        reconstruction_metadata = {"derived_feature_artifact": derived_feature_artifact,
            "recoverable_constant_columns": list(feature_selection_artifact["rec_cte_columns"]), "recoverable_constant_values": recov_cte_vals,
            "provenance_columns": prov_columns, "provenance_values": prov_vals, "standardized_column_order": list(standardized_column_order),
            "standardized_dtypes": dict(standardized_dtypes), "excluded_columns": omitted_columns, "discrete_count_columns": list(model_types["discrete_count"]),
            "nonnegative_columns": list(num_artifact["nonnegative_columns"]), "strictly_positive_minimums": strictly_positive_minimums,
            "include_provenance": bool(self.include_provenance), "traffic_label": recov_cte_vals.get("label"), "allowed_protocol_values": allowed_protocol_values,
            "binary_validation_enabled": validate_binary, "binary_allowed_values": binary_allowed_values,
            "dataset": deepcopy(preparation_result["metadata"]["dataset"]), "preparation_config": deepcopy(preparation_config),
            "pipeline_random_state": random_state, "pipeline_model_config": deepcopy(self.model_config),
            "generation_batch_size": int(selection_conf["batch_size"])}
        checkpoint = build_ladis_model_checkpoint(autoencoder=autoencoder, diffusion_model=diffusion_model, ae_conf=definitive_ae_conf,
            diffusion_conf=selected_diffusion_conf, selected_diffusion_cand=selected_diffusion_name, latent_dim=selected_latent_dim, cat_cards=cat_cards,
            num_columns=num_columns, cat_columns=cat_columns, model_feature_order=feature_order, num_artifact=num_artifact,
            num_quantile_transformer=quantile_transformer, cat_gen_artifact=cat_artifact, port_artifact=port_artifact,
            count_dequantization_artifact=count_artifact, mode_dequantization_artifact=mode_artifact,
            latent_standardization_artifact=latent_standardization_artifact, binary_columns=model_types["binary"], binary_val_enabled=validate_binary,
            binary_allowed_vals=binary_allowed_values, count_dequantization_diagnostics=count_diagnostics, mode_dequantization_diagnostics=mode_diagnostics,
            quantile_fit_summary=quantile_summary, cat_restriction_summary=cat_restriction_summary,
            latent_standardization_diagnostics=latent_standardization_diagnostics, latent_standardization_summary=latent_standardization_summary,
            latent_standardization_roundtrip=latent_roundtrip_summary, reconstruction=reconstruction_metadata)
        self.random_state_ = random_state
        self.preparation_result_ = preparation_result
        self.prepared_splits_ = prepared_splits
        self.feature_order_ = feature_order
        self.model_types_ = model_types
        self.num_columns_ = num_columns
        self.cat_columns_ = cat_columns
        self.cat_cards_ = cat_cards
        self.num_artifact_ = num_artifact
        self.cat_artifact_ = cat_artifact
        self.port_artifact_ = port_artifact
        self.count_dequantization_artifact_ = count_artifact
        self.mode_dequantization_artifact_ = mode_artifact
        self.quantile_transformer_ = quantile_transformer
        self.latent_standardization_artifact_ = latent_standardization_artifact
        self.autoencoder_ = autoencoder
        self.diffusion_model_ = diffusion_model
        self.latent_columns_ = [f"z_{dimension:02d}" for dimension in range(selected_latent_dim)]
        self.selected_latent_dimension_ = selected_latent_dim
        self.selected_diffusion_candidate_ = selected_diffusion_name
        self.selected_diffusion_configuration_ = selected_diffusion_conf
        self.checkpoint_ = checkpoint
        self.reconstruction_metadata_ = reconstruction_metadata
        self.latent_dimension_selection_ = latent_selection
        self.denoiser_selection_ = denoiser_selection
        self.training_diagnostics_ = {"partition_summaries": {"train": train_rep_summary, "validation": val_rep_summary, "test": test_rep_summary},
            "partition_diagnostics": {"train": train_rep_diag, "validation": val_rep_diag, "test": test_rep_diag}, "autoencoder": {"history": ae_history,
            "training_summary": ae_training_summary, "freeze_summary": ae_freeze_summary, "train_reconstruction_summary": train_ae_summary,
            "validation_reconstruction_summary": val_ae_summary, "train_numerical_diagnostics": train_ae_num_diag,
            "train_categorical_diagnostics": train_ae_cat_diag, "validation_numerical_diagnostics": val_ae_num_diag,
            "validation_categorical_diagnostics": val_ae_cat_diag}, "latent": {"train_raw_summary": train_z_raw_summary,
            "validation_raw_summary": val_z_raw_summary, "validation_standardized_summary": val_z0_summary,
            "standardization_summary": latent_standardization_summary, "standardization_diagnostics": latent_standardization_diagnostics,
            "roundtrip_summary": latent_roundtrip_summary, "coordinate_diagnostics": latent_coordinate_diag, "correlation_matrix": latent_corr_matrix,
            "space_summary": latent_space_summary}}
        self.model_data_ = {"train_numerical": train_quantile, "train_categorical": train_cat, "validation_numerical": val_quantile,
            "validation_categorical": val_cat, "test_numerical": test_quantile, "test_categorical": test_cat, "train_evaluation": train_eval,
            "validation_evaluation": val_eval, "test_evaluation": test_eval, "train_latent": train_z0, "validation_latent": val_z0}
        self.is_fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("The LaDIS model has not been fitted or loaded.")

    def sample(self, num_rows: int, *, random_state: int | None = None,
        include_provenance: bool | None = None, validate: bool = True,
        return_diagnostics: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        """Generate and reconstruct synthetic observations with the fitted LaDIS model."""

        self._check_fitted()
        if isinstance(num_rows, bool):
            raise TypeError("`num_rows` must be an integer.")
        num_rows = int(num_rows)
        if num_rows <= 0:
            raise ValueError("`num_rows` must be positive.")
        if random_state is not None and (isinstance(random_state, bool) or not isinstance(random_state, int)):
            raise TypeError("`random_state` must be an integer or None.")
        seed = int(self.random_state_ if random_state is None else random_state)
        generation_batch_size = min(int(self.reconstruction_metadata_.get("generation_batch_size", 4_096)), num_rows)
        if include_provenance is None:
            include_provenance = bool(self.reconstruction_metadata_.get("include_provenance", True))
        synth_z0, latent_sampling_summary = sample_ladis_standardized_latents(diffusion_model=self.diffusion_model_, num_rows=num_rows,
            latent_columns=self.latent_columns_, batch_size=generation_batch_size, random_state=seed, device=self.device)
        synth_z_raw = inverse_ladis_latent_standardization(standardized_latent_data=synth_z0, standardization_artifact=self.latent_standardization_artifact_)
        synth_num_quantile, synth_cat_codes, decoder_summary = decode_ladis_latent_partition(model=self.autoencoder_, raw_latent_data=synth_z_raw,
            num_columns=self.num_columns_, cat_columns=self.cat_columns_, cat_cards=self.cat_cards_, batch_size=generation_batch_size, device=self.device,
            cat_sampling="sample", random_state=seed)
        synth_num_prequantile = inverse_ladis_quantile_space(quantile_model_data=synth_num_quantile, quantile_transf=self.quantile_transformer_,
            expected_columns=self.num_columns_)
        synth_num_preproc = inverse_ladis_mode_dequantization(prequantile_num_data=synth_num_prequantile,
            dequantization_artifact=self.mode_dequantization_artifact_)
        synth_model_space = inverse_ladis_model_representation(num_model_data=synth_num_preproc, cat_model_data=synth_cat_codes, num_columns=self.num_columns_,
            cat_columns=self.cat_columns_, feature_order=self.feature_order_, num_artifact=self.num_artifact_, cat_artifact=self.cat_artifact_,
            round_disc=False, enforce_nonneg=False)
        port_columns = list(self.port_artifact_.get("port_columns", []))
        if port_columns:
            restored_port_data = inverse_transform_port_tokens(synth_model_space.loc[:, port_columns], self.port_artifact_, random_state=seed)
            restored_ports = restored_port_data.loc[:, port_columns].reset_index(drop=True)
        else:
            restored_ports = pd.DataFrame(index=np.arange(num_rows))
        rec = self.reconstruction_metadata_
        reconstructed, correction_report = reconstruct_to_original_space(synth_model_space=synth_model_space, restored_ports=restored_ports,
            model_feature_order=self.feature_order_, disc_count_columns=rec["discrete_count_columns"], nonneg_columns=rec["nonnegative_columns"],
            strictly_pos_mins=rec["strictly_positive_minimums"], derived_feature_artifact=rec["derived_feature_artifact"],
            recov_cte_vals=rec["recoverable_constant_values"], prov_columns=rec["provenance_columns"], prov_vals=rec["provenance_values"],
            standardized_column_order=rec["standardized_column_order"], standardized_dtypes=rec["standardized_dtypes"],
            excluded_id_columns=rec["excluded_columns"], include_prov=bool(include_provenance))
        reconstruction_validation = None
        derived_validation = None
        dtype_validation = None
        if validate:
            traffic_label = rec.get("traffic_label")
            allowed_protocol_values = list(rec.get("allowed_protocol_values", []))
            if traffic_label is not None and "protocol" in reconstructed.columns and allowed_protocol_values:
                reconstruction_validation, derived_validation, dtype_validation = validate_reconstructed_synthetic_data(rec_df=reconstructed,
                    expected_column_order=rec["standardized_column_order"], expected_dtypes=rec["standardized_dtypes"],
                    excluded_id_columns=rec["excluded_columns"], derived_feature_artifact=rec["derived_feature_artifact"], traffic_label=traffic_label,
                    allowed_protocol_vals=allowed_protocol_values, include_prov=bool(include_provenance), prov_columns=rec["provenance_columns"],
                    prov_vals=rec["provenance_values"], val_binary_columns=bool(rec.get("binary_validation_enabled", False)),
                    binary_allowed_vals=(rec.get("binary_allowed_values") if rec.get("binary_validation_enabled", False) else None))
                required_true = ["schema_matches", "dtypes_match", "label_is_constant", "protocol_values_are_valid",
                    "ports_are_valid", "derived_relationships_are_valid"]
                if bool(rec.get("binary_validation_enabled", False)):
                    required_true.extend(["binary_validation_performed", "binary_values_are_valid"])
                failed_true = [name for name in required_true if not bool(reconstruction_validation[name])]
                if failed_true or int(reconstruction_validation["missing_values"]) != 0 or int(reconstruction_validation["infinite_values"]) != 0:
                    raise RuntimeError(f"Generated LaDIS data failed reconstruction validation: {failed_true}")
            else:
                expected_columns = [column for column in rec["standardized_column_order"] if column not in set(rec["excluded_columns"])
                    and (bool(include_provenance) or column not in set(rec["provenance_columns"]))
                    and not (column in set(rec["provenance_columns"]) and column not in set(rec["provenance_values"]))]
                if reconstructed.columns.tolist() != expected_columns:
                    raise RuntimeError("Generated LaDIS data do not preserve the reconstructable schema.")
                if reconstructed.isna().any().any():
                    raise RuntimeError("Generated LaDIS data contain missing values.")
                numeric_columns = reconstructed.select_dtypes(include=np.number).columns
                if len(numeric_columns) and np.isinf(reconstructed[numeric_columns].to_numpy(dtype=float)).any():
                    raise RuntimeError("Generated LaDIS data contain infinite numerical values.")
        diagnostics = {"latent_sampling_summary": latent_sampling_summary, "decoder_summary": decoder_summary, "reconstruction_corrections": correction_report,
            "reconstruction_validation_summary": reconstruction_validation, "derived_feature_validation": derived_validation, "dtype_validation": dtype_validation}
        self.last_sample_diagnostics_ = diagnostics
        if return_diagnostics:
            return reconstructed, diagnostics
        return reconstructed

    def fit_sample(self, source: str | Path | pd.DataFrame, num_rows: int, *, random_state: int | None = None, validate: bool = True) -> pd.DataFrame:
        """Fit LaDIS and immediately generate a reconstructed synthetic table."""

        self.fit(source)
        return self.sample(num_rows, random_state=random_state, validate=validate)

    def save(self, path: str | Path, *, compress: int = 3) -> Path:
        """Save the fitted portable LaDIS checkpoint."""

        self._check_fitted()
        return save_ladis_model_checkpoint(self.checkpoint_, path, compress=compress)

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | str | None = None, random_state: int | None = None, verbose: bool = True) -> "LaDIS":
        """Restore a saved LaDIS checkpoint as a generation-ready object."""

        if random_state is not None and (isinstance(random_state, bool) or not isinstance(random_state, int)):
            raise TypeError("`random_state` must be an integer or None.")
        resolved_device = _resolve_device(device)
        checkpoint, autoencoder, diffusion_model, num_columns, cat_columns, cat_cards, latent_columns = load_ladis_for_generation(path, device=resolved_device)
        if "reconstruction" not in checkpoint:
            raise RuntimeError("The stored checkpoint does not contain reconstruction metadata and cannot provide fully reconstructed samples.")
        reconstruction = dict(checkpoint["reconstruction"])
        stored_random_state = int(reconstruction.get("pipeline_random_state", 42))
        resolved_random_state = stored_random_state if random_state is None else int(random_state)
        stored_model_config = reconstruction.get("pipeline_model_config")
        obj = cls(preparation_config_path=None, model_config=stored_model_config, random_state=resolved_random_state, device=resolved_device,
            include_provenance=bool(reconstruction.get("include_provenance", True)), verbose=verbose)
        preprocessing = checkpoint["preprocessing"]
        obj.random_state_ = resolved_random_state
        obj.feature_order_ = list(checkpoint["model_feature_order"])
        obj.num_columns_ = list(num_columns)
        obj.cat_columns_ = list(cat_columns)
        obj.cat_cards_ = dict(cat_cards)
        obj.num_artifact_ = preprocessing["numerical_artifact"]
        obj.cat_artifact_ = preprocessing["categorical_generation_artifact"]
        obj.port_artifact_ = preprocessing["port_artifact"]
        obj.count_dequantization_artifact_ = preprocessing["count_dequantization_artifact"]
        obj.mode_dequantization_artifact_ = preprocessing["mode_dequantization_artifact"]
        obj.quantile_transformer_ = preprocessing["numerical_quantile_transformer"]
        obj.latent_standardization_artifact_ = preprocessing["latent_standardization_artifact"]
        obj.autoencoder_ = autoencoder
        obj.diffusion_model_ = diffusion_model
        obj.latent_columns_ = list(latent_columns)
        obj.selected_latent_dimension_ = int(checkpoint["representation_model"]["latent_dimension"])
        obj.selected_diffusion_candidate_ = str(checkpoint["latent_diffusion_model"].get("selected_candidate", "unknown"))
        obj.selected_diffusion_configuration_ = dict(checkpoint["latent_diffusion_model"]["configuration"])
        obj.checkpoint_ = checkpoint
        obj.reconstruction_metadata_ = reconstruction
        obj.is_fitted_ = True
        return obj
