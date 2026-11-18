"""
I/O tests for the LaDIS package.

These tests verify that a fitted LaDIS generator can be serialized, restored,
and reused without retraining. They also validate the complete generation
artifact and synthetic-data Parquet interfaces.

The tests use a small artificial mixed-type dataset. Their purpose is
reproducibility and persistence validation, not synthetic-data quality.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import QuantileTransformer

from ladis.io import (LADIS_GEN_ARTIFACT_KEYS, LADIS_MODEL_CHECKPOINT_KEYS, build_ladis_generation_artifact, build_ladis_model_checkpoint,
    collect_ladis_software_versions, load_ladis_for_generation, load_ladis_generation_artifact, load_ladis_model_checkpoint, load_ladis_synthetic_data,
    restore_ladis_models, save_ladis_generation_artifact, save_ladis_model_checkpoint, save_ladis_synthetic_data, validate_ladis_generation_artifact,
    validate_ladis_model_checkpoint)
from ladis.latent import (encode_ladis_partition, fit_ladis_latent_standardization, freeze_ladis_autoencoder, inverse_ladis_latent_standardization,
    transform_ladis_latent_space)
from ladis.models import derive_ladis_embedding_dimensions
from ladis.sampling import decode_ladis_latent_partition, sample_ladis_standardized_latents
from ladis.training import fit_ladis_autoencoder, fit_ladis_clean_z0_diffusion


RANDOM_STATE = 42
LATENT_DIMENSION = 4


def _make_model_ready_data(n_rows: int, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create finite numerical data and zero-based categorical codes."""

    rng = np.random.default_rng(seed)
    numerical = pd.DataFrame({"num_0": rng.normal(loc=0.0, scale=1.0, size=n_rows), "num_1": rng.normal(loc=0.5, scale=1.2, size=n_rows),
        "num_2": rng.normal(loc=-0.5, scale=0.8, size=n_rows)}, dtype=np.float32)
    categorical = pd.DataFrame({"cat_0": rng.integers(0, 3, size=n_rows), "cat_1": rng.integers(0, 4, size=n_rows)}, dtype=np.int64)
    return numerical, categorical


@pytest.fixture(scope="module")
def trained_ladis_bundle() -> dict[str, Any]:
    """
    Fit one compact LaDIS model and build a portable checkpoint.

    CPU is used deliberately so that deterministic save/load comparisons do not
    depend on GPU kernels and the test can run in a generic CI environment.
    """

    device = torch.device("cpu")
    numerical_columns = ["num_0", "num_1", "num_2"]
    categorical_columns = ["cat_0", "cat_1"]
    categorical_cardinalities = {"cat_0": 3, "cat_1": 4}
    model_feature_order = numerical_columns + categorical_columns
    train_num, train_cat = _make_model_ready_data(128, seed=RANDOM_STATE)
    val_num, val_cat = _make_model_ready_data(48, seed=RANDOM_STATE + 1)
    embedding_dimensions = derive_ladis_embedding_dimensions(cat_columns=categorical_columns, cat_cards=categorical_cardinalities, min_dim=2, max_dim=4)
    autoencoder_configuration = {"embedding_dimensions": embedding_dimensions, "latent_dimension": LATENT_DIMENSION,
        "encoder_hidden_dimensions": (16, 8), "decoder_hidden_dimensions": (8, 16), "dropout": 0.0, "learning_rate": 1e-3, "weight_decay": 0.0,
        "batch_size": 32, "maximum_epochs": 3, "early_stopping_patience": 3, "minimum_improvement": 0.0, "gradient_clip_norm": 1.0,
        "numerical_loss_weight": 1.0, "categorical_loss_weight": 1.0, "variance_loss_weight": 0.0, "marginal_loss_weight": 0.0,
        "normalize_categorical_loss": True}
    autoencoder, _, autoencoder_summary = fit_ladis_autoencoder(train_num_data=train_num, train_cat_data=train_cat, val_num_data=val_num,
        val_cat_data=val_cat, num_columns=numerical_columns, cat_columns=categorical_columns, cat_cards=categorical_cardinalities,
        conf=autoencoder_configuration, random_state=RANDOM_STATE, device=device, verbose=False)
    autoencoder, _ = freeze_ladis_autoencoder(model=autoencoder)
    raw_train_latent, _ = encode_ladis_partition(model=autoencoder, num_data=train_num, cat_data=train_cat, num_columns=numerical_columns,
        cat_columns=categorical_columns, cat_cards=categorical_cardinalities, batch_size=32, device=device, data_name="I/O smoke training")
    raw_val_latent, _ = encode_ladis_partition(model=autoencoder, num_data=val_num, cat_data=val_cat, num_columns=numerical_columns,
        cat_columns=categorical_columns, cat_cards=categorical_cardinalities, batch_size=32, device=device, data_name="I/O smoke validation")
    (standardized_train_latent, latent_standardization_artifact, latent_standardization_diagnostics, 
        latent_standardization_summary) = fit_ladis_latent_standardization(train_latent_data=raw_train_latent)
    standardized_val_latent, _ = transform_ladis_latent_space(latent_data=raw_val_latent, standardization_artifact=latent_standardization_artifact,
        split_name="validation")
    diffusion_configuration = {"latent_dimension": LATENT_DIMENSION, "hidden_dimensions": (16, 16), "timestep_embedding_dimension": 8,
        "dropout": 0.0, "num_timesteps": 8, "scheduler": "cosine", "prediction_target": "z0", "bound_predicted_z0": False, "learning_rate": 1e-3,
        "weight_decay": 0.0, "batch_size": 32, "training_steps": 20, "validation_interval": 10, "validation_rows": 32, "minimum_improvement": 0.0,
        "gradient_clip_norm": 1.0}
    diffusion_model, _, diffusion_summary = fit_ladis_clean_z0_diffusion(train_latent_data=standardized_train_latent,
        val_latent_data=standardized_val_latent, conf=diffusion_configuration, random_state=RANDOM_STATE, device=device, verbose=False)
    # Store an actual fitted sklearn transformer to verify that nontrivial preprocessing objects survive Joblib serialization
    quantile_transformer = QuantileTransformer(n_quantiles=min(32, len(train_num)), output_distribution="normal", subsample=len(train_num),
        random_state=RANDOM_STATE, copy=True)
    quantile_transformer.fit(train_num.to_numpy(dtype=np.float64))
    numerical_artifact = {"numerical_columns": numerical_columns, "continuous_columns": numerical_columns, "discrete_columns": [],
        "nonnegative_columns": []}
    categorical_generation_artifact = {"categorical_columns": categorical_columns, "column_information": {
            column: {"categories": list(range(categorical_cardinalities[column])), "number_of_categories": categorical_cardinalities[column]}
            for column in categorical_columns}}
    # No port variables are required by this artificial smoke dataset, but an explicit empty port artifact preserves the checkpoint interface
    port_artifact = {"port_columns": [], "column_information": {}}
    count_dequantization_artifact = {"columns": [], "column_information": {}}
    mode_dequantization_artifact = {"numerical_columns": numerical_columns, "column_information": {column: {"selected": False, "anchors": []}
        for column in numerical_columns}}
    checkpoint = build_ladis_model_checkpoint(autoencoder=autoencoder, diffusion_model=diffusion_model, ae_conf=autoencoder_configuration,
        diffusion_conf=diffusion_configuration, selected_diffusion_cand="reference_clean_z0_ladis", latent_dim=LATENT_DIMENSION, 
        cat_cards=categorical_cardinalities, num_columns=numerical_columns, cat_columns=categorical_columns, model_feature_order=model_feature_order,
        num_artifact=numerical_artifact, num_quantile_transformer=quantile_transformer, cat_gen_artifact=categorical_generation_artifact,
        port_artifact=port_artifact, count_dequantization_artifact=count_dequantization_artifact, 
        mode_dequantization_artifact=mode_dequantization_artifact, latent_standardization_artifact=latent_standardization_artifact,
        latent_standardization_diagnostics=latent_standardization_diagnostics, latent_standardization_summary=latent_standardization_summary)
    return {"device": device, "autoencoder": autoencoder, "diffusion_model": diffusion_model, "autoencoder_configuration": autoencoder_configuration,
        "diffusion_configuration": diffusion_configuration, "autoencoder_summary": autoencoder_summary, "diffusion_summary": diffusion_summary,
        "checkpoint": checkpoint, "numerical_columns": numerical_columns, "categorical_columns": categorical_columns, 
        "categorical_cardinalities": categorical_cardinalities, "model_feature_order": model_feature_order,
        "latent_standardization_artifact": latent_standardization_artifact, "latent_columns": standardized_train_latent.columns.tolist(),
        "training_rows": len(train_num), "validation_rows": len(val_num)}


def test_model_checkpoint_save_load_restore_is_reproducible(trained_ladis_bundle: dict[str, Any], tmp_path) -> None:
    """
    Save and restore LaDIS, then verify identical seeded generation.

    This is the principal I/O reproducibility test.
    """

    bundle = trained_ladis_bundle
    checkpoint = bundle["checkpoint"]
    device = bundle["device"]
    validate_ladis_model_checkpoint(checkpoint)
    assert list(checkpoint.keys())[: len(LADIS_MODEL_CHECKPOINT_KEYS)] == (LADIS_MODEL_CHECKPOINT_KEYS)
    checkpoint_path = tmp_path / "ladis_checkpoint.joblib"
    saved_path = save_ladis_model_checkpoint(checkpoint, checkpoint_path)
    assert saved_path == checkpoint_path
    assert saved_path.is_file()
    loaded_checkpoint = load_ladis_model_checkpoint(checkpoint_path)
    assert loaded_checkpoint["model_feature_order"] == bundle["model_feature_order"]
    assert (loaded_checkpoint["latent_diffusion_model"]["selected_candidate"] == "reference_clean_z0_ladis")
    (restored_autoencoder, restored_diffusion, restored_num_columns, restored_cat_columns, restored_cat_cards,
        restored_latent_columns) = restore_ladis_models(loaded_checkpoint, device=device)
    assert restored_num_columns == bundle["numerical_columns"]
    assert restored_cat_columns == bundle["categorical_columns"]
    assert restored_cat_cards == bundle["categorical_cardinalities"]
    assert restored_latent_columns == bundle["latent_columns"]
    assert not restored_autoencoder.training
    assert not restored_diffusion.training
    assert all(not parameter.requires_grad for parameter in restored_autoencoder.parameters())
    # Same model state + same CPU generator seed must reproduce the same standardized latent sample
    original_standardized_latent, _ = sample_ladis_standardized_latents(diffusion_model=bundle["diffusion_model"], num_rows=24,
        latent_columns=bundle["latent_columns"], batch_size=12, random_state=RANDOM_STATE + 100, device=device)
    restored_standardized_latent, _ = sample_ladis_standardized_latents(diffusion_model=restored_diffusion, num_rows=24,
        latent_columns=restored_latent_columns, batch_size=12, random_state=RANDOM_STATE + 100, device=device)
    np.testing.assert_allclose(restored_standardized_latent.to_numpy(dtype=np.float32), original_standardized_latent.to_numpy(dtype=np.float32),
        rtol=0.0, atol=0.0)
    original_raw_latent = inverse_ladis_latent_standardization(standardized_latent_data=original_standardized_latent,
        standardization_artifact=bundle["latent_standardization_artifact"])
    restored_raw_latent = inverse_ladis_latent_standardization(standardized_latent_data=restored_standardized_latent,
        standardization_artifact=loaded_checkpoint["preprocessing"]["latent_standardization_artifact"])
    original_num, original_cat, _ = decode_ladis_latent_partition(model=bundle["autoencoder"], raw_latent_data=original_raw_latent,
        num_columns=bundle["numerical_columns"], cat_columns=bundle["categorical_columns"], cat_cards=bundle["categorical_cardinalities"],
        batch_size=12, device=device, cat_sampling="sample", random_state=RANDOM_STATE + 200)
    restored_num, restored_cat, _ = decode_ladis_latent_partition(model=restored_autoencoder, raw_latent_data=restored_raw_latent,
        num_columns=restored_num_columns, cat_columns=restored_cat_columns, cat_cards=restored_cat_cards, batch_size=12, device=device,
        cat_sampling="sample", random_state=RANDOM_STATE + 200)
    np.testing.assert_allclose(restored_num.to_numpy(dtype=np.float32), original_num.to_numpy(dtype=np.float32), rtol=0.0, atol=0.0)
    pd.testing.assert_frame_equal(restored_cat.reset_index(drop=True), original_cat.reset_index(drop=True), check_dtype=True)
    # Also validate the public convenience loader
    (convenience_checkpoint, convenience_autoencoder, convenience_diffusion, convenience_num_columns, convenience_cat_columns, convenience_cat_cards,
        convenience_latent_columns) = load_ladis_for_generation(checkpoint_path, device=device)
    assert convenience_checkpoint["model_feature_order"] == bundle["model_feature_order"]
    assert convenience_num_columns == bundle["numerical_columns"]
    assert convenience_cat_columns == bundle["categorical_columns"]
    assert convenience_cat_cards == bundle["categorical_cardinalities"]
    assert convenience_latent_columns == bundle["latent_columns"]
    assert not convenience_autoencoder.training
    assert not convenience_diffusion.training


def test_generation_artifact_save_load_roundtrip(trained_ladis_bundle: dict[str, Any], tmp_path) -> None:
    """Build, save, load, and validate the complete LaDIS experiment artifact."""

    bundle = trained_ladis_bundle
    synthetic_data = pd.DataFrame({"num_0": [0.1, 0.2, 0.3], "num_1": [1.0, 1.1, 1.2], "num_2": [-0.1, -0.2, -0.3], "cat_0": [0, 1, 2],
        "cat_1": [1, 2, 3]})
    artifact = build_ladis_generation_artifact(model_checkpoint=bundle["checkpoint"], source_database="synthetic_smoke_test", traffic_cat="artificial",
        random_state=RANDOM_STATE, training_rows=bundle["training_rows"], val_rows=bundle["validation_rows"], test_rows_used=0,
        tuning={"latent_dimension_selection": "smoke_test", "diffusion_selection": "smoke_test"},
        final_training={"autoencoder_summary": bundle["autoencoder_summary"], "diffusion_summary": bundle["diffusion_summary"]},
        validation={"status": "smoke_test"}, reconstruction_corrections=pd.DataFrame({"column": [], "correction": []}),
        reconstruction_val_summary=pd.Series({"validation_passed": True}, name="value"), derived_feature_val=pd.DataFrame(),
        dtype_val=pd.DataFrame(), discrete_columns=[], port_columns=[], standardized_column_order=bundle["model_feature_order"],
        standardized_dtypes={column: str(dtype) for column, dtype in synthetic_data.dtypes.items()}, excluded_identifier_columns=[],
        recoverable_constant_columns=[], recoverable_constant_vals={}, provenance_columns=[], provenance_vals={}, 
        gen_data_path="synthetic_smoke.parquet", synth_data=synthetic_data, gen_model="LaDIS")
    validate_ladis_generation_artifact(artifact)
    assert list(artifact.keys()) == LADIS_GEN_ARTIFACT_KEYS
    assert artifact["generative_model"] == "LaDIS"
    assert artifact["synthetic_rows"] == len(synthetic_data)
    assert artifact["synthetic_columns"] == synthetic_data.columns.tolist()
    artifact_path = tmp_path / "ladis_generation_artifact.joblib"
    saved_path = save_ladis_generation_artifact(artifact, artifact_path)
    assert saved_path == artifact_path
    assert saved_path.is_file()
    loaded_artifact = load_ladis_generation_artifact(artifact_path)
    assert loaded_artifact["source_database"] == "synthetic_smoke_test"
    assert loaded_artifact["traffic_category"] == "artificial"
    assert loaded_artifact["random_state"] == RANDOM_STATE
    assert loaded_artifact["synthetic_rows"] == len(synthetic_data)
    # A complete generation artifact must also be accepted wherever a portable model checkpoint is accepted
    extracted_checkpoint = load_ladis_model_checkpoint(artifact_path)
    assert extracted_checkpoint["model_feature_order"] == bundle["model_feature_order"]
    assert (extracted_checkpoint["latent_diffusion_model"]["selected_candidate"] == "reference_clean_z0_ladis")


def test_synthetic_data_parquet_roundtrip(tmp_path) -> None:
    """Verify the public LaDIS synthetic-data save/load helpers."""

    synthetic_data = pd.DataFrame({"continuous_feature": pd.Series([0.25, 1.5, 2.75], dtype="float64"), "discrete_feature": pd.Series([1, 2, 3],
        dtype="int64"), "categorical_feature": pd.Series(["tcp", "udp", "icmp"], dtype="string")})
    parquet_path = tmp_path / "synthetic_ladis.parquet"
    saved_path = save_ladis_synthetic_data(synthetic_data, parquet_path)
    assert saved_path == parquet_path
    assert saved_path.is_file()
    loaded_data = load_ladis_synthetic_data(parquet_path)
    pd.testing.assert_frame_equal(loaded_data, synthetic_data, check_dtype=True)


def test_software_version_collection_smoke() -> None:
    """Verify that the environment metadata helper returns core versions."""

    versions = collect_ladis_software_versions()
    required_keys = {"python", "numpy", "pandas", "torch", "cuda", "joblib", "scikit_learn", "sdmetrics"}
    assert set(versions) == required_keys
    assert isinstance(versions["python"], str)
    assert isinstance(versions["numpy"], str)
    assert isinstance(versions["pandas"], str)
    assert isinstance(versions["torch"], str)