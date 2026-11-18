"""
Smoke tests for the LaDIS package.

These tests use small artificial datasets so that they can run without access
to Bot-IoT, CICIoT23, CETEM, or project-specific files. Their purpose is to
verify package wiring and basic numerical validity, not synthetic-data quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ladis.models import derive_ladis_embedding_dimensions
from ladis.preprocessing import (fit_ordinal_categorical_encoder, fit_port_tokenizer, fit_reversible_numerical_preprocessor, 
    inverse_transform_categorical_features, inverse_transform_numerical_features, inverse_transform_port_tokens, transform_categorical_features,
    transform_numerical_features, transform_port_columns)
from ladis.training import fit_ladis_autoencoder, fit_ladis_clean_z0_diffusion
from ladis.latent import (encode_ladis_partition, fit_ladis_latent_standardization, freeze_ladis_autoencoder, inverse_ladis_latent_standardization,
    transform_ladis_latent_space)
from ladis.sampling import decode_ladis_latent_partition, sample_ladis_standardized_latents

RANDOM_STATE = 42


def _make_raw_preprocessing_data(n_rows: int = 128) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    return pd.DataFrame({"duration": rng.exponential(scale=2.0, size=n_rows), "packet_count": rng.integers(1, 12, size=n_rows), 
        "protocol": rng.choice(["tcp", "udp", "icmp"], size=n_rows), "src_port": rng.choice([0, 22, 53, 80, 443, 8080, 55000], size=n_rows),
        "dst_port": rng.choice([0, 22, 53, 80, 443, 8443, 60000], size=n_rows)})


def _make_model_ready_data(n_rows: int, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    num_data = pd.DataFrame({"num_0": rng.normal(0.0, 1.0, n_rows), "num_1": rng.normal(0.5, 1.2, n_rows), "num_2": rng.normal(-0.5, 0.8, n_rows)},
        dtype=np.float32)
    cat_data = pd.DataFrame({"cat_0": rng.integers(0, 3, n_rows), "cat_1": rng.integers(0, 4, n_rows)}, dtype=np.int64)
    return num_data, cat_data


def test_preprocessing_roundtrip_smoke() -> None:
    raw_data = _make_raw_preprocessing_data()
    num_artifact = fit_reversible_numerical_preprocessor(raw_data, cont_columns=["duration"], disc_columns=["packet_count"])
    transformed_num = transform_numerical_features(raw_data, num_artifact)
    restored_num = inverse_transform_numerical_features(transformed_num, num_artifact, round_disc=True, enforce_nonneg=True)
    assert transformed_num.columns.tolist() == ["duration", "packet_count"]
    assert np.isfinite(transformed_num.to_numpy(dtype=float)).all()
    assert np.allclose(restored_num["duration"].to_numpy(dtype=float), raw_data["duration"].to_numpy(dtype=float), rtol=1e-6, atol=1e-6)
    assert np.array_equal(restored_num["packet_count"].to_numpy(dtype=int), raw_data["packet_count"].to_numpy(dtype=int))
    cat_artifact = fit_ordinal_categorical_encoder(raw_data, cat_columns=["protocol"])
    encoded_cat, cat_diagnostics = transform_categorical_features(raw_data, cat_artifact)
    restored_cat = inverse_transform_categorical_features(encoded_cat, cat_artifact)
    assert encoded_cat.columns.tolist() == ["protocol"]
    assert int(cat_diagnostics["unknown_category_n"].sum()) == 0
    assert np.array_equal(restored_cat["protocol"].astype(str).to_numpy(), raw_data["protocol"].astype(str).to_numpy())
    port_artifact = fit_port_tokenizer(raw_data, port_columns=["src_port", "dst_port"], top_k_ports=4, min_exact_count=2)
    tokenized_ports, port_diagnostics = transform_port_columns(raw_data, port_artifact, split_name="train")
    restored_ports = inverse_transform_port_tokens(tokenized_ports, port_artifact, random_state=RANDOM_STATE)
    assert tokenized_ports.columns.tolist() == ["src_port", "dst_port"]
    assert int(port_diagnostics["unknown_token_rows"].sum()) == 0
    assert restored_ports.shape == tokenized_ports.shape
    restored_port_values = restored_ports.astype("int64").to_numpy()
    assert np.all(restored_port_values >= 0)
    assert np.all(restored_port_values <= 65535)


def test_core_ladis_pipeline_smoke() -> None:
    device = torch.device("cpu")
    num_columns = ["num_0", "num_1", "num_2"]
    cat_columns = ["cat_0", "cat_1"]
    cat_cards = {"cat_0": 3, "cat_1": 4}
    train_num, train_cat = _make_model_ready_data(128, seed=RANDOM_STATE)
    val_num, val_cat = _make_model_ready_data(48, seed=RANDOM_STATE + 1)
    embedding_dimensions = derive_ladis_embedding_dimensions(cat_columns=cat_columns, cat_cards=cat_cards, min_dim=2, max_dim=4)
    autoencoder_conf = {"embedding_dimensions": embedding_dimensions, "latent_dimension": 4, "encoder_hidden_dimensions": (16, 8),
        "decoder_hidden_dimensions": (8, 16), "dropout": 0.0, "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 32, "maximum_epochs": 3,
        "early_stopping_patience": 3, "minimum_improvement": 0.0, "gradient_clip_norm": 1.0, "numerical_loss_weight": 1.0,
        "categorical_loss_weight": 1.0, "variance_loss_weight": 0.0, "marginal_loss_weight": 0.0, "normalize_categorical_loss": True}
    autoencoder, ae_history, ae_summary = fit_ladis_autoencoder(train_num_data=train_num, train_cat_data=train_cat, val_num_data=val_num,
        val_cat_data=val_cat, num_columns=num_columns, cat_columns=cat_columns, cat_cards=cat_cards, conf=autoencoder_conf,
        random_state=RANDOM_STATE, device=device, verbose=False)
    assert not ae_history.empty
    assert int(ae_summary["latent_dimension"]) == 4
    autoencoder, freeze_summary = freeze_ladis_autoencoder(model=autoencoder)
    assert bool(freeze_summary["autoencoder_frozen"])
    assert int(freeze_summary["trainable_parameters_after_freezing"]) == 0
    raw_train_latent, _ = encode_ladis_partition(model=autoencoder, num_data=train_num, cat_data=train_cat, num_columns=num_columns,
        cat_columns=cat_columns, cat_cards=cat_cards, batch_size=32, device=device, data_name="smoke train")
    raw_val_latent, _ = encode_ladis_partition(model=autoencoder, num_data=val_num, cat_data=val_cat, num_columns=num_columns,
        cat_columns=cat_columns, cat_cards=cat_cards, batch_size=32, device=device, data_name="smoke validation")
    (standardized_train_latent, standardization_artifact, _,
        standardization_summary) = fit_ladis_latent_standardization(train_latent_data=raw_train_latent)
    standardized_val_latent, _ = transform_ladis_latent_space(latent_data=raw_val_latent, standardization_artifact=standardization_artifact,
        split_name="validation")
    assert bool(standardization_summary["latent_standardization_valid"])
    assert np.isfinite(standardized_train_latent.to_numpy(dtype=float)).all()
    assert np.isfinite(standardized_val_latent.to_numpy(dtype=float)).all()
    diffusion_conf = {"latent_dimension": 4, "hidden_dimensions": (16, 16), "timestep_embedding_dimension": 8, "dropout": 0.0, "num_timesteps": 8,
        "scheduler": "cosine", "prediction_target": "z0", "bound_predicted_z0": False, "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 32,
        "training_steps": 20, "validation_interval": 10, "validation_rows": 32, "minimum_improvement": 0.0, "gradient_clip_norm": 1.0}
    diffusion_model, diffusion_history, diffusion_summary = (fit_ladis_clean_z0_diffusion(train_latent_data=standardized_train_latent,
        val_latent_data=standardized_val_latent, conf=diffusion_conf, random_state=RANDOM_STATE, device=device, verbose=False))
    assert not diffusion_history.empty
    assert int(diffusion_summary["latent_dimension"]) == 4
    latent_columns = standardized_train_latent.columns.tolist()
    generated_standardized_latent, generation_summary = (sample_ladis_standardized_latents(diffusion_model=diffusion_model, num_rows=24,
        latent_columns=latent_columns, batch_size=12, random_state=RANDOM_STATE + 100, device=device))
    assert generated_standardized_latent.shape == (24, 4)
    assert bool(generation_summary["generated_values_are_finite"])
    generated_raw_latent = inverse_ladis_latent_standardization(standardized_latent_data=generated_standardized_latent,
        standardization_artifact=standardization_artifact)
    generated_num, generated_cat, decoding_summary = (decode_ladis_latent_partition(model=autoencoder, raw_latent_data=generated_raw_latent,
        num_columns=num_columns, cat_columns=cat_columns, cat_cards=cat_cards, batch_size=12, device=device, cat_sampling="sample",
        random_state=RANDOM_STATE + 200))
    assert generated_num.shape == (24, len(num_columns))
    assert generated_cat.shape == (24, len(cat_columns))
    assert generated_num.columns.tolist() == num_columns
    assert generated_cat.columns.tolist() == cat_columns
    assert np.isfinite(generated_num.to_numpy(dtype=float)).all()
    for column in cat_columns:
        values = generated_cat[column].to_numpy(dtype=int)
        assert np.all(values >= 0)
        assert np.all(values < cat_cards[column])
    assert bool(decoding_summary["numerical_values_are_finite"])
    assert int(decoding_summary["invalid_categorical_codes"]) == 0
    assert bool(decoding_summary["decoding_valid"])