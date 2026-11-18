"""
Minimal LaDIS example with artificial mixed-type data.

This example demonstrates the core LaDIS train-and-sample workflow without
requiring any external IoT dataset. Numerical variables are assumed to be
already represented in the model space expected by LaDIS, and categorical
variables are zero-based integer codes.

The dataset-specific network-traffic preprocessing and reconstruction example
should be added separately once the core package smoke test passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ladis.models import derive_ladis_embedding_dimensions
from ladis.training import fit_ladis_autoencoder, fit_ladis_clean_z0_diffusion
from ladis.latent import (encode_ladis_partition, fit_ladis_latent_standardization, freeze_ladis_autoencoder,
    inverse_ladis_latent_standardization, transform_ladis_latent_space)
from ladis.sampling import decode_ladis_latent_partition, sample_ladis_standardized_latents

RANDOM_STATE = 42


def make_example_data(n_rows: int, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    numerical = pd.DataFrame({"feature_0": rng.normal(size=n_rows), "feature_1": rng.normal(loc=0.5, scale=1.5, size=n_rows),
        "feature_2": rng.normal(loc=-0.5, scale=0.75, size=n_rows)}, dtype=np.float32)
    categorical = pd.DataFrame({"protocol_code": rng.integers(0, 3, size=n_rows), "port_group_code": rng.integers(0, 4, size=n_rows)}, dtype=np.int64)
    return numerical, categorical


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_columns = ["feature_0", "feature_1", "feature_2"]
    cat_columns = ["protocol_code", "port_group_code"]
    cat_cards = {"protocol_code": 3, "port_group_code": 4}
    train_num, train_cat = make_example_data(256, seed=RANDOM_STATE)
    val_num, val_cat = make_example_data(64, seed=RANDOM_STATE + 1)
    embedding_dimensions = derive_ladis_embedding_dimensions(cat_columns=cat_columns, cat_cards=cat_cards)
    autoencoder_conf = {"embedding_dimensions": embedding_dimensions, "latent_dimension": 4, "encoder_hidden_dimensions": (32, 16),
        "decoder_hidden_dimensions": (16, 32), "dropout": 0.0, "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 64, "maximum_epochs": 10,
        "early_stopping_patience": 5, "minimum_improvement": 1e-4, "gradient_clip_norm": 1.0, "numerical_loss_weight": 1.0,
        "categorical_loss_weight": 1.0, "variance_loss_weight": 0.0, "marginal_loss_weight": 0.0, "normalize_categorical_loss": True}
    autoencoder, _, ae_summary = fit_ladis_autoencoder(train_num_data=train_num, train_cat_data=train_cat, val_num_data=val_num, val_cat_data=val_cat,
        num_columns=num_columns, cat_columns=cat_columns, cat_cards=cat_cards, conf=autoencoder_conf, random_state=RANDOM_STATE, device=device,
        verbose=True)
    print("\nAutoencoder summary:")
    print(ae_summary)
    autoencoder, _ = freeze_ladis_autoencoder(model=autoencoder)
    raw_train_latent, _ = encode_ladis_partition(model=autoencoder, num_data=train_num, cat_data=train_cat, num_columns=num_columns,
        cat_columns=cat_columns, cat_cards=cat_cards, batch_size=64, device=device, data_name="training")
    raw_val_latent, _ = encode_ladis_partition(model=autoencoder, num_data=val_num, cat_data=val_cat, num_columns=num_columns,
        cat_columns=cat_columns, cat_cards=cat_cards, batch_size=64, device=device, data_name="validation")
    (standardized_train_latent, standardization_artifact, _, _) = fit_ladis_latent_standardization(train_latent_data=raw_train_latent)
    standardized_val_latent, _ = transform_ladis_latent_space(latent_data=raw_val_latent, standardization_artifact=standardization_artifact,
        split_name="validation")
    diffusion_conf = {"latent_dimension": 4, "hidden_dimensions": (64, 64), "timestep_embedding_dimension": 16, "dropout": 0.0,
        "num_timesteps": 50, "scheduler": "cosine", "prediction_target": "z0", "bound_predicted_z0": False, "learning_rate": 5e-4,
        "weight_decay": 1e-5, "batch_size": 64, "training_steps": 200, "validation_interval": 50, "validation_rows": 64,
        "minimum_improvement": 1e-6, "gradient_clip_norm": 1.0}
    diffusion_model, _, diffusion_summary = fit_ladis_clean_z0_diffusion(train_latent_data=standardized_train_latent,
        val_latent_data=standardized_val_latent, conf=diffusion_conf, random_state=RANDOM_STATE, device=device, verbose=True)
    print("\nDiffusion summary:")
    print(diffusion_summary)
    latent_columns = standardized_train_latent.columns.tolist()
    generated_standardized_latent, generation_summary = (sample_ladis_standardized_latents(diffusion_model=diffusion_model, num_rows=100,
        latent_columns=latent_columns, batch_size=100, random_state=RANDOM_STATE + 100, device=device))
    generated_raw_latent = inverse_ladis_latent_standardization(standardized_latent_data=generated_standardized_latent,
        standardization_artifact=standardization_artifact)
    synthetic_num, synthetic_cat, _ = decode_ladis_latent_partition(model=autoencoder, raw_latent_data=generated_raw_latent, num_columns=num_columns,
        cat_columns=cat_columns, cat_cards=cat_cards, batch_size=100, device=device, cat_sampling="sample", random_state=RANDOM_STATE + 200)
    synthetic = pd.concat([synthetic_num, synthetic_cat], axis=1)
    print("\nGeneration summary:")
    print(generation_summary)
    print("\nSynthetic sample:")
    print(synthetic.head())


if __name__ == "__main__":
    main()