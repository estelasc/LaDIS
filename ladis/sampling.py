"""
Synthetic sampling utilities for LaDIS.

This module performs ancestral reverse diffusion in the standardized latent
space and decodes generated latent representations into numerical values and
categorical variables using the trained mixed-type autoencoder.
"""

from __future__ import annotations
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .utils import _validate_ladis_latent_dataframe
from .models import MixedTypeAutoencoder, CleanLatentGaussianDiffusion


def sample_ladis_standardized_latents(*, diffusion_model: CleanLatentGaussianDiffusion, num_rows: int, latent_columns: Sequence[str],
    batch_size: int, random_state: int, device: torch.device) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate standardized clean latent samples by ancestral DDPM sampling.
    """

    num_rows = int(num_rows)
    batch_size = int(batch_size)
    latent_columns = list(latent_columns)
    if num_rows <= 0:
        raise ValueError("`num_rows` must be positive.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")
    if len(latent_columns) != diffusion_model.latent_dim:
        raise RuntimeError("The supplied latent-column count does not match the diffusion model latent dimension.")
    generator = torch.Generator(device=device)
    generator.manual_seed(random_state)
    gen_batches = []
    diffusion_model.eval()
    start_time = perf_counter()
    with torch.no_grad():
        for start_index in range(0, num_rows, batch_size):
            current_batch_size = min(batch_size, num_rows - start_index)
            zt = torch.randn((current_batch_size, diffusion_model.latent_dim), device=device, dtype=torch.float32, generator=generator)
            for timestep in reversed(range(diffusion_model.num_timesteps)):
                timestep_tensor = torch.full((current_batch_size,), timestep, device=device, dtype=torch.long)
                zt = diffusion_model.p_sample(zt=zt, timesteps=timestep_tensor, generator=generator)
                if not torch.isfinite(zt).all().item():
                    raise RuntimeError(f"Non-finite latent values were produced during reverse diffusion at timestep {timestep}.")
            gen_batches.append(zt.cpu().numpy().astype(np.float32, copy=False))
    sampling_s = float(perf_counter() - start_time)
    gen_vals = np.concatenate(gen_batches, axis=0)
    gen_data = pd.DataFrame(gen_vals, columns=latent_columns).astype(np.float32)
    gen_stds = np.std(gen_vals.astype(np.float64), axis=0, ddof=0)
    gen_summary = pd.Series({"generated_rows": len(gen_data), "latent_dimension": gen_data.shape[1], "num_timesteps": diffusion_model.num_timesteps,
        "sampling_method": "ancestral", "generated_max_abs_coordinate_mean": float(np.abs(np.mean(gen_vals, axis=0)).max()),
        "generated_minimum_coordinate_std": float(gen_stds.min()), "generated_median_coordinate_std": float(np.median(gen_stds)),
        "generated_maximum_coordinate_std": float(gen_stds.max()), "generated_absolute_q999": float(np.quantile(np.abs(gen_vals), 0.999)),
        "generated_maximum_absolute_value": float(np.abs(gen_vals).max()), "generated_values_are_finite": True, "sampling_seconds": sampling_s}, 
        name="value")
    return (gen_data, gen_summary)


def decode_ladis_latent_partition(*, model: MixedTypeAutoencoder, raw_latent_data: pd.DataFrame, num_columns: Sequence[str], cat_columns: Sequence[str],
    cat_cards: Mapping[str, int], batch_size: int, device: torch.device, cat_sampling: str, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, 
    pd.Series]:
    """
    Decode raw autoencoder latents into numerical values and categorical codes.
    """

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")
    latent_columns = [f"z_{dimension:02d}" for dimension in range(model.latent_dim)]
    latent_vals = _validate_ladis_latent_dataframe(latent_data=raw_latent_data, expected_columns=latent_columns, 
        data_name="latent decoder input").astype(np.float32)
    num_columns = list(num_columns)
    cat_columns = list(cat_columns)
    if cat_sampling not in {"argmax", "sample"}:
        raise ValueError("`cat_sampling` must be either 'argmax' or 'sample'.")
    generator = torch.Generator(device=device)
    generator.manual_seed(random_state)
    num_batches = []
    cat_batches = []
    model.eval()
    with torch.no_grad():
        for start_index in range(0, len(latent_vals), batch_size):
            end_index = min(start_index + batch_size, len(latent_vals))
            current_batch_size = end_index - start_index
            latent_batch = torch.from_numpy(latent_vals[start_index:end_index]).to(device=device, dtype=torch.float32)
            (num_rec, cat_logits) = model.decode(latent_batch)
            if not torch.isfinite(num_rec).all().item():
                raise RuntimeError("The frozen decoder produced non-finite numerical values.")
            cat_preds = []
            for column_index, logits in enumerate(cat_logits):
                expected_card = int(cat_cards[cat_columns[column_index]])
                if logits.shape[1] != expected_card:
                    raise RuntimeError("A categorical decoder head does not preserve its expected cardinality.")
                if cat_sampling == "argmax":
                    predictions = logits.argmax(dim=1)
                else:
                    probabilities = torch.softmax(logits, dim=1)
                    predictions = torch.multinomial(probabilities, num_samples=1, replacement=True, generator=generator).squeeze(1)
                cat_preds.append(predictions)
            if cat_preds:
                cat_batch_values = torch.stack(cat_preds, dim=1).cpu().numpy()
            else:
                cat_batch_values = np.empty((current_batch_size, 0), dtype=np.int64)
            num_batches.append(num_rec.cpu().numpy())
            cat_batches.append(cat_batch_values)
    num_data = pd.DataFrame(np.concatenate(num_batches, axis=0), columns=num_columns).astype(np.float32)
    cat_data = pd.DataFrame(np.concatenate(cat_batches, axis=0), columns=cat_columns).astype(np.int64)
    invalid_codes = 0
    for column in cat_columns:
        codes = cat_data[column].to_numpy(dtype=np.int64)
        cardinality = int(cat_cards[column])
        invalid_codes += int(((codes < 0) | (codes >= cardinality)).sum())
    if invalid_codes:
        raise RuntimeError("The latent decoder produced invalid categorical codes.")
    decoding_summary = pd.Series({"decoded_rows": len(num_data), "numerical_variables": len(num_columns), "categorical_variables": len(cat_columns),
        "categorical_sampling": cat_sampling, "numerical_values_are_finite": True, "invalid_categorical_codes": invalid_codes, 
        "decoding_valid": True}, name="value")
    return (num_data, cat_data, decoding_summary)