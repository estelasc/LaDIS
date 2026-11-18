"""
Neural network architectures used by LaDIS.

This module defines the mixed-type autoencoder, categorical embedding
configuration, timestep embeddings, clean-latent denoiser, and Gaussian
diffusion process operating in the standardized latent space.
"""

from __future__ import annotations
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn


def derive_ladis_embedding_dimensions(*, cat_columns: Sequence[str], cat_cards: Mapping[str, int], min_dim: int = 2, max_dim: int = 16) -> dict[str, int]:
    """
    Derive compact embedding dimensions from categorical cardinalities.

    A square-root rule is used so that larger categorical variables receive
    more representational capacity without allowing their embeddings to
    dominate the numerical input block.
    """

    cat_columns = list(cat_columns)
    if min_dim <= 0:
        raise ValueError("`min_dimension` must be positive.")
    if max_dim < min_dim:
        raise ValueError("`max_dimension` must be greater than or equal to `min_dimension`.")
    missing_cards = sorted(set(cat_columns).difference(cat_cards))
    if missing_cards:
        raise RuntimeError(f"No categorical cardinality was provided for: {missing_cards}")
    emb_dims = {}
    for column in cat_columns:
        cardinality = int(cat_cards[column])
        if cardinality < 2:
            raise ValueError(f"Categorical variable '{column}' has invalid cardinality {cardinality}.")
        emb_dim = int(np.ceil(np.sqrt(cardinality)))
        emb_dim = max(min_dim, min(max_dim, emb_dim))
        emb_dims[column] = emb_dim
    return emb_dims


def _build_ladis_mlp(*, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> tuple[nn.Sequential, int]:
    """
    Build one LayerNorm-SiLU multilayer-perceptron body.
    """

    hidden_dims = tuple(int(width) for width in hidden_dims)
    if input_dim <= 0:
        raise ValueError("`input_dim` must be positive.")
    if any(width <= 0 for width in hidden_dims):
        raise ValueError("Every hidden-layer dimension must be positive.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("`dropout` must lie in [0, 1).")
    layers = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    return (nn.Sequential(*layers), prev_dim)


class MixedTypeAutoencoder(nn.Module):
    """
    Deterministic autoencoder for continuous and categorical tabular data.

    Numerical variables enter the encoder directly. Each categorical
    variable is first represented by an independent trainable embedding.

    The decoder contains one numerical reconstruction head and one
    categorical classification head per categorical variable.
    """

    def __init__(self, *, num_features: int, cat_cards: Sequence[int], cat_emb_dims: Sequence[int], latent_dim: int, 
        enc_hidden_dims: Sequence[int], dec_hidden_dims: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.cat_cards = tuple(int(value) for value in cat_cards)
        self.cat_emb_dims = tuple(int(value) for value in cat_emb_dims)
        self.latent_dim = int(latent_dim)
        if self.num_features <= 0:
            raise ValueError("`num_features` must be positive.")
        if self.latent_dim <= 0:
            raise ValueError("`latent_dim` must be positive.")
        if (len(self.cat_cards) != len(self.cat_emb_dims)):
            raise ValueError("Categorical cardinalities and embedding dimensions must have the same length.")
        if any(cardinality < 2 for cardinality in self.cat_cards):
            raise ValueError("Every categorical variable must have at least two states.")
        self.cat_embs = (nn.ModuleList([nn.Embedding(cardinality, emb_dim) for (cardinality, emb_dim)
                    in zip(self.cat_cards, self.cat_emb_dims)]))
        enc_input_dim = self.num_features + sum(self.cat_emb_dims)
        (self.enc_body, enc_output_dim) = _build_ladis_mlp(input_dim=enc_input_dim, hidden_dims=enc_hidden_dims, dropout=dropout)
        self.to_latent = nn.Linear(enc_output_dim, self.latent_dim)
        (self.dec_body, dec_output_dim) = _build_ladis_mlp(input_dim=self.latent_dim, hidden_dims=dec_hidden_dims, dropout=dropout)
        self.num_head = nn.Linear(dec_output_dim, self.num_features)
        self.cat_heads = (nn.ModuleList([nn.Linear(dec_output_dim, cardinality) for cardinality in self.cat_cards]))

    def encode(self, num_input: torch.Tensor, cat_input: torch.Tensor) -> torch.Tensor:
        """
        Map one mixed-type batch into the continuous latent space.
        """

        emb_cats = [embedding(cat_input[:, column_index]) for column_index, embedding in enumerate(self.cat_embs)]
        enc_input = torch.cat([num_input, *emb_cats], dim=1)
        hidden_rep = (self.enc_body(enc_input))
        latent_rep = (self.to_latent(hidden_rep))
        return latent_rep

    def decode(self, latent_input: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Decode latent vectors into numerical values and categorical logits.
        """

        hidden_rep = self.dec_body(latent_input)
        num_rec = self.num_head(hidden_rep)
        cat_logits = [head(hidden_rep) for head in self.cat_heads]
        return (num_rec, cat_logits)

    def forward(self, num_input: torch.Tensor, cat_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Encode and reconstruct one mixed-type batch.
        """

        latent_rep = self.encode(num_input, cat_input)
        (num_rec, cat_logits) = self.decode(latent_rep)
        return (latent_rep, num_rec, cat_logits)


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Deterministic sinusoidal representation of discrete diffusion timesteps.
    """

    def __init__(self, *, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_period = float(max_period)
        if (self.dim < 4 or self.dim % 2 != 0):
            raise ValueError("`dim` must be an even integer greater than or equal to four.")
        if self.max_period <= 1.0:
            raise ValueError("`max_period` must be greater than one.")

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Convert integer timesteps into sinusoidal embeddings.
        """

        if timesteps.ndim != 1:
            raise RuntimeError("Diffusion timesteps must be a one-dimensional tensor.")
        half_dim = self.dim // 2
        freqs = torch.exp(-float(np.log(self.max_period))*torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / float(half_dim - 1))
        arguments = timesteps.to(dtype=torch.float32)[:, None]*freqs[None, :]
        return torch.cat([torch.sin(arguments), torch.cos(arguments),], dim=1)


class CleanLatentDenoiser(nn.Module):
    """
    Timestep-conditioned MLP that predicts the clean latent vector z0.
    """

    def __init__(self, *, latent_dim: int, hidden_dims: Sequence[int], timestep_emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.timestep_emb_dim = int(timestep_emb_dim)
        if self.latent_dim <= 0:
            raise ValueError("`latent_dimension` must be positive.")
        self.timestep_emb = nn.Sequential(SinusoidalTimestepEmbedding(dim=(self.timestep_emb_dim)), 
            nn.Linear(self.timestep_emb_dim, self.timestep_emb_dim), nn.SiLU(), nn.Linear(self.timestep_emb_dim, self.timestep_emb_dim))
        (self.denoising_body, final_hidden_dim) = _build_ladis_mlp(input_dim=(self.latent_dim + self.timestep_emb_dim), hidden_dims=hidden_dims,
            dropout=dropout)
        self.output_head = nn.Linear(final_hidden_dim, self.latent_dim)

    def forward(self, noisy_latent: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Predict the clean standardized latent vector from z_t.
        """

        if noisy_latent.ndim != 2:
            raise RuntimeError("`noisy_latent` must have shape [batch, latent_dimension].")
        if (noisy_latent.shape[1] != self.latent_dim):
            raise RuntimeError("The noisy latent tensor does not preserve the expected latent dimension.")
        if (timesteps.ndim != 1 or timesteps.shape[0] != noisy_latent.shape[0]):
            raise RuntimeError("The timestep tensor does not align with the latent batch.")
        time_rep = self.timestep_emb(timesteps)
        denoising_input = torch.cat([noisy_latent, time_rep,], dim=1)
        hidden_rep = (self.denoising_body(denoising_input))
        pred_z0 = self.output_head(hidden_rep)
        return pred_z0


class CleanLatentGaussianDiffusion(nn.Module):
    """
    Gaussian DDPM operating exclusively in the standardized latent space.

    The denoiser directly predicts the clean latent representation z0.
    """

    def __init__(self, *, denoiser: CleanLatentDenoiser, num_timesteps: int, scheduler: str = "cosine") -> None:
        super().__init__()
        self.denoiser = denoiser
        self.latent_dim = int(denoiser.latent_dim)
        self.num_timesteps = int(num_timesteps)
        self.scheduler = str(scheduler)
        if self.num_timesteps <= 1:
            raise ValueError("`num_timesteps` must be greater than one.")
        if self.scheduler != "cosine":
            raise ValueError("The current proposed LDM implementation supports only the cosine diffusion schedule.")
        # Cosine cumulative-alpha schedule
        cosine_offset = 0.008
        diffusion_steps = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((diffusion_steps / self.num_timesteps + cosine_offset) / (1.0 + cosine_offset))*torch.pi / 2.0) ** 2
        alpha_bar = (alpha_bar / alpha_bar[0])
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = torch.clamp(betas, min=1e-8, max=0.999)
        alphas = (1.0 - betas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1],], dim=0)
        # Forward-process coefficients
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        # Exact DDPM posterior q(z_{t-1} | z_t, z0)
        posterior_var = (betas*(1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        posterior_mean_coef1 = (betas*torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev)*torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        # Store all schedule quantities as float32 model buffers
        self.register_buffer("betas", betas.to(dtype=torch.float32))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(dtype=torch.float32))
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod.to(dtype=torch.float32))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod.to(dtype=torch.float32))
        self.register_buffer("posterior_variance", posterior_var.to(dtype=torch.float32))
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1.to(dtype=torch.float32))
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2.to(dtype=torch.float32))

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """
        Extract one schedule coefficient per observation.
        """

        extracted = values.gather(0, timesteps)
        return extracted.reshape(timesteps.shape[0], *([1]*(len(target_shape) - 1)))

    def q_sample(self, *, z0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Sample z_t from the forward Gaussian process q(z_t | z0).
        """

        if (z0.shape != noise.shape):
            raise RuntimeError("`z0` and `noise` must have identical shapes.")
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, z0.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, z0.shape)
        return (sqrt_alpha_bar*z0 + sqrt_one_minus_alpha_bar*noise)

    def predict_z0(self, *, zt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Predict clean z0 directly from a noisy latent representation.
        """

        return self.denoiser(zt, timesteps)

    def p_mean_variance(self, *, zt: torch.Tensor, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct p(z_{t-1} | z_t) using the predicted clean z0.
        """

        pred_z0 = self.predict_z0(zt=zt, timesteps=timesteps)
        posterior_mean = (self._extract(self.posterior_mean_coef1, timesteps, zt.shape)*pred_z0 + 
            self._extract(self.posterior_mean_coef2, timesteps, zt.shape)*zt)
        posterior_var = self._extract(self.posterior_variance, timesteps, zt.shape)
        return (posterior_mean, posterior_var, pred_z0)

    def p_sample(self, *, zt: torch.Tensor, timesteps: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """
        Perform one ancestral reverse-diffusion step.
        """

        (posterior_mean, posterior_var, _) = self.p_mean_variance(zt=zt, timesteps=timesteps)
        noise = torch.randn(zt.shape, device=zt.device, dtype=zt.dtype, generator=generator)
        nonzero_mask = (timesteps != 0).to(dtype=zt.dtype).reshape(timesteps.shape[0], *([1]* (zt.ndim - 1)))
        return (posterior_mean + nonzero_mask*torch.sqrt(torch.clamp(posterior_var, min=0.0))*noise)