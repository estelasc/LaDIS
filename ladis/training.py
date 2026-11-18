"""
Training utilities for the LaDIS autoencoder and latent diffusion model.

This module implements autoencoder loss computation and optimization, clean-z0
latent diffusion training and validation, and the optional decoder-aware
diffusion objectives used for experimental configurations.
"""

from __future__ import annotations
from copy import deepcopy
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .utils import _validate_ladis_latent_dataframe, _validate_ladis_mixed_type_data
from .models import MixedTypeAutoencoder, CleanLatentGaussianDiffusion, CleanLatentDenoiser

# Autoencoder losses
def _compute_ladis_variance_preservation_loss(*, ref_num: torch.Tensor, rec_num: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Penalize feature-wise variance distortion in the numerical reconstruction.

    The loss compares the standard deviation of each reconstructed numerical
    variable with the standard deviation of the corresponding reference
    variable within the current batch. A squared log-ratio treats equivalent
    relative under- and over-dispersion symmetrically.
    """

    if eps <= 0.0:
        raise ValueError("`epsilon` must be positive.")
    if (ref_num.shape != rec_num.shape):
        raise RuntimeError("Reference and reconstructed numerical tensors must have identical shapes when calculating the variance-preservation loss.")
    ref_var = torch.var(ref_num.detach(), dim=0, unbiased=False)
    rec_var = torch.var(rec_num, dim=0, unbiased=False)
    ref_std = torch.sqrt(ref_var + eps)
    rec_std = torch.sqrt(rec_var + eps)
    log_std_ratio = (torch.log(rec_std) - torch.log(ref_std))
    var_loss = torch.mean(torch.square(log_std_ratio))
    return var_loss


def _compute_ladis_marginal_preservation_loss(*, ref_num: torch.Tensor, rec_num: torch.Tensor) -> torch.Tensor:
    """
    Penalize numerical marginal-distribution distortion.

    For each numerical variable, the reference and reconstructed values are
    sorted independently within the current batch. The mean absolute distance
    between corresponding empirical quantiles is then calculated. With equal
    sample sizes, this corresponds to an empirical one-dimensional
    Wasserstein-1 distance averaged across numerical variables.
    """

    if (ref_num.shape != rec_num.shape):
        raise RuntimeError("Reference and reconstructed numerical tensors must have identical shapes when calculating the marginal-preservation loss.")
    if ref_num.ndim != 2:
        raise RuntimeError("The marginal-preservation loss expects a two-dimensional [batch, features] numerical tensor.")
    if ref_num.shape[0] < 2:
        raise RuntimeError("At least two observations are required to calculate the marginal-preservation loss.")
    # The reference observations define the target empirical distribution and therefore do not require gradients
    sorted_ref = torch.sort(ref_num.detach(), dim=0).values
    # Gradients are preserved through the sorted reconstructed values
    sorted_rec = torch.sort(rec_num, dim=0).values
    marginal_loss = F.l1_loss(sorted_rec, sorted_ref, reduction="mean")
    return marginal_loss


def _compute_ladis_autoencoder_loss(*, model: MixedTypeAutoencoder, num_input: torch.Tensor, cat_input: torch.Tensor, num_loss_weight: float,
    cat_loss_weight: float, var_loss_weight: float, var_loss_eps: float, marginal_loss_weight: float, 
    norm_cat_loss: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the mixed reconstruction objective.

    The objective can combine pointwise numerical reconstruction,
    feature-wise variance preservation, empirical marginal-distribution
    preservation, and categorical reconstruction.
    """

    (_, num_rec, cat_logits) = model(num_input, cat_input)
    # Pointwise numerical reconstruction
    num_loss = F.mse_loss(num_rec, num_input, reduction="mean")
    # Numerical variance preservation
    var_loss = _compute_ladis_variance_preservation_loss(ref_num=num_input, rec_num=num_rec, eps=var_loss_eps)
    # Numerical marginal-distribution preservation
    marginal_loss = _compute_ladis_marginal_preservation_loss(ref_num=num_input, rec_num=num_rec)
    # Categorical reconstruction
    cat_losses = []
    for column_index, logits in enumerate(cat_logits):
        column_loss = F.cross_entropy(logits, cat_input[:, column_index], reduction="mean")
        if norm_cat_loss:
            cardinality = model.cat_cards[column_index]
            column_loss = column_loss / float(np.log(cardinality))
        cat_losses.append(column_loss)
    cat_loss = (torch.stack(cat_losses).mean())
    # Complete mixed-type reconstruction objective
    total_loss = num_loss_weight*num_loss + var_loss_weight*var_loss + marginal_loss_weight*marginal_loss + cat_loss_weight*cat_loss
    return (total_loss, num_loss, var_loss, marginal_loss, cat_loss)


def _run_ladis_autoencoder_epoch(*, model: MixedTypeAutoencoder, data_loader: DataLoader, device: torch.device, num_loss_weight: float,
    cat_loss_weight: float, var_loss_weight: float, var_loss_eps: float, marginal_loss_weight: float, norm_cat_loss: bool,
    optimizer: torch.optim.Optimizer | None, gradient_clip_norm: float | None) -> dict[str, float]:
    """
    Run one training or validation autoencoder epoch.
    """

    training = (optimizer is not None)
    if training:
        model.train()
    else:
        model.eval()
    total_rows = 0
    total_loss_sum = 0.0
    num_loss_sum = 0.0
    var_loss_sum = 0.0
    marginal_loss_sum = 0.0
    cat_loss_sum = 0.0
    gradient_context = (torch.enable_grad() if training else torch.no_grad())
    with gradient_context:
        for (num_batch, cat_batch) in data_loader:
            num_batch = num_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            cat_batch = cat_batch.to(device=device, dtype=torch.long, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            (total_loss, num_loss, var_loss, marginal_loss, cat_loss) = _compute_ladis_autoencoder_loss(model=model, num_input=num_batch, 
                cat_input=cat_batch, num_loss_weight=num_loss_weight, cat_loss_weight=cat_loss_weight, var_loss_weight=var_loss_weight,
                var_loss_eps=var_loss_eps, marginal_loss_weight=marginal_loss_weight, norm_cat_loss=norm_cat_loss)
            if not torch.isfinite(total_loss).item():
                raise RuntimeError("The mixed-type autoencoder produced a non-finite loss.")
            if training:
                total_loss.backward()
                if (gradient_clip_norm is not None):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()
            batch_rows = int(num_batch.shape[0])
            total_rows += batch_rows
            total_loss_sum += float(total_loss.detach().item())*batch_rows
            num_loss_sum += float(num_loss.detach().item())*batch_rows
            var_loss_sum += float(var_loss.detach().item())*batch_rows
            marginal_loss_sum += float(marginal_loss.detach().item())*batch_rows
            cat_loss_sum += float(cat_loss.detach().item())*batch_rows
    if total_rows == 0:
        raise RuntimeError("The autoencoder DataLoader produced no observations.")
    return {"total_loss": (total_loss_sum / total_rows), "numerical_loss": (num_loss_sum / total_rows), "variance_loss": (var_loss_sum / total_rows),
        "marginal_loss": (marginal_loss_sum / total_rows), "categorical_loss": (cat_loss_sum / total_rows)}


def fit_ladis_autoencoder(*, train_num_data: pd.DataFrame, train_cat_data: pd.DataFrame, val_num_data: pd.DataFrame, val_cat_data: pd.DataFrame,
    num_columns: Sequence[str], cat_columns: Sequence[str], cat_cards: Mapping[str, int], conf: Mapping[str, Any], random_state: int,
    device: torch.device, verbose: bool = True) -> tuple[MixedTypeAutoencoder, pd.DataFrame, pd.Series]:
    """
    Fit the mixed-type autoencoder and restore its best validation checkpoint.
    """

    num_columns = list(num_columns)
    cat_columns = list(cat_columns)
    cat_cards = {column: int(cat_cards[column]) for column in cat_columns}
    (train_num_vals, train_cat_vals) = _validate_ladis_mixed_type_data(num_data=train_num_data, cat_data=train_cat_data, num_columns=num_columns,
        cat_columns=cat_columns, cat_cards=cat_cards, data_name="training")
    (val_num_vals, val_cat_vals) = _validate_ladis_mixed_type_data(num_data=val_num_data, cat_data=val_cat_data, num_columns=num_columns,
        cat_columns=cat_columns, cat_cards=cat_cards, data_name="val")
    emb_dims = dict(conf["embedding_dimensions"])
    cat_emb_dims = [int(emb_dims[column]) for column in cat_columns]
    cat_cards = [int(cat_cards[column]) for column in cat_columns]
    latent_dim = int(conf["latent_dimension"])
    enc_hidden_dims = tuple(int(value) for value in conf["encoder_hidden_dimensions"])
    dec_hidden_dims = tuple(int(value) for value in conf["decoder_hidden_dimensions"])
    dropout = float(conf.get("dropout", 0.0))
    learning_rate = float(conf.get("learning_rate", 1e-3))
    weight_decay = float(conf.get("weight_decay", 1e-5))
    batch_size = int(conf.get("batch_size", 4096))
    max_epochs = int(conf.get("maximum_epochs", 100))
    early_stopping_patience = int(conf.get("early_stopping_patience", 10))
    min_improvement = float(conf.get("minimum_improvement", 1e-4))
    gradient_clip_norm = conf.get("gradient_clip_norm", 1.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    num_loss_weight = float(conf.get("numerical_loss_weight", 1.0))
    cat_loss_weight = float(conf.get("categorical_loss_weight", 1.0))
    var_loss_weight = float(conf.get("variance_loss_weight", 0.0))
    var_loss_eps = float(conf.get("variance_loss_epsilon", 1e-6))
    marginal_loss_weight = float(conf.get("marginal_loss_weight", 0.0))
    if var_loss_weight < 0.0:
        raise ValueError("`variance_loss_weight` cannot be negative.")
    if marginal_loss_weight < 0.0:
        raise ValueError("`marginal_loss_weight` cannot be negative.")
    if var_loss_eps <= 0.0:
        raise ValueError("`variance_loss_epsilon` must be positive.")
    norm_cat_loss = bool(conf.get("normalize_categorical_loss", True))
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_state)
    model = MixedTypeAutoencoder(num_features=len(num_columns),  cat_cards=cat_cards, cat_emb_dims=cat_emb_dims, latent_dim=latent_dim,
        enc_hidden_dims=enc_hidden_dims, dec_hidden_dims=dec_hidden_dims, dropout=dropout).to(device)
    training_dataset = TensorDataset(torch.from_numpy(train_num_vals), torch.from_numpy(train_cat_vals))
    val_dataset = TensorDataset(torch.from_numpy(val_num_vals), torch.from_numpy(val_cat_vals))
    loader_gen = torch.Generator()
    loader_gen.manual_seed(random_state)
    pin_memory = (device.type == "cuda")
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=pin_memory,
        generator=loader_gen)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=pin_memory)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val_loss = np.inf
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0
    history_records = []
    start_time = perf_counter()
    for epoch in range(1, max_epochs + 1):
        training_metrics = _run_ladis_autoencoder_epoch(model=model, data_loader=training_loader, device=device, num_loss_weight=num_loss_weight,
            cat_loss_weight=cat_loss_weight, var_loss_weight=var_loss_weight, var_loss_eps=var_loss_eps, marginal_loss_weight=marginal_loss_weight,
            norm_cat_loss=norm_cat_loss, optimizer=optimizer, gradient_clip_norm=gradient_clip_norm)
        val_metrics = _run_ladis_autoencoder_epoch(model=model, data_loader=val_loader, device=device, num_loss_weight=num_loss_weight,
            cat_loss_weight=cat_loss_weight, var_loss_weight=var_loss_weight, var_loss_eps=var_loss_eps, marginal_loss_weight=marginal_loss_weight,
            norm_cat_loss=norm_cat_loss, optimizer=None, gradient_clip_norm=None)
        val_total_loss = float(val_metrics["total_loss"])
        improved = bool(val_total_loss < (best_val_loss - min_improvement))
        if improved:
            best_val_loss = val_total_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        history_records.append({"epoch": epoch, "training_total_loss": training_metrics["total_loss"], 
            "training_numerical_loss": training_metrics["numerical_loss"], "training_variance_loss": training_metrics["variance_loss"],
            "training_marginal_loss": training_metrics["marginal_loss"], "training_categorical_loss": training_metrics["categorical_loss"],
            "validation_total_loss": val_metrics["total_loss"], "validation_numerical_loss": val_metrics[ "numerical_loss"],
            "validation_variance_loss": val_metrics["variance_loss"], "validation_marginal_loss": val_metrics["marginal_loss"],
            "validation_categorical_loss": val_metrics["categorical_loss"], "best_checkpoint": improved, 
            "epochs_without_improvement": epochs_without_improvement})
        if (verbose and (epoch == 1 or epoch % 5 == 0 or improved)):
            print(f"Epoch {epoch:03d} | train={training_metrics['total_loss']:.6f} | val={val_metrics['total_loss']:.6f} | "
                f"num={val_metrics['numerical_loss']:.6f} | var={val_metrics['variance_loss']:.6f} | marg={val_metrics['marginal_loss']:.6f} | "
                f"cat={val_metrics['categorical_loss']:.6f}")
        if (epochs_without_improvement >= early_stopping_patience):
            if verbose:
                print(f"Early stopping after {early_stopping_patience} epochs without sufficient validation improvement.")
            break
    training_s = float(perf_counter() - start_time)
    if best_state_dict is None:
        raise RuntimeError("The mixed-type autoencoder did not produce a valid best checkpoint.")
    model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()
    history = pd.DataFrame(history_records)
    loss_columns = ["training_total_loss", "training_numerical_loss", "training_variance_loss", "training_marginal_loss", "training_categorical_loss",
        "validation_total_loss", "validation_numerical_loss", "validation_variance_loss", "validation_marginal_loss",  "validation_categorical_loss"]
    if not np.isfinite(history[loss_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("The autoencoder training history contains non-finite loss values.")
    model.training_configuration = deepcopy(dict(conf))
    training_summary = pd.Series({"training_rows": len(train_num_data), "validation_rows": len(val_num_data), "numerical_variables": len(num_columns),
        "categorical_variables": len(cat_columns), "categorical_embedding_dimension": sum(cat_emb_dims), 
        "encoder_input_dimension": (len(num_columns) + sum(cat_emb_dims)), "latent_dimension": latent_dim, "variance_loss_weight": var_loss_weight,
        "variance_loss_epsilon": var_loss_eps, "marginal_loss_weight": marginal_loss_weight, "epochs_completed": int(history["epoch"].iloc[-1]),
        "best_epoch": int(best_epoch), "best_validation_loss": float(best_val_loss), "stopped_early": bool(history["epoch"].iloc[-1] < max_epochs),
        "training_seconds": training_s, "losses_are_finite": True}, name="value")
    return (model, history, training_summary)


# Standard clean-z0 latent diffusion training
def _evaluate_ladis_clean_z0_validation_loss(*, diffusion_model: CleanLatentGaussianDiffusion, val_z0: np.ndarray, val_timesteps: np.ndarray,
    val_noise: np.ndarray, batch_size: int, device: torch.device) -> float:
    """
    Evaluate direct-z0 MSE using fixed validation noise and timesteps.
    """

    if not (len(val_z0) == len(val_timesteps) == len(val_noise)):
        raise RuntimeError("The fixed diffusion validation arrays contain different numbers of observations.")
    diffusion_model.eval()
    squared_error_sum = 0.0
    element_count = 0
    with torch.no_grad():
        for start_index in range(0, len(val_z0), batch_size):
            end_index = min(start_index + batch_size, len(val_z0))
            z0_batch = torch.from_numpy(val_z0[start_index:end_index]).to(device=device, dtype=torch.float32)
            timestep_batch = torch.from_numpy(val_timesteps[start_index:end_index]).to(device=device, dtype=torch.long)
            noise_batch = torch.from_numpy(val_noise[start_index:end_index]).to(device=device, dtype=torch.float32)
            zt_batch = diffusion_model.q_sample(z0=z0_batch, timesteps=timestep_batch, noise=noise_batch)
            pred_z0 = diffusion_model.predict_z0(zt=zt_batch, timesteps=timestep_batch)
            squared_error_sum += float(torch.sum((pred_z0 - z0_batch) ** 2).item())
            element_count += int(z0_batch.numel())
    if element_count == 0:
        raise RuntimeError("The diffusion validation evaluation processed no elements.")
    return (squared_error_sum / element_count)


def fit_ladis_clean_z0_diffusion(*, train_latent_data: pd.DataFrame, val_latent_data: pd.DataFrame, conf: Mapping[str, Any], random_state: int,
    device: torch.device, verbose: bool = True) -> tuple[CleanLatentGaussianDiffusion, pd.DataFrame, pd.Series]:
    """
    Train Gaussian latent diffusion using direct clean-z0 prediction.
    """

    latent_columns = train_latent_data.columns.tolist()
    train_vals = _validate_ladis_latent_dataframe(latent_data=train_latent_data, expected_columns=latent_columns, 
        data_name="diffusion training latent data").astype(np.float32)
    val_vals = _validate_ladis_latent_dataframe(latent_data=val_latent_data, expected_columns=latent_columns, 
        data_name="diffusion validation latent data").astype(np.float32)
    latent_dim = int(conf["latent_dimension"])
    if latent_dim != len(latent_columns):
        raise RuntimeError("The diffusion configuration latent dimension differs from the prepared latent representation.")
    if str(conf.get("prediction_target", "")) != "z0":
        raise ValueError("The proposed reference latent diffusion must predict clean z0.")
    if bool(conf.get("bound_predicted_z0", False)):
        raise ValueError("Predicted z0 bounding must remain disabled in the initial latent diffusion experiment.")
    hidden_dims = tuple(int(value) for value in conf["hidden_dimensions"])
    timestep_emb_dim = int(conf["timestep_embedding_dimension"])
    dropout = float(conf.get("dropout", 0.0))
    num_timesteps = int(conf["num_timesteps"])
    scheduler = str(conf["scheduler"])
    learning_rate = float(conf.get("learning_rate", 5e-4))
    weight_decay = float(conf.get("weight_decay", 1e-5))
    batch_size = int(conf.get("batch_size", 4_096))
    training_steps = int(conf.get("training_steps", 10_000))
    val_interval = int(conf.get("validation_interval", 500))
    val_rows = int(conf.get("validation_rows", 10_000))
    min_improvement = float(conf.get("minimum_improvement", 1e-6))
    gradient_clip_norm = conf.get("gradient_clip_norm", 1.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")
    if training_steps <= 0:
        raise ValueError("`training_steps` must be positive.")
    if val_interval <= 0:
        raise ValueError("`val_interval` must be positive.")
    # Reproducibility
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_state)
    # Construct the direct-z0 denoiser and Gaussian diffusion process
    denoiser = CleanLatentDenoiser(latent_dim=latent_dim, hidden_dims=hidden_dims, timestep_emb_dim=timestep_emb_dim, dropout=dropout)
    diffusion_model = CleanLatentGaussianDiffusion(denoiser=denoiser, num_timesteps=num_timesteps, scheduler=scheduler).to(device)
    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Training loader
    training_dataset = TensorDataset(torch.from_numpy(train_vals))
    loader_gen = torch.Generator()
    loader_gen.manual_seed(random_state)
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
        pin_memory=(device.type == "cuda"), generator=loader_gen)
    training_iterator = iter(training_loader)
    # Fixed validation corruption
    effective_val_rows = min(val_rows, len(val_vals))
    val_rng = np.random.default_rng(random_state + 10_000)
    val_indices = val_rng.choice(len(val_vals), size=effective_val_rows, replace=False)
    fixed_val_z0 = val_vals[val_indices].astype(np.float32, copy=True)
    fixed_val_timesteps = val_rng.integers(low=0, high=num_timesteps, size=effective_val_rows, dtype=np.int64)
    fixed_val_noise = val_rng.standard_normal(size=(effective_val_rows, latent_dim)).astype(np.float32)
    # Step-based diffusion training
    history_records = []
    best_val_loss = np.inf
    best_step = 0
    best_state_dict = None
    running_training_loss = 0.0
    running_training_steps = 0
    start_time = perf_counter()
    for step in range(1, training_steps + 1):
        try:
            (z0_batch,) = next(training_iterator)
        except StopIteration:
            training_iterator = iter(training_loader)
            (z0_batch,) = next(training_iterator)
        z0_batch = z0_batch.to(device=device, dtype=torch.float32, non_blocking=True)
        batch_rows = int(z0_batch.shape[0])
        timesteps = torch.randint(low=0, high=num_timesteps, size=(batch_rows,), device=device, dtype=torch.long)
        gaussian_noise = torch.randn_like(z0_batch)
        zt_batch = diffusion_model.q_sample(z0=z0_batch, timesteps=timesteps, noise=gaussian_noise)
        diffusion_model.train()
        optimizer.zero_grad(set_to_none=True)
        pred_z0 = diffusion_model.predict_z0(zt=zt_batch, timesteps=timesteps)
        training_loss = F.mse_loss(pred_z0, z0_batch, reduction="mean")
        if not torch.isfinite(training_loss).item():
            raise RuntimeError(f"Non-finite direct-z0 diffusion loss at step {step}.")
        training_loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()
        running_training_loss += float(training_loss.detach().item())
        running_training_steps += 1
        evaluate_now = bool(step == 1 or step % val_interval == 0 or step == training_steps)
        if not evaluate_now:
            continue
        mean_training_loss = (running_training_loss / running_training_steps)
        val_loss = _evaluate_ladis_clean_z0_validation_loss(diffusion_model=diffusion_model, val_z0=fixed_val_z0, val_timesteps=fixed_val_timesteps,
            val_noise=fixed_val_noise, batch_size=batch_size, device=device)
        improved = bool(val_loss < (best_val_loss - min_improvement))
        if improved:
            best_val_loss = val_loss
            best_step = int(step)
            best_state_dict = {key: (value.detach().cpu().clone()) for key, value in diffusion_model.state_dict().items()}
        history_records.append({"step": step, "training_z0_mse": mean_training_loss, "validation_z0_mse": val_loss, "best_checkpoint": improved})
        if verbose:
            print(f"Step {step:05d} | train_z0_mse={mean_training_loss:.6f} | val_z0_mse={val_loss:.6f}")
        running_training_loss = 0.0
        running_training_steps = 0
    training_s = float(perf_counter() - start_time)
    if best_state_dict is None:
        raise RuntimeError("The clean-z0 latent diffusion model did not produce a valid checkpoint.")
    diffusion_model.load_state_dict(best_state_dict)
    diffusion_model.to(device)
    diffusion_model.eval()
    history = pd.DataFrame(history_records)
    if not np.isfinite(history[["training_z0_mse", "validation_z0_mse"]].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("The clean-z0 diffusion training history contains non-finite losses.")
    diffusion_model.training_configuration = deepcopy(dict(conf))
    total_params = int(sum(param.numel() for param in diffusion_model.parameters()))
    training_summary = pd.Series({"training_rows": len(train_latent_data), "validation_rows_used": effective_val_rows, "latent_dimension": latent_dim,
        "num_timesteps": num_timesteps, "scheduler": scheduler, "prediction_target": "z0", "predicted_z0_bounded": False,
        "training_steps": training_steps, "best_step": best_step, "best_validation_z0_mse": float(best_val_loss), "model_parameters": total_params,
        "training_seconds": training_s, "losses_are_finite": True, "best_checkpoint_restored": True}, name="value")
    return (diffusion_model, history, training_summary)


# Decoder-aware diffusion training
def _compute_ladis_decoder_aware_losses(*, diffusion_model: CleanLatentGaussianDiffusion, dec_model: MixedTypeAutoencoder, z0_target: torch.Tensor,
    zt: torch.Tensor, timesteps: torch.Tensor, num_target: torch.Tensor, cat_target: torch.Tensor, latent_training_mean: torch.Tensor,
    latent_training_std: torch.Tensor, cat_columns: Sequence[str], cat_cards: Mapping[str, int], dec_aware_loss_weight: float, 
    dec_num_loss_weight: float, dec_cat_loss_weight: float, norm_dec_cat_loss: bool, dec_target_mode: str, 
    dec_timestep_weighting: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the decoder-aware direct-z0 diffusion objective.

    Two decoder-target modes are supported:
    - ``original``:
      decoded predictions are compared with the original numerical and
      categorical encoder inputs. This reproduces the initial decoder-aware
      ablation.
    - ``decoder_consistency``:
      decoded predictions are compared with the frozen decoder outputs
      obtained from the true clean latent z0. This keeps the latent and
      decoded-space objectives mutually consistent.

    Decoder guidance can additionally be weighted by alpha_bar(t), so that
    sample-specific decoded-space supervision progressively vanishes as the
    noisy latent loses information about its originating observation.
    """

    if dec_target_mode not in {"original", "decoder_consistency"}:
        raise ValueError("`dec_target_mode` must be either 'original' or 'decoder_consistency'.")
    if dec_timestep_weighting not in {"none", "alpha_bar"}:
        raise ValueError("`dec_timestep_weighting` must be either 'none' or 'alpha_bar'.")
    # Predict the clean standardized latent representation
    pred_z0 = diffusion_model.predict_z0(zt=zt, timesteps=timesteps)
    # Primary latent denoising objective
    z0_loss = F.mse_loss(pred_z0, z0_target, reduction="mean")
    # Return predicted z0 differentiably to the raw AE latent space
    pred_raw_latent = (pred_z0*latent_training_std + latent_training_mean)
    # Decode predicted latent vectors
    (pred_num, pred_cat_logits) = dec_model.decode(pred_raw_latent)
    if not torch.isfinite(pred_num).all().item():
        raise RuntimeError("The frozen decoder produced non-finite numerical values during decoder-aware diffusion training.")
    # Construct decoded-space targets
    if dec_target_mode == "original":
        target_num = num_target
        target_cat_logits = None
    else:
        # The true clean latent z0 defines the canonical decoder target
        target_raw_latent = z0_target*latent_training_std + latent_training_mean
        # Target decoder outputs must not create a gradient path
        with torch.no_grad():
            (target_num, target_cat_logits) = dec_model.decode(target_raw_latent)
        target_num = target_num.detach()
    # Per-observation numerical decoder loss
    dec_num_loss_per_row = torch.mean((pred_num - target_num)**2, dim=1)
    # Per-observation categorical decoder loss
    cat_loss_per_column = []
    for column_index, pred_logits in enumerate(pred_cat_logits):
        column = cat_columns[column_index]
        cardinality = int(cat_cards[column])
        if pred_logits.shape[1] != cardinality:
            raise RuntimeError(f"The decoder head for '{column}' has cardinality {pred_logits.shape[1]}, but {cardinality} was expected.")
        if dec_target_mode == "original":
            column_loss = F.cross_entropy(pred_logits, cat_target[:, column_index], reduction="none")
        else:
            target_logits = target_cat_logits[column_index].detach()
            target_probs = torch.softmax(target_logits, dim=1)
            pred_log_probs = torch.log_softmax(pred_logits, dim=1)
            column_loss = F.kl_div(pred_log_probs, target_probs, reduction="none").sum(dim=1)
        if norm_dec_cat_loss:
            column_loss = column_loss / float(np.log(cardinality))
        cat_loss_per_column.append(column_loss)
    dec_cat_loss_per_row = torch.stack(cat_loss_per_column, dim=1).mean(dim=1)
    # Complete decoded-space consistency loss per observation
    dec_loss_per_row = dec_num_loss_weight*dec_num_loss_per_row + dec_cat_loss_weight*dec_cat_loss_per_row
    dec_num_loss = dec_num_loss_per_row.mean()
    dec_cat_loss = dec_cat_loss_per_row.mean()
    dec_loss = dec_loss_per_row.mean()
    # Noise-aware decoder weighting
    if dec_timestep_weighting == "none":
        timestep_weights = torch.ones(timesteps.shape[0], device=timesteps.device, dtype=pred_z0.dtype)
    else:
        timestep_weights = diffusion_model.alphas_cumprod.gather(0, timesteps).detach().to(dtype=pred_z0.dtype)
    weighted_dec_contribution = dec_aware_loss_weight*torch.mean(timestep_weights*dec_loss_per_row)
    # Complete diffusion objective
    total_loss = z0_loss + weighted_dec_contribution
    return (total_loss, z0_loss, dec_loss, dec_num_loss, dec_cat_loss, weighted_dec_contribution)


def _evaluate_ladis_decoder_aware_validation_losses(*, diffusion_model: CleanLatentGaussianDiffusion, dec_model: MixedTypeAutoencoder, 
    val_z0: np.ndarray, val_num: np.ndarray, val_cat: np.ndarray, val_timesteps: np.ndarray, val_noise: np.ndarray, 
    latent_training_mean: torch.Tensor, latent_training_std: torch.Tensor, cat_columns: Sequence[str], cat_cards: Mapping[str, int],
    dec_aware_loss_weight: float, dec_num_loss_weight: float, dec_cat_loss_weight: float, norm_dec_cat_loss: bool, dec_target_mode: str,
    dec_timestep_weighting: str, batch_size: int, device: torch.device) -> dict[str, float]:
    """
    Evaluate the decoder-aware diffusion objective using fixed validation
    observations, timesteps, and Gaussian corruption.
    """

    n_rows = len(val_z0)
    if not (n_rows == len(val_num) == len(val_cat) == len(val_timesteps) == len(val_noise)):
        raise RuntimeError("The fixed decoder-aware validation arrays do not preserve the same row count.")
    diffusion_model.eval()
    dec_model.eval()
    metric_sums = {"total_loss": 0.0, "z0_loss": 0.0, "decoder_loss": 0.0, "decoder_numerical_loss": 0.0, "decoder_categorical_loss": 0.0,
        "weighted_decoder_contribution": 0.0}
    total_rows = 0
    for start_index in range(0, n_rows, batch_size):
        end_index = min(start_index + batch_size, n_rows)
        z0_batch = torch.from_numpy(val_z0[start_index:end_index]).to(device=device, dtype=torch.float32)
        num_batch = torch.from_numpy(val_num[start_index:end_index]).to(device=device, dtype=torch.float32)
        cat_batch = torch.from_numpy(val_cat[start_index:end_index]).to(device=device, dtype=torch.long)
        timestep_batch = torch.from_numpy(val_timesteps[start_index:end_index]).to(device=device, dtype=torch.long)
        noise_batch = torch.from_numpy(val_noise[start_index:end_index]).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            zt_batch = diffusion_model.q_sample(z0=z0_batch, timesteps=timestep_batch, noise=noise_batch)
        # Gradient tracking is unnecessary for validation
        with torch.no_grad():
            (total_loss, z0_loss, dec_loss, dec_num_loss, dec_cat_loss, 
                weighted_dec_contribution) = _compute_ladis_decoder_aware_losses(diffusion_model=diffusion_model, dec_model=dec_model, 
                z0_target=z0_batch, zt=zt_batch, timesteps=timestep_batch, num_target=num_batch, cat_target=cat_batch, 
                latent_training_mean=latent_training_mean, latent_training_std=latent_training_std, cat_columns=cat_columns, cat_cards=cat_cards,
                dec_aware_loss_weight=dec_aware_loss_weight, dec_num_loss_weight=dec_num_loss_weight, dec_cat_loss_weight=dec_cat_loss_weight,
                norm_dec_cat_loss=norm_dec_cat_loss, dec_target_mode=dec_target_mode, dec_timestep_weighting=dec_timestep_weighting)
        batch_rows = int(z0_batch.shape[0])
        current_metrics = {"total_loss": total_loss, "z0_loss": z0_loss, "decoder_loss": dec_loss, "decoder_numerical_loss": dec_num_loss,
            "decoder_categorical_loss": dec_cat_loss, "weighted_decoder_contribution": weighted_dec_contribution}
        for metric_name, metric_val in (current_metrics.items()):
            metric_sums[metric_name] += (
                float(metric_val.item())*batch_rows)
        total_rows += batch_rows
    if total_rows == 0:
        raise RuntimeError("Decoder-aware validation processed no observations.")
    return {metric_name: (metric_sum / total_rows) for metric_name, metric_sum in metric_sums.items()}


def fit_ladis_decoder_aware_z0_diffusion(*, train_latent_data: pd.DataFrame, val_latent_data: pd.DataFrame, train_num_data: pd.DataFrame,
    train_cat_data: pd.DataFrame, val_num_data: pd.DataFrame, val_cat_data: pd.DataFrame, dec_model: MixedTypeAutoencoder, 
    latent_standardization_artifact: Mapping[str, Any], num_columns: Sequence[str], cat_columns: Sequence[str], cat_cards: Mapping[str, int],
    conf: Mapping[str, Any], random_state: int, device: torch.device, verbose: bool = True) -> tuple[CleanLatentGaussianDiffusion, pd.DataFrame,
    pd.Series]:
    """
    Train direct-z0 latent diffusion with a frozen decoder-aware objective.
    """

    latent_columns = (train_latent_data.columns.tolist())
    train_z0 = _validate_ladis_latent_dataframe(latent_data=train_latent_data, expected_columns=latent_columns,
        data_name="decoder-aware diffusion training latent data").astype(np.float32)
    val_z0 = _validate_ladis_latent_dataframe(latent_data=val_latent_data, expected_columns=latent_columns,
        data_name="decoder-aware diffusion validation latent data").astype(np.float32)
    (train_num, train_cat) = _validate_ladis_mixed_type_data(num_data=train_num_data, cat_data=train_cat_data, num_columns=num_columns,
        cat_columns=cat_columns, cat_cards=cat_cards, data_name="decoder-aware diffusion training data")
    (val_num, val_cat) = _validate_ladis_mixed_type_data(num_data=val_num_data, cat_data=val_cat_data, num_columns=num_columns,
        cat_columns=cat_columns, cat_cards=cat_cards, data_name="decoder-aware diffusion validation data")
    if not (len(train_z0) == len(train_num) == len(train_cat)):
        raise RuntimeError("Training z0, numerical data, and categorical data are not row-aligned.")
    if not (len(val_z0) == len(val_num) == len(val_cat)):
        raise RuntimeError("Validation z0, numerical data, and categorical data are not row-aligned.")
    latent_dim = int(conf["latent_dimension"])
    if latent_dim != len(latent_columns):
        raise RuntimeError("The decoder-aware diffusion configuration does not preserve the prepared latent dimension.")
    if str(conf.get("prediction_target", "")) != "z0":
        raise ValueError("Decoder-aware diffusion must use direct clean-z0 prediction.")
    if bool(conf.get("bound_predicted_z0", False)):
        raise ValueError("Predicted-z0 bounding must remain disabled.")
    dec_aware_loss_weight = float(conf.get("decoder_aware_loss_weight", 0.0))
    dec_num_loss_weight = float(conf.get("decoder_numerical_loss_weight", 1.0))
    dec_cat_loss_weight = float(conf.get("decoder_categorical_loss_weight", 1.0))
    norm_dec_cat_loss = bool(conf.get("normalize_decoder_categorical_loss", True))
    dec_target_mode = str(conf.get("decoder_target_mode", "original"))
    dec_timestep_weighting = str(conf.get("decoder_timestep_weighting", "none"))
    if dec_target_mode not in {"original", "decoder_consistency"}:
        raise ValueError("`dec_target_mode` must be either 'original' or 'decoder_consistency'.")
    if dec_timestep_weighting not in {"none", "alpha_bar"}:
        raise ValueError("`dec_timestep_weighting` must be either 'none' or 'alpha_bar'.")
    if dec_aware_loss_weight <= 0.0:
        raise ValueError("`dec_aware_loss_weight` must be positive for the decoder-aware experiment.")
    if dec_num_loss_weight < 0.0:
        raise ValueError("`dec_num_loss_weight` cannot be negative.")
    if dec_cat_loss_weight < 0.0:
        raise ValueError("`dec_cat_loss_weight` cannot be negative.")
    # Verify that the pretrained representation model is genuinely frozen
    trainable_dec_params = int(sum(param.numel() for param in dec_model.parameters() if param.requires_grad))
    if trainable_dec_params != 0:
        raise RuntimeError("The pretrained autoencoder must be frozen before decoder-aware diffusion training.")
    dec_model.to(device)
    dec_model.eval()
    # Recover the training-fitted latent affine transformation
    artifact_columns = list(latent_standardization_artifact["latent_columns"])
    if artifact_columns != latent_columns:
        raise RuntimeError("The latent-standardization artifact does not preserve the diffusion latent schema.")
    latent_training_mean = torch.from_numpy(np.asarray(latent_standardization_artifact["training_mean"], 
        dtype=np.float32)).to(device=device).reshape(1, latent_dim)
    latent_training_std = torch.from_numpy(np.asarray(latent_standardization_artifact["training_std"], 
        dtype=np.float32)).to(device=device).reshape(1, latent_dim)
    if (not torch.isfinite(latent_training_mean).all().item() or not torch.isfinite(latent_training_std).all().item() 
        or torch.any(latent_training_std <= 0.0).item()):
        raise RuntimeError("The latent-standardization parameters are invalid.")
    # Read the unchanged Section 5.5 diffusion architecture
    hidden_dims = tuple(int(value) for value in conf["hidden_dimensions"])
    timestep_emb_dim = int(conf["timestep_embedding_dimension"])
    dropout = float(conf.get("dropout", 0.0))
    num_timesteps = int(conf["num_timesteps"])
    scheduler = str(conf["scheduler"])
    learning_rate = float(conf.get("learning_rate", 5e-4))
    weight_decay = float(conf.get("weight_decay", 1e-5))
    batch_size = int(conf.get("batch_size", 4_096))
    training_steps = int(conf.get("training_steps", 10_000))
    val_interval = int(conf.get("validation_interval", 500))
    val_rows = int(conf.get("validation_rows", 10_000))
    min_improvement = float(conf.get("minimum_improvement", 1e-6))
    gradient_clip_norm = conf.get("gradient_clip_norm", 1.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    # Reproducibility
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_state)
    # Construct a fresh denoiser
    denoiser = CleanLatentDenoiser(latent_dim=latent_dim, hidden_dims=hidden_dims, timestep_emb_dim=timestep_emb_dim, dropout=dropout)
    diffusion_model = CleanLatentGaussianDiffusion(denoiser=denoiser, num_timesteps=num_timesteps, scheduler=scheduler).to(device)
    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Row-aligned diffusion and decoded-space training targets
    training_dataset = TensorDataset(torch.from_numpy(train_z0), torch.from_numpy(train_num), torch.from_numpy(train_cat))
    loader_gen = torch.Generator()
    loader_gen.manual_seed(random_state)
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
        pin_memory=(device.type == "cuda"), generator=loader_gen)
    training_iterator = iter(training_loader)
    # Fixed and row-aligned validation corruption
    effective_val_rows = min(val_rows, len(val_z0))
    val_rng = np.random.default_rng(random_state + 10_000)
    val_indices = val_rng.choice(len(val_z0), size=effective_val_rows, replace=False)
    fixed_val_z0 = val_z0[val_indices].astype(np.float32, copy=True)
    fixed_val_num = val_num[val_indices].astype(np.float32, copy=True)
    fixed_val_cat = val_cat[val_indices].astype(np.int64, copy=True)
    fixed_val_timesteps = val_rng.integers(low=0, high=num_timesteps, size=effective_val_rows, dtype=np.int64)
    fixed_val_noise = val_rng.standard_normal(size=(effective_val_rows, latent_dim)).astype(np.float32)
    # Training
    history_records = []
    running_metric_sums = {"total_loss": 0.0, "z0_loss": 0.0, "decoder_loss": 0.0, "decoder_numerical_loss": 0.0,
        "decoder_categorical_loss": 0.0, "weighted_decoder_contribution": 0.0}
    running_steps = 0
    best_val_total_loss = np.inf
    best_step = 0
    best_state_dict = None
    best_val_metrics = None
    start_time = perf_counter()
    for step in range(1, training_steps + 1):
        try:
            (z0_batch, num_batch, cat_batch) = next(training_iterator)
        except StopIteration:
            training_iterator = iter(training_loader)
            (z0_batch, num_batch, cat_batch) = next(training_iterator)
        z0_batch = z0_batch.to(device=device, dtype=torch.float32, non_blocking=True)
        num_batch = num_batch.to(device=device, dtype=torch.float32, non_blocking=True)
        cat_batch = cat_batch.to(device=device, dtype=torch.long, non_blocking=True)
        batch_rows = int(z0_batch.shape[0])
        timesteps = torch.randint(low=0, high=num_timesteps, size=(batch_rows,), device=device, dtype=torch.long)
        gaussian_noise = torch.randn_like(z0_batch)
        zt_batch = diffusion_model.q_sample(z0=z0_batch, timesteps=timesteps, noise=gaussian_noise)
        diffusion_model.train()
        dec_model.eval()
        optimizer.zero_grad(set_to_none=True)
        (total_loss, z0_loss, dec_loss, dec_num_loss, dec_cat_loss, 
            weighted_dec_contribution) = _compute_ladis_decoder_aware_losses(diffusion_model=diffusion_model, dec_model=dec_model, z0_target=z0_batch,
            zt=zt_batch, timesteps=timesteps, num_target=num_batch, cat_target=cat_batch, latent_training_mean=latent_training_mean,
            latent_training_std=latent_training_std, cat_columns=cat_columns, cat_cards=cat_cards, dec_aware_loss_weight=dec_aware_loss_weight,
            dec_num_loss_weight=dec_num_loss_weight, dec_cat_loss_weight=dec_cat_loss_weight, norm_dec_cat_loss=norm_dec_cat_loss,
            dec_target_mode=dec_target_mode, dec_timestep_weighting=dec_timestep_weighting)
        if not torch.isfinite(total_loss).item():
            raise RuntimeError(f"Non-finite decoder-aware diffusion loss at step {step}.")
        total_loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()
        current_training_metrics = {"total_loss": total_loss, "z0_loss": z0_loss, "decoder_loss": dec_loss, "decoder_numerical_loss": dec_num_loss,
            "decoder_categorical_loss": dec_cat_loss, "weighted_decoder_contribution": weighted_dec_contribution}
        for metric_name, metric_val in (current_training_metrics.items()):
            running_metric_sums[metric_name] += float(metric_val.detach().item())
        running_steps += 1
        eval_now = bool(step == 1 or step % val_interval == 0 or step == training_steps)
        if not eval_now:
            continue
        mean_training_metrics = {metric_name: (metric_sum / running_steps) for metric_name, metric_sum in running_metric_sums.items()}
        val_metrics = _evaluate_ladis_decoder_aware_validation_losses(diffusion_model=diffusion_model, dec_model=dec_model, val_z0=fixed_val_z0,
            val_num=fixed_val_num, val_cat=fixed_val_cat, val_timesteps=fixed_val_timesteps, val_noise=fixed_val_noise, 
            latent_training_mean=latent_training_mean, latent_training_std=latent_training_std, cat_columns=cat_columns, cat_cards=cat_cards,
            dec_aware_loss_weight=dec_aware_loss_weight, dec_num_loss_weight=dec_num_loss_weight, dec_cat_loss_weight=dec_cat_loss_weight,
            norm_dec_cat_loss=norm_dec_cat_loss, dec_target_mode=dec_target_mode, dec_timestep_weighting=dec_timestep_weighting,
            batch_size=batch_size, device=device)
        improved = bool(val_metrics["total_loss"] < (best_val_total_loss - min_improvement))
        if improved:
            best_val_total_loss = float(val_metrics["total_loss"])
            best_step = int(step)
            best_val_metrics = dict(val_metrics)
            best_state_dict = {key: (value.detach().cpu().clone()) for key, value in diffusion_model.state_dict().items()}
        history_records.append({"step": step, "training_total_loss": mean_training_metrics["total_loss"], 
            "training_z0_mse": mean_training_metrics["z0_loss"], "training_decoder_loss": mean_training_metrics["decoder_loss"],
            "training_decoder_numerical_loss": mean_training_metrics["decoder_numerical_loss"],
            "training_decoder_categorical_loss": mean_training_metrics["decoder_categorical_loss"],
            "training_weighted_decoder_contribution": mean_training_metrics["weighted_decoder_contribution"],
            "validation_total_loss": val_metrics["total_loss"], "validation_z0_mse": val_metrics["z0_loss"], 
            "validation_decoder_loss": val_metrics["decoder_loss"], "validation_decoder_numerical_loss": val_metrics["decoder_numerical_loss"],
            "validation_decoder_categorical_loss": val_metrics["decoder_categorical_loss"],
            "validation_weighted_decoder_contribution": val_metrics["weighted_decoder_contribution"], "best_checkpoint": improved})
        if verbose:
            print(f"Step {step:05d} | train={mean_training_metrics['total_loss']:.6f} | val={val_metrics['total_loss']:.6f} | "
                f"z0={val_metrics['z0_loss']:.6f} | dec_num={val_metrics['decoder_numerical_loss']:.6f} | "
                f"dec_cat={val_metrics['decoder_categorical_loss']:.6f} | weighted_dec={val_metrics['weighted_decoder_contribution']:.6f}")
        running_metric_sums = {metric_name: 0.0 for metric_name in running_metric_sums}
        running_steps = 0
    training_s = float(perf_counter() - start_time)
    if (best_state_dict is None or best_val_metrics is None):
        raise RuntimeError("The decoder-aware diffusion model did not produce a valid checkpoint.")
    diffusion_model.load_state_dict(best_state_dict)
    diffusion_model.to(device)
    diffusion_model.eval()
    history = pd.DataFrame(history_records)
    loss_columns = [column for column in history.columns if ("loss" in column or "mse" in column or "contribution" in column)]
    if not np.isfinite(history[loss_columns].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("The decoder-aware diffusion history contains non-finite losses.")
    diffusion_model.training_configuration = deepcopy(dict(conf))
    total_params = int(sum(param.numel() for param in diffusion_model.parameters()))
    training_summary = pd.Series({"training_rows": len(train_latent_data), "validation_rows_used": effective_val_rows, "latent_dimension": latent_dim,
        "num_timesteps": num_timesteps, "scheduler": scheduler, "prediction_target": "z0", "predicted_z0_bounded": False,
        "decoder_aware_loss_weight": dec_aware_loss_weight, "decoder_numerical_loss_weight": dec_num_loss_weight,
        "decoder_categorical_loss_weight": dec_cat_loss_weight, "decoder_target_mode": dec_target_mode, 
        "decoder_timestep_weighting": dec_timestep_weighting, "decoder_frozen": True, "training_steps": training_steps, "best_step": best_step,
        "best_validation_total_loss": best_val_total_loss, "best_validation_z0_mse": float(best_val_metrics["z0_loss"]),
        "best_validation_decoder_loss": float(best_val_metrics["decoder_loss"]),
        "best_validation_decoder_numerical_loss": float(best_val_metrics["decoder_numerical_loss"]),
        "best_validation_decoder_categorical_loss": float(best_val_metrics["decoder_categorical_loss"]),
        "best_validation_weighted_decoder_contribution": float(best_val_metrics["weighted_decoder_contribution"]), "model_parameters": total_params,
        "training_seconds": training_s, "losses_are_finite": True, "best_checkpoint_restored": True}, name="value")
    return (diffusion_model, history, training_summary)