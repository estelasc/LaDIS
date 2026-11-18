"""
Selection tests for the LaDIS package.

These tests validate the deterministic model-selection logic used by LaDIS
without running expensive autoencoder or diffusion searches.

They cover:
    1. latent-dimension mean-rank selection,
    2. deterministic Quality Score tie-breaking,
    3. construction of the three controlled clean-z0 denoiser variants,
    4. generated-latent diagnostics,
    5. the complete clean-z0 selection hierarchy using monkeypatched candidate
       training/evaluation functions.

The purpose is to verify the scientific selection protocol independently from
the cost and stochasticity of model training.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import ladis.selection as selection_module
from ladis.selection import (_summarize_ladis_generated_latent, build_ladis_clean_z0_candidate_configurations, rank_ladis_latent_dimensions,
    run_ladis_clean_z0_selection)

REFERENCE_NAME = "reference_clean_z0_ladis"


def _reference_diffusion_configuration() -> dict:
    """Return the reference clean-z0 LaDIS denoiser configuration."""

    return {"name": REFERENCE_NAME, "latent_dimension": 4, "hidden_dimensions": (256, 512, 512, 256), "timestep_embedding_dimension": 128,
        "dropout": 0.0, "num_timesteps": 500, "scheduler": "cosine", "prediction_target": "z0", "bound_predicted_z0": False, "learning_rate": 5e-4,
        "weight_decay": 1e-5, "batch_size": 4096, "training_steps": 10_000, "validation_interval": 500, "validation_rows": 10_000,
        "minimum_improvement": 1e-6, "gradient_clip_norm": 1.0}


def test_rank_ladis_latent_dimensions_uses_mean_rank() -> None:
    """
    The selected dimension must minimize the mean of the three generative ranks.

    The DataFrame index is deliberately the latent dimension because the
    selection routine returns the winning index as the selected dimension.
    """

    study = pd.DataFrame({"synthetic_quality": [0.80, 0.95, 0.85, 0.70], "synthetic_shapes": [0.90, 0.80, 0.85, 0.70],
        "synthetic_pair_trends": [0.85, 0.75, 0.95, 0.65]}, index=pd.Index([4, 8, 12, 16], name="latent_dimension"))
    ranking, selected_dimension = rank_ladis_latent_dimensions(study)
    assert selected_dimension == 12
    assert ranking.index[0] == 12
    assert ranking.loc[4, "synthetic_quality_rank"] == pytest.approx(3.0)
    assert ranking.loc[4, "synthetic_shapes_rank"] == pytest.approx(1.0)
    assert ranking.loc[4, "synthetic_pair_trends_rank"] == pytest.approx(2.0)
    assert ranking.loc[4, "generative_mean_rank"] == pytest.approx(2.0)
    assert ranking.loc[12, "generative_mean_rank"] == pytest.approx(5.0 / 3.0)


def test_rank_ladis_latent_dimensions_uses_quality_as_tie_breaker() -> None:
    """
    When generative mean ranks tie, higher synthetic Quality Score must win.
    """

    study = pd.DataFrame({"synthetic_quality": [0.95, 0.90, 0.85], "synthetic_shapes": [0.90, 0.85, 0.95], "synthetic_pair_trends": [0.80, 0.95, 0.90]},
        index=pd.Index([4, 8, 12], name="latent_dimension"))
    ranking, selected_dimension = rank_ladis_latent_dimensions(study)
    assert np.allclose(ranking["generative_mean_rank"].to_numpy(dtype=float), 2.0)
    assert selected_dimension == 4
    assert ranking.index[0] == 4
    assert ranking.iloc[0]["synthetic_quality"] == pytest.approx(0.95)


def test_rank_ladis_latent_dimensions_validates_input() -> None:
    """Missing metrics and empty studies must fail clearly."""

    incomplete = pd.DataFrame({"synthetic_quality": [0.9], "synthetic_shapes": [0.8]}, index=[8])
    with pytest.raises(ValueError, match="missing required columns"):
        rank_ladis_latent_dimensions(incomplete)
    empty = pd.DataFrame(columns=["synthetic_quality", "synthetic_shapes", "synthetic_pair_trends"])
    with pytest.raises(ValueError, match="study is empty"):
        rank_ladis_latent_dimensions(empty)


def test_build_ladis_clean_z0_candidate_configurations() -> None:
    """
    The controlled denoiser search must contain exactly the three variants.

    Together with the reference configuration supplied by the caller, these
    yield the four configurations reported for LaDIS.
    """

    reference = _reference_diffusion_configuration()
    reference_before = deepcopy(reference)
    candidates = build_ladis_clean_z0_candidate_configurations(ref_conf=reference)
    # The helper must not mutate the supplied reference configuration
    assert reference == reference_before
    assert len(candidates) == 3
    candidate_by_name = {candidate["name"]: candidate for candidate in candidates}
    assert set(candidate_by_name) == {"lower_lr_clean_z0_ladis", "wide_clean_z0_ladis", "wide_lower_lr_clean_z0_ladis"}
    lower_lr = candidate_by_name["lower_lr_clean_z0_ladis"]
    assert lower_lr["hidden_dimensions"] == (256, 512, 512, 256)
    assert lower_lr["learning_rate"] == pytest.approx(2.5e-4)
    wide = candidate_by_name["wide_clean_z0_ladis"]
    assert wide["hidden_dimensions"] == (512, 1024, 1024, 512)
    assert wide["learning_rate"] == pytest.approx(5e-4)
    wide_lower_lr = candidate_by_name["wide_lower_lr_clean_z0_ladis"]
    assert wide_lower_lr["hidden_dimensions"] == (512, 1024, 1024, 512)
    assert wide_lower_lr["learning_rate"] == pytest.approx(2.5e-4)
    # Every other configuration item must remain inherited from the reference
    controlled_keys = {"name", "hidden_dimensions", "learning_rate"}
    for candidate in candidates:
        for key, value in reference.items():
            if key not in controlled_keys:
                assert candidate[key] == value


def test_build_ladis_clean_z0_candidate_configurations_requires_name() -> None:
    """A reference configuration without an identifier is invalid."""

    reference = _reference_diffusion_configuration()
    reference.pop("name")
    with pytest.raises(ValueError, match="must contain a `name`"):
        build_ladis_clean_z0_candidate_configurations(ref_conf=reference)


def test_summarize_ladis_generated_latent_identical_distributions() -> None:
    """
    Identical latent tables must have unit std ratios and zero correlation error.
    """

    latent = pd.DataFrame({"z_00": [-1.0, -0.25, 0.50, 1.25, 0.75], "z_01": [0.20, 1.10, -0.80, 0.35, -0.40],
        "z_02": [1.50, -0.50, 0.25, -1.25, 0.60], "z_03": [-0.30, 0.85, 1.20, -0.70, 0.10]}, dtype=np.float64)
    summary = _summarize_ladis_generated_latent(gen_latent_data=latent.copy(), training_latent_data=latent.copy())
    assert summary["median_generated_to_training_std_ratio"] == pytest.approx(1.0)
    assert summary["minimum_generated_to_training_std_ratio"] == pytest.approx(1.0)
    assert summary["maximum_generated_to_training_std_ratio"] == pytest.approx(1.0)
    assert summary["mean_absolute_latent_correlation_error"] == pytest.approx(0.0, abs=1e-15)
    assert bool(summary["generated_latent_values_are_finite"])


@pytest.mark.parametrize(("candidate_metrics", "expected_candidate"),
    [
        # First criterion: highest Quality Score wins
        ({REFERENCE_NAME: (0.90, 0.20, 1.00, 100), "lower_lr_clean_z0_ladis": (0.95, 0.90, 1.50, 500), "wide_clean_z0_ladis": (0.92, 0.01, 1.00, 50),
            "wide_lower_lr_clean_z0_ladis": (0.91, 0.01, 1.00, 25)}, "lower_lr_clean_z0_ladis"),
        # Second criterion: if quality ties, lower latent CE wins
        ({REFERENCE_NAME: (0.95, 0.20, 1.00, 100), "lower_lr_clean_z0_ladis": (0.95, 0.10, 1.30, 500), "wide_clean_z0_ladis": (0.95, 0.30, 1.00, 50),
            "wide_lower_lr_clean_z0_ladis": (0.95, 0.40, 1.00, 25)}, "lower_lr_clean_z0_ladis"),
        # Third criterion: if quality and CE tie, std ratio closest to one wins
        ({REFERENCE_NAME: (0.95, 0.10, 1.20, 100), "lower_lr_clean_z0_ladis": (0.95, 0.10, 0.95, 500), "wide_clean_z0_ladis": (0.95, 0.10, 1.10, 50),
            "wide_lower_lr_clean_z0_ladis": (0.95, 0.10, 1.30, 25)}, "lower_lr_clean_z0_ladis"),
        # Fourth criterion: if all statistical criteria tie, fewer params wins
        ({REFERENCE_NAME: (0.95, 0.10, 1.00, 100), "lower_lr_clean_z0_ladis": (0.95, 0.10, 1.00, 90), "wide_clean_z0_ladis": (0.95, 0.10, 1.00, 80),
            "wide_lower_lr_clean_z0_ladis": (0.95, 0.10, 1.00, 70)}, "wide_lower_lr_clean_z0_ladis"),
    ])
def test_run_ladis_clean_z0_selection_hierarchy(monkeypatch, candidate_metrics: dict[str, tuple[float, float, float, int]],
    expected_candidate: str) -> None:
    """
    Verify the complete deterministic clean-z0 selection hierarchy.

    Expensive candidate fitting and generation are replaced with deterministic
    stubs. This isolates the selection rule itself:
        1. higher Quality Score,
        2. lower latent correlation error,
        3. smaller |median generated/train latent std ratio - 1|,
        4. fewer parameters.

    No selected candidate may be refitted after ranking.
    """

    train_latent = pd.DataFrame(np.arange(80, dtype=np.float32).reshape(20, 4), columns=["z_00", "z_01", "z_02", "z_03"])
    val_latent = pd.DataFrame(np.arange(40, dtype=np.float32).reshape(10, 4), columns=train_latent.columns)
    real_validation_data = pd.DataFrame({ "num": [0.0, 1.0, 2.0, 3.0], "cat": [0, 1, 0, 1]})
    fitted_training_data = real_validation_data.copy()
    fit_calls: list[str] = []

    def fake_fit_ladis_clean_z0_diffusion(*, train_latent_data, val_latent_data, conf, random_state, device, verbose):
        candidate_name = str(conf["name"])
        fit_calls.append(candidate_name)
        model = SimpleNamespace(cand_name=candidate_name)
        history = pd.DataFrame({"step": [1], "validation_z0_mse": [0.5]})
        model_parameters = candidate_metrics[candidate_name][3]
        summary = pd.Series({"model_parameters": int(model_parameters), "best_step": 1, "best_validation_z0_mse": 0.5}, name="value")
        return model, history, summary

    def fake_evaluate_ladis_clean_z0_denoising(*, diffusion_model, val_latent_data, timesteps, eval_rows, random_state, batch_size, device):
        return pd.DataFrame({"candidate": [diffusion_model.cand_name], "z0_mse": [0.5]})

    def fake_evaluate_ladis_diffusion_candidate(*, cand_name, diffusion_model, autoencoder, training_latent_data, real_val_data, fitted_training_data,
        metadata, latent_standardization_artifact, num_quantile_transformer, mode_dequantization_artifact, num_artifact, cat_artifact, feature_order,
        num_columns, cat_columns, cat_cards, sample_rows, batch_size, random_state, device):
        quality, corr_error, std_ratio, _ = candidate_metrics[cand_name]
        quality_summary = pd.Series({"quality_score": quality}, name="value")
        property_scores = pd.DataFrame({"Property": ["Column Shapes", "Column Pair Trends"], "Score": [quality - 0.01, quality - 0.02]})
        latent_summary = pd.Series({"mean_absolute_latent_correlation_error": corr_error, "median_generated_to_training_std_ratio": std_ratio},
            name="value")
        return {"quality_summary": quality_summary, "property_scores": property_scores, "latent_summary": latent_summary}

    monkeypatch.setattr(selection_module, "fit_ladis_clean_z0_diffusion", fake_fit_ladis_clean_z0_diffusion)
    monkeypatch.setattr(selection_module, "evaluate_ladis_clean_z0_denoising", fake_evaluate_ladis_clean_z0_denoising)
    monkeypatch.setattr(selection_module, "_evaluate_ladis_diffusion_candidate", fake_evaluate_ladis_diffusion_candidate)
    reference_configuration = _reference_diffusion_configuration()
    result = run_ladis_clean_z0_selection(train_latent_data=train_latent, val_latent_data=val_latent, autoencoder=SimpleNamespace(),
        ref_conf=reference_configuration, real_val_data=real_validation_data, fitted_training_data=fitted_training_data, metadata={"columns": {}},
        latent_standardization_artifact={"latent_columns": train_latent.columns.tolist()}, num_quantile_transformer=SimpleNamespace(), 
        mode_dequantization_artifact={}, num_artifact={}, cat_artifact={}, feature_order=["num", "cat"], num_columns=["num"], cat_columns=["cat"],
        cat_cards={"cat": 2}, random_state=42, device=torch.device("cpu"), batch_size=4, diagnostic_timesteps=(0,), diagnostic_rows=4, verbose=False)
    expected_fit_order = [REFERENCE_NAME, "lower_lr_clean_z0_ladis", "wide_clean_z0_ladis", "wide_lower_lr_clean_z0_ladis"]
    # Exactly four candidates are fitted, once each
    assert fit_calls == expected_fit_order
    assert len(fit_calls) == 4
    assert len(set(fit_calls)) == 4
    assert result["selected_candidate"] == expected_candidate
    assert result["selected_configuration"]["name"] == expected_candidate
    assert result["selection"].iloc[0]["candidate"] == expected_candidate
    assert result["selected_model"].cand_name == expected_candidate
    # The winning already-fitted checkpoint is retained: no post-selection refit
    assert result["refit_after_selection"] is False
    assert set(result["models"]) == set(expected_fit_order)
    assert set(result["histories"]) == set(expected_fit_order)
    assert set(result["training_summaries"]) == set(expected_fit_order)
    assert set(result["results"]) == set(expected_fit_order)
    # Check the explicit error term used by the third hierarchy criterion
    selection = result["selection"]
    expected_std_error = (selection["median_generated_latent_std_ratio"].sub(1.0).abs())
    np.testing.assert_allclose(selection["latent_std_ratio_error"].to_numpy(dtype=float), expected_std_error.to_numpy(dtype=float), rtol=0.0, atol=0.0)


def test_run_ladis_clean_z0_selection_validates_latent_dimension() -> None:
    """Reference latent dimensionality must match the prepared latent table."""

    train_latent = pd.DataFrame(np.ones((5, 4), dtype=np.float32), columns=["z_00", "z_01", "z_02", "z_03"])
    val_latent = train_latent.copy()
    reference = _reference_diffusion_configuration()
    reference["latent_dimension"] = 8
    with pytest.raises(RuntimeError, match="does not match the selected latent dimension"):
        run_ladis_clean_z0_selection(train_latent_data=train_latent, val_latent_data=val_latent, autoencoder=SimpleNamespace(), ref_conf=reference,
            real_val_data=pd.DataFrame({"x": [1.0]}), fitted_training_data=pd.DataFrame({"x": [1.0]}), metadata={"columns": {}},
            latent_standardization_artifact={}, num_quantile_transformer=SimpleNamespace(), mode_dequantization_artifact={}, num_artifact={},
            cat_artifact={}, feature_order=["x"], num_columns=["x"], cat_columns=[], cat_cards={}, random_state=42, device=torch.device("cpu"),
            verbose=False)