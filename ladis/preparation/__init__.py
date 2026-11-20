"""
Dataset-preparation utilities for LaDIS.

This package converts raw tabular datasets into the validated, reproducible
representation consumed by the LaDIS generative pipeline.
"""

from .config import load_preparation_config
from .loading import load_tabular_data
from .schema import normalize_column_name, standardize_dataframe
from .audit import build_column_audit, build_infinite_value_summary, build_quality_summary
from .features import build_initial_feature_set, validate_identifier_candidates
from .types import build_model_variable_types, classify_variable_types
from .derived_features import (build_derived_feature_set, find_std_var_pairs, identify_rate_candidates, validate_ratio_reconstruction_rules,
    validate_sum_reconstruction_rules)
from .splitting import apply_split_assignment, split_train_validation_test
from .ports import fit_port_tokenizer, inverse_transform_port_tokens, prepare_port_splits, transform_port_columns
from .storage import load_preparation_bundle, save_preparation_bundle
from .pipeline import prepare_dataset

__all__ = ["load_preparation_config", "load_tabular_data", "normalize_column_name", "standardize_dataframe", "build_column_audit",
    "build_infinite_value_summary", "build_quality_summary", "build_initial_feature_set", "validate_identifier_candidates", 
    "build_model_variable_types", "classify_variable_types", "build_derived_feature_set", "find_std_var_pairs", "identify_rate_candidates", 
    "validate_ratio_reconstruction_rules", "validate_sum_reconstruction_rules", "apply_split_assignment", "split_train_validation_test", "fit_port_tokenizer",
    "inverse_transform_port_tokens", "prepare_port_splits", "transform_port_columns", "load_preparation_bundle", "save_preparation_bundle", "prepare_dataset"]