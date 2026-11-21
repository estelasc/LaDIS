"""
Optional PCAP-to-tabular feature extraction for LaDIS experiments.

This package reproduces the common 114-column flow representation used by the
Bot-IoT and CICIoT2023 experiments. It is optional: the LaDIS generative
pipeline can also start directly from a user-provided CSV, Excel file, Parquet
file, or pandas DataFrame.
"""

from .core import (BASE_FLOW_COLUMNS, DEFAULT_FLOW_TIMEOUT, DEFAULT_MAX_PACKETS_PER_FLOW, FLOW_TABLE_COLUMNS, STATISTIC_COLUMNS, TCP_FLAG_COLUMNS,
    FeatureExtractionResult, PCAPFeatureExtractor, canonical_flow_key, discover_pcap_files, validate_feature_table_schema)
from .botiot import extract_botiot_folder, extract_botiot_pcaps, load_botiot_ground_truth, split_botiot_traffic
from .ciciot import extract_ciciot_folder, extract_ciciot_pcaps

__all__ = ["BASE_FLOW_COLUMNS", "DEFAULT_FLOW_TIMEOUT", "DEFAULT_MAX_PACKETS_PER_FLOW", "FLOW_TABLE_COLUMNS", "STATISTIC_COLUMNS", "TCP_FLAG_COLUMNS",
    "FeatureExtractionResult", "PCAPFeatureExtractor", "canonical_flow_key", "discover_pcap_files", "validate_feature_table_schema", "load_botiot_ground_truth",
    "extract_botiot_pcaps", "extract_botiot_folder", "extract_ciciot_pcaps", "extract_ciciot_folder", "split_botiot_traffic"]