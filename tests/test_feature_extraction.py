"""
Lightweight regression tests for the optional LaDIS PCAP feature extractor.

These tests validate the public feature schema and dataset-independent helper
behavior without requiring access to Bot-IoT or CICIoT2023 packet captures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scapy")

from ladis.feature_extraction import (BASE_FLOW_COLUMNS, FLOW_TABLE_COLUMNS, STATISTIC_COLUMNS, TCP_FLAG_COLUMNS, canonical_flow_key, discover_pcap_files,
    split_botiot_traffic, validate_feature_table_schema)
from ladis.feature_extraction.core import safe_stats


def _build_minimal_feature_table() -> pd.DataFrame:
    """Create a two-row table with the exact canonical 114-column schema."""

    rows = []
    for pcap_name, label in (("capture_1.pcap", "Keylogging"), ("capture_1.pcap", "Normal")):
        row = {column: 0 for column in FLOW_TABLE_COLUMNS}
        row["pcap"] = pcap_name
        row["src_ip"] = "10.0.0.1"
        row["dst_ip"] = "10.0.0.2"
        row["protocol"] = "TCP"
        row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows, columns=FLOW_TABLE_COLUMNS)


def test_canonical_feature_schema() -> None:
    """The experimental PCAP representation must remain exactly 114 columns."""

    assert len(BASE_FLOW_COLUMNS) == 23
    assert len(STATISTIC_COLUMNS) == 66
    assert len(TCP_FLAG_COLUMNS) == 24
    assert len(FLOW_TABLE_COLUMNS) == 114
    assert len(set(FLOW_TABLE_COLUMNS)) == 114
    assert FLOW_TABLE_COLUMNS[0] == "pcap"
    assert FLOW_TABLE_COLUMNS[-1] == "label"


def test_safe_stats_matches_numpy_population_statistics() -> None:
    """Summary statistics must preserve the notebook NumPy definitions."""

    values = [1.0, 2.0, 3.0]
    result = safe_stats(values, "x")
    assert result["x_min"] == 1.0
    assert result["x_max"] == 3.0
    assert result["x_mean"] == pytest.approx(np.mean(values))
    assert result["x_std"] == pytest.approx(np.std(values))
    assert result["x_var"] == pytest.approx(np.var(values))
    assert result["x_sum"] == 6.0


def test_safe_stats_returns_zeros_for_empty_input() -> None:
    """Empty packet-derived vectors must map to six zero-valued statistics."""

    result = safe_stats([], "x")
    assert result == {"x_min": 0.0, "x_max": 0.0, "x_mean": 0.0, "x_std": 0.0, "x_var": 0.0, "x_sum": 0.0}


def test_canonical_flow_key_is_bidirectional() -> None:
    """Reversing the packet direction must preserve the same canonical key."""

    forward_key = canonical_flow_key("10.0.0.1", "10.0.0.2", 80, 50_000, "TCP")
    reverse_key = canonical_flow_key("10.0.0.2", "10.0.0.1", 50_000, 80, "TCP")
    assert forward_key == reverse_key


def test_discover_pcap_files_supports_expected_extensions(tmp_path: Path) -> None:
    """PCAP discovery must include .pcap, .pcapng, and .cap files only."""

    for filename in ("a.pcap", "b.pcapng", "c.cap", "ignore.csv"):
        (tmp_path / filename).touch()
    discovered = discover_pcap_files(tmp_path)
    assert [path.name for path in discovered] == ["a.pcap", "b.pcapng", "c.cap"]


def test_validate_feature_table_schema_accepts_canonical_table() -> None:
    """A correctly ordered 114-column table must pass schema validation."""

    data = _build_minimal_feature_table()
    validate_feature_table_schema(data)


def test_split_botiot_traffic_separates_labels_and_writes_csvs(tmp_path: Path) -> None:
    """Matched Bot-IoT traffic must be separable into one table per label."""

    data = _build_minimal_feature_table()
    traffic_tables = split_botiot_traffic(data, output_dir=tmp_path)
    assert set(traffic_tables) == {"Keylogging", "Normal"}
    assert traffic_tables["Keylogging"].shape == (1, 114)
    assert traffic_tables["Normal"].shape == (1, 114)
    assert traffic_tables["Keylogging"]["label"].eq("Keylogging").all()
    assert traffic_tables["Normal"]["label"].eq("Normal").all()
    assert (tmp_path / "Keylogging_features.csv").is_file()
    assert (tmp_path / "Normal_features.csv").is_file()