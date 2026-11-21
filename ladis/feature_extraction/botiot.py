"""
Bot-IoT-specific labeling for the shared LaDIS PCAP feature representation.

The flow features themselves are defined in ``core.py``. This module only
implements the Bot-IoT ground-truth normalization and matching procedure used
by the experimental feature-extraction notebook.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import FLOW_TABLE_COLUMNS, FeatureExtractionResult, PCAPFeatureExtractor, canonical_flow_key, discover_pcap_files, validate_feature_table_schema


_REQUIRED_GROUND_TRUTH_COLUMNS = {"stime", "ltime", "proto", "sport", "dport", "saddr", "daddr", "attack", "category", "subcategory"}


def _parse_port(value: Any) -> int:
    """Normalize the heterogeneous port representation used by Bot-IoT."""

    if pd.isna(value):
        return 0
    value = str(value).strip()
    if value == "" or value.lower() in {"nan", "who"}:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        try:
            return int(value, 16)
        except (TypeError, ValueError):
            return 0


def _normalize_protocol(value: Any) -> str:
    """Normalize Bot-IoT protocol names to the extractor representation."""

    value = str(value).strip().lower()
    if value == "tcp":
        return "TCP"
    if value == "udp":
        return "UDP"
    if value == "icmp":
        return "ICMP"
    if value == "arp":
        return "ARP"
    return value.upper()


def load_botiot_ground_truth(gt_path: str | Path) -> pd.DataFrame:
    """
    Load and normalize a Bot-IoT ground-truth CSV.

    The implementation reproduces the notebook matching representation:
    timestamps are numerical, protocols and ports are normalized, and each row
    receives the same canonical bidirectional flow key used by PCAP extraction.
    """

    gt_path = Path(gt_path)
    if not gt_path.is_file():
        raise FileNotFoundError(f"Bot-IoT ground-truth file does not exist: {gt_path}")
    gt = pd.read_csv(gt_path, sep=None, engine="python")
    gt.columns = [str(column).strip().lower() for column in gt.columns]
    missing_columns = sorted(_REQUIRED_GROUND_TRUTH_COLUMNS.difference(gt.columns))
    if missing_columns:
        raise ValueError(f"Bot-IoT ground truth is missing required columns: {missing_columns}")
    gt["stime"] = pd.to_numeric(gt["stime"], errors="coerce")
    gt["ltime"] = pd.to_numeric(gt["ltime"], errors="coerce")
    gt["proto_norm"] = gt["proto"].apply(_normalize_protocol)
    gt["sport_norm"] = gt["sport"].apply(_parse_port)
    gt["dport_norm"] = gt["dport"].apply(_parse_port)
    gt["saddr"] = gt["saddr"].astype(str).str.strip()
    gt["daddr"] = gt["daddr"].astype(str).str.strip()
    gt["attack"] = pd.to_numeric(gt["attack"], errors="coerce").fillna(0).astype(int)
    gt["binary_label"] = np.where(gt["attack"] == 1, "Attack", "Normal")
    gt["category_label"] = np.where(gt["attack"] == 1, gt["category"].astype(str), "Normal")
    gt["subcategory_label"] = np.where(gt["attack"] == 1, gt["subcategory"].astype(str), "Normal")
    gt = gt.dropna(subset=["stime", "ltime"]).copy()
    gt["flow_key"] = [canonical_flow_key(saddr, daddr, int(sport), int(dport), proto) for saddr, daddr, sport, dport, proto in zip(gt["saddr"], gt["daddr"],
        gt["sport_norm"], gt["dport_norm"], gt["proto_norm"])]
    return gt


def _build_ground_truth_index(ground_truth: pd.DataFrame) -> dict[tuple[str, int, str, int, str], pd.DataFrame]:
    """Index normalized ground-truth rows by canonical flow key."""

    return {key: group.sort_values("stime").reset_index(drop=True) for key, group in ground_truth.groupby("flow_key")}


def _label_flow_from_ground_truth(row: Mapping[str, Any], ground_truth_index: Mapping[tuple[str, int, str, int, str], pd.DataFrame]) -> dict[str, Any]:
    """Apply the exact Bot-IoT key-plus-time-overlap matching procedure."""

    labeled_row = dict(row)
    key = canonical_flow_key(labeled_row["src_ip"], labeled_row["dst_ip"], int(labeled_row["src_port"]), int(labeled_row["dst_port"]), labeled_row["protocol"])
    if key not in ground_truth_index:
        labeled_row["binary_label"] = "Unknown"
        labeled_row["category_label"] = "Unknown"
        labeled_row["subcategory_label"] = "Unknown"
        labeled_row["matched_gt"] = 0
        return labeled_row
    candidates = ground_truth_index[key]
    flow_start = labeled_row["flow_start"]
    flow_end = labeled_row["flow_end"]
    overlap = candidates[(candidates["stime"] <= flow_end) & (candidates["ltime"] >= flow_start)]
    if len(overlap) == 0:
        labeled_row["binary_label"] = "Unknown"
        labeled_row["category_label"] = "Unknown"
        labeled_row["subcategory_label"] = "Unknown"
        labeled_row["matched_gt"] = 0
        return labeled_row
    overlap = overlap.copy()
    overlap["overlap_duration"] = np.minimum(overlap["ltime"], flow_end) - np.maximum(overlap["stime"], flow_start)
    best = overlap.loc[overlap["overlap_duration"].idxmax()]
    labeled_row["binary_label"] = best["binary_label"]
    labeled_row["category_label"] = best["category_label"]
    labeled_row["subcategory_label"] = best["subcategory_label"]
    labeled_row["matched_gt"] = 1
    return labeled_row


def extract_botiot_pcaps(pcap_paths: Sequence[str | Path], *, ground_truth_path: str | Path, scenario_label: str, output_path: str | Path | None = None,
    extractor: PCAPFeatureExtractor | None = None, verbose: bool = True) -> FeatureExtractionResult:
    """
    Extract and label one Bot-IoT scenario from an explicit list of PCAP files.

    Unmatched flows are removed exactly as in the experimental notebook.
    Normal ground-truth flows keep label ``Normal``; attack flows receive
    ``scenario_label``.
    """

    if not isinstance(verbose, bool):
        raise TypeError("`verbose` must be Boolean.")
    if not str(scenario_label).strip():
        raise ValueError("`scenario_label` must be a non-empty string.")
    pcap_paths = sorted([Path(path) for path in pcap_paths], key=lambda path: path.name.lower())
    if not pcap_paths:
        raise ValueError("At least one Bot-IoT PCAP file must be supplied.")
    extractor = extractor or PCAPFeatureExtractor()
    start_time = perf_counter()
    ground_truth = load_botiot_ground_truth(ground_truth_path)
    ground_truth_index = _build_ground_truth_index(ground_truth)
    gt_seconds = perf_counter() - start_time
    if verbose:
        print(f"Ground truth loaded: {ground_truth.shape}; {len(ground_truth_index)} flow keys; {gt_seconds:.2f} s.")
    output_path_obj = None if output_path is None else Path(output_path)
    if output_path_obj is not None:
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    final_partitions: list[pd.DataFrame] = []
    diagnostic_records: list[dict[str, Any]] = []
    first_write = True
    for pcap_path in pcap_paths:
        if verbose:
            print(f"Processing PCAP: {pcap_path.name}")
        raw_data = extractor.extract_pcap(pcap_path, label="Unknown")
        labeled_rows = [_label_flow_from_ground_truth(row, ground_truth_index) for row in raw_data.to_dict(orient="records")]
        labeled_data = pd.DataFrame(labeled_rows)
        extracted_flows = len(labeled_data)
        matched_mask = labeled_data["matched_gt"].eq(1)
        matched_flows = int(matched_mask.sum())
        unmatched_flows = int((~matched_mask).sum())
        final_data = labeled_data.loc[matched_mask].copy()
        final_data["final_label"] = np.where(final_data["binary_label"] == "Normal", "Normal", scenario_label)
        auxiliary_columns = ["label", "binary_label", "category_label", "subcategory_label", "matched_gt", "scenario"]
        final_data = final_data.drop(columns=[column for column in auxiliary_columns if column in final_data.columns])
        final_data = final_data.rename(columns={"final_label": "label"})
        final_data = final_data.loc[:, FLOW_TABLE_COLUMNS].reset_index(drop=True)
        validate_feature_table_schema(final_data)
        if output_path_obj is not None:
            final_data.to_csv(output_path_obj, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False
        final_partitions.append(final_data)
        diagnostic_records.append({"pcap": pcap_path.name, "extracted_flows": extracted_flows, "matched_flows": matched_flows, "unmatched_flows": unmatched_flows,
            "match_pct": (100.0 * matched_flows / extracted_flows if extracted_flows else np.nan), "output_flows": len(final_data)})
        if verbose:
            print(f"  extracted={extracted_flows}, matched={matched_flows}, unmatched={unmatched_flows}, output={len(final_data)}")
    if final_partitions:
        data = pd.concat(final_partitions, ignore_index=True)
    else:
        data = pd.DataFrame(columns=FLOW_TABLE_COLUMNS)
    validate_feature_table_schema(data)
    diagnostics = pd.DataFrame(diagnostic_records)
    return FeatureExtractionResult(data=data, diagnostics=diagnostics, output_path=output_path_obj)


def extract_botiot_folder(folder_path: str | Path, *, ground_truth_path: str | Path | None = None, scenario_label: str | None = None,
    output_path: str | Path | None = None, extractor: PCAPFeatureExtractor | None = None, verbose: bool = True) -> FeatureExtractionResult:
    """
    Extract one Bot-IoT scenario folder containing PCAP files and ground truth.
    """

    folder_path = Path(folder_path)
    pcap_files = discover_pcap_files(folder_path)
    if not pcap_files:
        raise FileNotFoundError(f"No supported PCAP files were found in {folder_path}")
    if ground_truth_path is None:
        csv_files = sorted(folder_path.glob("*.csv"))
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No Bot-IoT ground-truth CSV was found in {folder_path}")
        if len(csv_files) > 1:
            raise ValueError("More than one CSV file was found in the Bot-IoT scenario folder. Supply `ground_truth_path` explicitly for reproducible selection.")
        ground_truth_path = csv_files[0]
    if scenario_label is None:
        scenario_label = folder_path.name
    if output_path is None:
        output_path = folder_path / f"{scenario_label}_features.csv"
    return extract_botiot_pcaps(pcap_files, ground_truth_path=ground_truth_path, scenario_label=scenario_label, output_path=output_path,
        extractor=extractor, verbose=verbose)


def split_botiot_traffic(data: pd.DataFrame, *, output_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Split matched Bot-IoT flows into separate traffic-specific tables.

    Bot-IoT PCAP captures may contain both benign and attack traffic. After
    ground-truth matching, this function separates the extracted flow table by
    its final ``label`` value so that each traffic type can be analysed or
    modelled independently.

    Parameters
    ----------
    data: pandas.DataFrame
        Matched Bot-IoT flow table following the canonical 114-column LaDIS
        feature schema.
    output_dir: str or pathlib.Path, optional
        Directory where one CSV file per traffic label should be saved. When
        omitted, the separated tables are returned without being written.

    Returns
    -------
    dict of str to pandas.DataFrame
        Mapping from traffic label to its corresponding feature table.
    """

    validate_feature_table_schema(data)
    if data.empty:
        raise ValueError("`data` must contain at least one extracted flow.")
    traffic_tables: dict[str, pd.DataFrame] = {}
    for label, label_data in data.groupby("label", sort=True):
        label_name = str(label)
        label_table = label_data.reset_index(drop=True).copy()
        validate_feature_table_schema(label_table)
        traffic_tables[label_name] = label_table
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for label_name, label_table in traffic_tables.items():
            safe_label = label_name.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
            output_path = output_dir / f"{safe_label}_features.csv"
            label_table.to_csv(output_path, index=False)
    return traffic_tables