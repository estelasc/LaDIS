"""
CICIoT2023-specific labeling for the shared LaDIS PCAP representation.

CICIoT2023 experiments assign one scenario label to all flows extracted from
the corresponding folder. Feature computation itself is implemented only once
in ``core.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .core import FeatureExtractionResult, PCAPFeatureExtractor, discover_pcap_files, validate_feature_table_schema


def extract_ciciot_pcaps(pcap_paths: Sequence[str | Path], *, label: str, output_path: str | Path | None = None, extractor: PCAPFeatureExtractor | None = None,
    verbose: bool = True) -> FeatureExtractionResult:
    """Extract an explicitly supplied set of CICIoT2023 PCAP files."""

    if not isinstance(verbose, bool):
        raise TypeError("`verbose` must be Boolean.")
    if not str(label).strip():
        raise ValueError("`label` must be a non-empty string.")
    pcap_paths = sorted([Path(path) for path in pcap_paths], key=lambda path: path.name.lower())
    if not pcap_paths:
        raise ValueError("At least one CICIoT2023 PCAP file must be supplied.")
    extractor = extractor or PCAPFeatureExtractor()
    partitions: list[pd.DataFrame] = []
    diagnostic_records: list[dict[str, Any]] = []
    for pcap_path in pcap_paths:
        if verbose:
            print(f"Processing PCAP: {pcap_path.name}")
        data = extractor.extract_pcap(pcap_path, label=label)
        partitions.append(data)
        diagnostic_records.append({"pcap": pcap_path.name, "extracted_flows": len(data), "output_flows": len(data)})
        if verbose:
            print(f"  output={len(data)}")
    data = pd.concat(partitions, ignore_index=True)
    validate_feature_table_schema(data)
    output_path_obj = None if output_path is None else Path(output_path)
    if output_path_obj is not None:
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path_obj, index=False)
    return FeatureExtractionResult(data=data, diagnostics=pd.DataFrame(diagnostic_records), output_path=output_path_obj)


def extract_ciciot_folder(folder_path: str | Path, *, label: str | None = None, output_path: str | Path | None = None,
    extractor: PCAPFeatureExtractor | None = None, verbose: bool = True) -> FeatureExtractionResult:
    """Extract one CICIoT2023 scenario folder into the 114-column flow table."""

    folder_path = Path(folder_path)
    pcap_files = discover_pcap_files(folder_path)
    if not pcap_files:
        raise FileNotFoundError(f"No supported PCAP files were found in {folder_path}")
    if label is None:
        label = folder_path.name
    if output_path is None:
        output_path = folder_path / f"{label}_features.csv"
    return extract_ciciot_pcaps(pcap_files, label=label, output_path=output_path, extractor=extractor, verbose=verbose)