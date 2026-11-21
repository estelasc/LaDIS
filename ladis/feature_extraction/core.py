"""
Shared PCAP feature extraction for LaDIS.

This module reproduces the flow construction and feature definitions used in
the Bot-IoT and CICIoT2023 feature extraction associated with LaDIS paper.
It converts IP packets into bidirectional flows and represents each completed
flow using the common 114-column table used by the experiments.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from scapy.all import ICMP, IP, TCP, UDP, PcapReader, conf
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    raise ImportError("PCAP feature extraction requires the optional LaDIS PCAP dependencies. "
        "Install them with `pip install -e '.[pcap]'` from the repository root.") from exc
try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    raise ImportError("PCAP feature extraction requires the optional LaDIS PCAP dependencies. "
        "Install them with `pip install -e '.[pcap]'` from the repository root.") from exc


logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
conf.verb = 0

DEFAULT_FLOW_TIMEOUT = 30.0
DEFAULT_MAX_PACKETS_PER_FLOW = 5_000
SUPPORTED_PCAP_PATTERNS = ("*.pcap", "*.pcapng", "*.cap")

STATISTIC_NAMES = ("min", "max", "mean", "std", "var", "sum")
STATISTIC_PREFIXES = ("pkt_len", "fwd_pkt_len", "bwd_pkt_len", "payload_len", "fwd_payload_len", "bwd_payload_len", "header_len", "ttl", "flow_iat",
    "fwd_iat", "bwd_iat")
TCP_FLAG_NAMES = ("fin", "syn", "rst", "psh", "ack", "urg", "ece", "cwe")
BASE_FLOW_COLUMNS = ("pcap", "flow_start", "flow_end", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "flow_duration", "total_pkts", "total_bytes",
    "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "flow_pkts_s", "flow_bytes_s", "fwd_pkts_s", "bwd_pkts_s", "fwd_bytes_s", "bwd_bytes_s",
    "down_up_ratio", "byte_down_up_ratio")
STATISTIC_COLUMNS = tuple(f"{prefix}_{statistic}" for prefix in STATISTIC_PREFIXES for statistic in STATISTIC_NAMES)
TCP_FLAG_COLUMNS = tuple(column for flag in TCP_FLAG_NAMES for column in (f"{flag}_cnt", f"fwd_{flag}_cnt", f"bwd_{flag}_cnt"))
FLOW_TABLE_COLUMNS = (*BASE_FLOW_COLUMNS, *STATISTIC_COLUMNS, *TCP_FLAG_COLUMNS, "label")
if len(FLOW_TABLE_COLUMNS) != 114:
    raise RuntimeError("The canonical LaDIS PCAP feature schema must contain exactly 114 columns.")


@dataclass
class FeatureExtractionResult:
    """
    Result returned by a dataset-level feature-extraction helper.

    Attributes
    ----------
    data: pandas.DataFrame
        Extracted 114-column flow table.
    diagnostics: pandas.DataFrame
        Per-PCAP extraction diagnostics.
    output_path: pathlib.Path or None
        CSV path when the result was also written to disk.
    """

    data: pd.DataFrame
    diagnostics: pd.DataFrame
    output_path: Path | None = None


def safe_stats(values: Sequence[float], prefix: str) -> dict[str, float]:
    """
    Compute the six statistical descriptors used by the LaDIS PCAP schema.
    """

    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {f"{prefix}_min": 0.0, f"{prefix}_max": 0.0, f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0, f"{prefix}_var": 0.0, f"{prefix}_sum": 0.0}
    return {f"{prefix}_min": float(np.min(values)), f"{prefix}_max": float(np.max(values)), f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)), f"{prefix}_var": float(np.var(values)), f"{prefix}_sum": float(np.sum(values))}


def get_proto_and_ports(packet: Any) -> tuple[str, int, int]:
    """
    Return protocol, source port, and destination port for one IP packet.
    """

    if TCP in packet:
        return "TCP", int(packet[TCP].sport), int(packet[TCP].dport)
    if UDP in packet:
        return "UDP", int(packet[UDP].sport), int(packet[UDP].dport)
    if ICMP in packet:
        return "ICMP", 0, 0
    return str(packet[IP].proto), 0, 0


def canonical_flow_key(src: str, dst: str, sport: int, dport: int, proto: str) -> tuple[str, int, str, int, str]:
    """
    Create the canonical bidirectional five-tuple used by the experiments.
    """

    endpoint_a = (src, sport)
    endpoint_b = (dst, dport)
    if endpoint_a <= endpoint_b:
        return src, sport, dst, dport, proto
    return dst, dport, src, sport, proto


def tcp_flags(packet: Any) -> dict[str, int]:
    """
    Extract the eight TCP flag indicators used by the experimental notebooks.

    Notes
    -----
    The historical LaDIS schema uses the column stem ``cwe`` for the TCP CWR
    flag. The spelling is preserved intentionally for exact schema
    compatibility with the paper datasets.
    """

    flags = {"fin": 0, "syn": 0, "rst": 0, "psh": 0, "ack": 0, "urg": 0, "ece": 0, "cwe": 0}
    if TCP not in packet:
        return flags
    value = int(packet[TCP].flags)
    flags["fin"] = int(bool(value & 0x01))
    flags["syn"] = int(bool(value & 0x02))
    flags["rst"] = int(bool(value & 0x04))
    flags["psh"] = int(bool(value & 0x08))
    flags["ack"] = int(bool(value & 0x10))
    flags["urg"] = int(bool(value & 0x20))
    flags["ece"] = int(bool(value & 0x40))
    flags["cwe"] = int(bool(value & 0x80))
    return flags


class _Flow:
    """Accumulate packet information for one canonical bidirectional flow."""

    def __init__(self, key: tuple[str, int, str, int, str], first_time: float, label: str, pcap_name: str) -> None:

        self.key = key
        self.label = label
        self.pcap_name = pcap_name
        self.start = first_time
        self.end = first_time
        (self.fwd_src, self.fwd_sport, self.fwd_dst, self.fwd_dport, self.proto) = key
        self.times: list[float] = []
        self.fwd_times: list[float] = []
        self.bwd_times: list[float] = []
        self.lengths: list[int] = []
        self.fwd_lengths: list[int] = []
        self.bwd_lengths: list[int] = []
        self.payload_lengths: list[int] = []
        self.fwd_payload_lengths: list[int] = []
        self.bwd_payload_lengths: list[int] = []
        self.header_lengths: list[int] = []
        self.ttls: list[int] = []
        self.flag_counts: defaultdict[str, int] = defaultdict(int)
        self.fwd_flag_counts: defaultdict[str, int] = defaultdict(int)
        self.bwd_flag_counts: defaultdict[str, int] = defaultdict(int)

    def add_packet(self, packet: Any, timestamp: float) -> None:
        """Add one packet to the flow using the notebook feature definitions."""

        src = packet[IP].src
        sport = 0
        dport = 0
        if TCP in packet:
            sport = int(packet[TCP].sport)
            dport = int(packet[TCP].dport)
        elif UDP in packet:
            sport = int(packet[UDP].sport)
            dport = int(packet[UDP].dport)
        is_forward = src == self.fwd_src and sport == self.fwd_sport
        length = int(len(packet))
        ip_header_len = (int(packet[IP].ihl * 4) if IP in packet and packet[IP].ihl else 0)
        transport_header_len = 0
        if TCP in packet:
            transport_header_len = (int(packet[TCP].dataofs * 4) if packet[TCP].dataofs else 0)
        elif UDP in packet:
            transport_header_len = 8
        header_len = ip_header_len + transport_header_len
        payload_len = max(0, length - header_len)
        self.start = min(self.start, timestamp)
        self.end = max(self.end, timestamp)
        self.times.append(timestamp)
        self.lengths.append(length)
        self.payload_lengths.append(payload_len)
        self.header_lengths.append(header_len)
        if IP in packet:
            self.ttls.append(int(packet[IP].ttl))
        flags = tcp_flags(packet)
        for name, value in flags.items():
            self.flag_counts[name] += value
        if is_forward:
            self.fwd_times.append(timestamp)
            self.fwd_lengths.append(length)
            self.fwd_payload_lengths.append(payload_len)
            for name, value in flags.items():
                self.fwd_flag_counts[name] += value
        else:
            self.bwd_times.append(timestamp)
            self.bwd_lengths.append(length)
            self.bwd_payload_lengths.append(payload_len)
            for name, value in flags.items():
                self.bwd_flag_counts[name] += value

    def to_row(self) -> dict[str, Any]:
        """Convert the completed flow into the canonical feature dictionary."""

        duration = max(self.end - self.start, 1e-9)
        times = np.sort(np.asarray(self.times))
        fwd_times = np.sort(np.asarray(self.fwd_times))
        bwd_times = np.sort(np.asarray(self.bwd_times))
        iat = np.diff(times)
        fwd_iat = np.diff(fwd_times)
        bwd_iat = np.diff(bwd_times)
        fwd_pkts = len(self.fwd_lengths)
        bwd_pkts = len(self.bwd_lengths)
        total_pkts = fwd_pkts + bwd_pkts
        fwd_bytes = np.sum(self.fwd_lengths) if fwd_pkts else 0
        bwd_bytes = np.sum(self.bwd_lengths) if bwd_pkts else 0
        total_bytes = fwd_bytes + bwd_bytes
        row: dict[str, Any] = {"pcap": self.pcap_name, "flow_start": self.start, "flow_end": self.end, "src_ip": self.fwd_src, "dst_ip": self.fwd_dst,
            "src_port": self.fwd_sport, "dst_port": self.fwd_dport, "protocol": self.proto, "flow_duration": duration, "total_pkts": total_pkts,
            "total_bytes": total_bytes, "fwd_pkts": fwd_pkts, "bwd_pkts": bwd_pkts, "fwd_bytes": float(fwd_bytes), "bwd_bytes": float(bwd_bytes),
            "flow_pkts_s": total_pkts / duration, "flow_bytes_s": total_bytes / duration, "fwd_pkts_s": fwd_pkts / duration, "bwd_pkts_s": bwd_pkts / duration,
            "fwd_bytes_s": fwd_bytes / duration, "bwd_bytes_s": bwd_bytes / duration, "down_up_ratio": bwd_pkts / max(fwd_pkts, 1),
            "byte_down_up_ratio": bwd_bytes / max(fwd_bytes, 1)}
        row.update(safe_stats(self.lengths, "pkt_len"))
        row.update(safe_stats(self.fwd_lengths, "fwd_pkt_len"))
        row.update(safe_stats(self.bwd_lengths, "bwd_pkt_len"))
        row.update(safe_stats(self.payload_lengths, "payload_len"))
        row.update(safe_stats(self.fwd_payload_lengths, "fwd_payload_len"))
        row.update(safe_stats(self.bwd_payload_lengths, "bwd_payload_len"))
        row.update(safe_stats(self.header_lengths, "header_len"))
        row.update(safe_stats(self.ttls, "ttl"))
        row.update(safe_stats(iat, "flow_iat"))
        row.update(safe_stats(fwd_iat, "fwd_iat"))
        row.update(safe_stats(bwd_iat, "bwd_iat"))
        for flag in TCP_FLAG_NAMES:
            row[f"{flag}_cnt"] = int(self.flag_counts[flag])
            row[f"fwd_{flag}_cnt"] = int(self.fwd_flag_counts[flag])
            row[f"bwd_{flag}_cnt"] = int(self.bwd_flag_counts[flag])
        row["label"] = self.label
        return row


def discover_pcap_files(folder_path: str | Path) -> list[Path]:
    """Return supported PCAP files directly contained in one folder."""

    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"PCAP folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"PCAP path is not a directory: {folder_path}")
    pcap_files: list[Path] = []
    for pattern in SUPPORTED_PCAP_PATTERNS:
        pcap_files.extend(folder_path.glob(pattern))
    return sorted(set(pcap_files), key=lambda path: path.name.lower())


def validate_feature_table_schema(data: pd.DataFrame) -> None:
    """Validate the exact 114-column LaDIS PCAP feature schema."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    observed_columns = data.columns.tolist()
    expected_columns = list(FLOW_TABLE_COLUMNS)
    if observed_columns != expected_columns:
        missing = [column for column in expected_columns if column not in observed_columns]
        extra = [column for column in observed_columns if column not in expected_columns]
        raise RuntimeError(f"Extracted PCAP feature schema does not match the canonical 114-column LaDIS schema. Missing: {missing}. Extra: {extra}.")


class PCAPFeatureExtractor:
    """
    Extract the common LaDIS flow representation from one or more PCAP files.

    Parameters
    ----------
    flow_timeout: float, default=30.0
        Maximum inactivity gap, in seconds, before an active flow is closed.
    max_packets_per_flow: int, default=5000
        Packet-count threshold used by the experimental notebooks.
    show_progress: bool, default=True
        Whether to show a tqdm packet-reading progress indicator.

    Notes
    -----
    The flow-closing condition intentionally reproduces the notebooks exactly:
    ``len(flow.times) > max_packets_per_flow`` is checked before adding the
    next packet. It is not changed to ``>=`` because doing so would alter the
    experimental feature extraction.
    """

    def __init__(self, *, flow_timeout: float = DEFAULT_FLOW_TIMEOUT, max_packets_per_flow: int = DEFAULT_MAX_PACKETS_PER_FLOW,
        show_progress: bool = True) -> None:

        if float(flow_timeout) <= 0.0:
            raise ValueError("`flow_timeout` must be positive.")
        if isinstance(max_packets_per_flow, bool) or int(max_packets_per_flow) <= 0:
            raise ValueError("`max_packets_per_flow` must be a positive integer.")
        if not isinstance(show_progress, bool):
            raise TypeError("`show_progress` must be Boolean.")
        self.flow_timeout = float(flow_timeout)
        self.max_packets_per_flow = int(max_packets_per_flow)
        self.show_progress = show_progress

    def _extract_pcap_rows(self, pcap_path: str | Path, *, label: str) -> list[dict[str, Any]]:

        pcap_path = Path(pcap_path)
        if not pcap_path.is_file():
            raise FileNotFoundError(f"PCAP file does not exist: {pcap_path}")
        active_flows: dict[tuple[str, int, str, int, str], _Flow] = {}
        finished_rows: list[dict[str, Any]] = []
        with PcapReader(str(pcap_path)) as reader:
            iterator = tqdm(reader, desc=pcap_path.name, leave=False, disable=not self.show_progress)
            for packet in iterator:
                if IP not in packet:
                    continue
                timestamp = float(packet.time)
                src = packet[IP].src
                dst = packet[IP].dst
                proto, sport, dport = get_proto_and_ports(packet)
                key = canonical_flow_key(src, dst, sport, dport, proto)
                if key in active_flows:
                    flow = active_flows[key]
                    if (timestamp - flow.end > self.flow_timeout or len(flow.times) > self.max_packets_per_flow):
                        finished_rows.append(flow.to_row())
                        active_flows[key] = _Flow(key, timestamp, label, pcap_path.name)
                else:
                    active_flows[key] = _Flow(key, timestamp, label, pcap_path.name)
                active_flows[key].add_packet(packet, timestamp)
        for flow in active_flows.values():
            finished_rows.append(flow.to_row())
        return finished_rows

    def extract_pcap(self, pcap_path: str | Path, *, label: str) -> pd.DataFrame:
        """Extract one PCAP into the canonical 114-column flow table."""

        rows = self._extract_pcap_rows(pcap_path, label=label)
        data = pd.DataFrame(rows, columns=FLOW_TABLE_COLUMNS)
        validate_feature_table_schema(data)
        return data

    def extract_pcaps(self, pcap_paths: Sequence[str | Path], *, label: str) -> pd.DataFrame:
        """Extract several PCAPs and concatenate their flow tables."""

        pcap_paths = [Path(path) for path in pcap_paths]
        if not pcap_paths:
            raise ValueError("At least one PCAP file must be supplied.")
        partitions = [self.extract_pcap(path, label=label) for path in pcap_paths]
        data = pd.concat(partitions, ignore_index=True)
        validate_feature_table_schema(data)
        return data