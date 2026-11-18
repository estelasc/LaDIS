"""
Tabular-data loading utilities for LaDIS preparation.

Supported file formats are CSV, TSV, Parquet, XLSX, and XLS. A pandas
DataFrame can also be supplied directly when LaDIS is used programmatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_TABULAR_SUFFIXES = {".csv", ".tsv", ".parquet", ".xlsx", ".xls"}


def load_tabular_data(source: str | Path | pd.DataFrame, *, sheet_name: str | int = 0) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a tabular dataset without applying model-specific preprocessing.

    Parameters
    ----------
    source: str, pathlib.Path, or pandas.DataFrame
        Input tabular dataset.
    sheet_name: str or int, default=0
        Excel sheet to load when the source is an Excel workbook.

    Returns
    -------
    data: pandas.DataFrame
        Loaded table with its original schema preserved.
    metadata: dict
        Basic information about the input source and loaded table.
    """

    if isinstance(source, pd.DataFrame):
        data = source.copy()
        if data.empty:
            raise ValueError("The supplied DataFrame is empty.")
        metadata = {"source_type": "dataframe", "source_filename": None, "input_format": "dataframe", "rows_in_file": int(data.shape[0]),
            "columns_in_file": int(data.shape[1]), "file_size_mb": None, 
            "dataframe_memory_mb": float(data.memory_usage(index=True, deep=True).sum() / (1024**2)), "sheet_name": None}
        return data, metadata
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"The input dataset was not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_TABULAR_SUFFIXES:
        raise ValueError(f"Unsupported tabular format '{suffix}'. Supported formats are: {sorted(SUPPORTED_TABULAR_SUFFIXES)}")
    if suffix == ".csv":
        data = pd.read_csv(path, low_memory=False)
    elif suffix == ".tsv":
        data = pd.read_csv(path, sep="\t", low_memory=False)
    elif suffix == ".parquet":
        data = pd.read_parquet(path)
    elif suffix in {".xlsx", ".xls"}:
        try:
            data = pd.read_excel(path, sheet_name=sheet_name)
        except ImportError as error:
            raise ImportError("Reading Excel files requires an appropriate pandas Excel engine. Install `openpyxl` for XLSX files and "
                "`xlrd` for legacy XLS files.") from error
    else:
        raise RuntimeError(f"No loader was configured for '{suffix}'.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The selected input produced more than one table. Specify one Excel sheet using `sheet_name`.")
    if data.empty:
        raise ValueError(f"The loaded dataset is empty: {path}")
    metadata = {"source_type": "file", "source_filename": path.name, "input_format": suffix.removeprefix("."), "rows_in_file": int(data.shape[0]),
        "columns_in_file": int(data.shape[1]), "file_size_mb": float(path.stat().st_size / (1024**2)),
        "dataframe_memory_mb": float(data.memory_usage(index=True, deep=True).sum() / (1024**2)),
        "sheet_name": (sheet_name if suffix in {".xlsx", ".xls"} else None)}
    return data, metadata