"""
Excel data extractor (pandas-based, simple format).

Used as a fallback for Excel files with a clean single-row header.
The generative pipeline uses funnels/excel_to_vision.py instead,
which handles all real-world financial statement formats correctly.

Usage:
    from funnels.excel_utils import excel_to_list
    rows = excel_to_list("/path/to/data.xlsx")

Input:
    excel_path: str — path to an .xlsx file with a simple header row

Output:
    List of dicts:
    [
        {
            "row_header": str,
            "values": {"Column A": 123.0, "Column B": 456.0}
        },
        ...
    ]
    Rows with an empty first cell or no numeric values are skipped.
"""

import logging
import pandas as pd
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def excel_to_list(excel_path: str) -> List[Dict[str, Any]]:
    """Convert a simple Excel file to a list of metric rows.

    Args:
        excel_path: Path to the .xlsx file.

    Returns:
        List of {"row_header": str, "values": {col_header: value}} dicts.
    """
    df = pd.read_excel(excel_path)
    assert len(df.columns) >= 2, "Excel file must have at least two columns"

    row_header_col = df.columns[0]
    metrics: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        raw_header = row[row_header_col]
        if pd.isna(raw_header) or str(raw_header).strip() == "":
            continue

        row_header = str(raw_header).strip()
        values: Dict[str, Any] = {}

        for col in df.columns[1:]:
            cell = row[col]
            if pd.isna(cell):
                continue
            try:
                values[str(col)] = float(cell)
            except (ValueError, TypeError):
                text = str(cell).strip()
                if text:
                    values[str(col)] = text

        if values:
            metrics.append({"row_header": row_header, "values": values})

    logger.info(f"Extracted {len(metrics)} metric rows from {excel_path}")
    return metrics
