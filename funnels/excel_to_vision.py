"""
Excel Vision Extractor

Converts each sheet of an Excel file into a rendered image, then uses a
vision LLM to extract all financial data as structured rows.

This is the correct approach for financial statement exports: the files often
have merged cells, multi-row headers, and inconsistent formatting that makes
pandas-based parsing unreliable. The vision model reads the table exactly as
a human analyst would.

Usage:
    from funnels.excel_to_vision import extract_excel_via_vision
    rows = extract_excel_via_vision("financials.xlsx", llm, timeout=90)

Input:
    excel_path: str     — path to an .xlsx file (all sheets are processed)
    llm: LLMProvider    — configured LLM provider with a vision model
    timeout: int        — per-sheet vision call timeout in seconds

Output:
    List of dicts, one per data row across all sheets:
    [
        {
            "row_header": "Interest Income",
            "sheet": "Income Statement",
            "values": {
                "3 Months Ended Jun. 30, 2024": 348.0,
                "3 Months Ended Jun. 30, 2023": 238.0,
                "6 Months Ended Jun. 30, 2024": 698.0,
                "6 Months Ended Jun. 30, 2023": 451.0
            }
        },
        ...
    ]
"""

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_VISION_PROMPT = """You are reading a financial statement table from an Excel spreadsheet image.

Extract EVERY row of data you can see in the table.

Output ONLY a valid JSON array — no markdown, no explanation, just the JSON.

Each element in the array represents one data row:
{{
  "row_header": "exact row label from the leftmost column",
  "values": {{
    "exact column header 1": numeric_value_or_string,
    "exact column header 2": numeric_value_or_string
  }}
}}

Rules:
- Use the EXACT text you see for row_header and column header keys.
- For numeric cells, output the number as-is (do not scale or convert).
  If the table header says "$ in Millions", leave the values as shown (e.g. 25500, not 25500000000).
- If a cell is blank or shows only a dash, omit that key from "values".
- Skip section-label rows that have no numeric values in any column.
- Include ALL rows that have at least one numeric value.
- Column headers may span multiple rows — combine them with a space
  (e.g. "3 Months Ended" above "Jun. 30, 2024" becomes "3 Months Ended Jun. 30, 2024").

Output ONLY the JSON array, starting with [ and ending with ]."""


def _parse_vision_rows(raw: str, sheet_name: str) -> List[Dict[str, Any]]:
    """Parse the vision model's JSON array response into row dicts."""
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        assert match, f"No JSON array found in vision response for sheet '{sheet_name}'"
        data = json.loads(match.group(0))

    assert isinstance(data, list), f"Vision response for '{sheet_name}' was not a JSON array"

    rows = []
    for item in data:
        row_header = str(item.get("row_header", "")).strip()
        values = item.get("values", {})
        if not row_header or not isinstance(values, dict):
            continue
        # Keep only entries with at least one non-blank value
        clean_values: Dict[str, Any] = {}
        for k, v in values.items():
            if v is None or v == "":
                continue
            clean_values[str(k).strip()] = v
        if clean_values:
            rows.append({
                "row_header": row_header,
                "sheet": sheet_name,
                "values": clean_values,
            })
    return rows


def extract_excel_via_vision(excel_path: str, llm, timeout: int = 90) -> List[Dict[str, Any]]:
    """Extract all rows from every sheet in an Excel file using vision OCR.

    Each sheet is rendered to a PNG image and sent to the vision LLM. The
    model returns a structured list of {row_header, values} dicts which are
    collected across all sheets.

    Args:
        excel_path: Path to the .xlsx file.
        llm: LLMProvider instance with a vision model configured.
        timeout: Seconds to wait for each vision model call.

    Returns:
        List of {"row_header", "sheet", "values"} dicts.
    """
    from utils.excel_to_image import convert_excel_to_image
    import openpyxl

    wb = openpyxl.load_workbook(excel_path, data_only=True)
    sheet_names = wb.sheetnames
    logger.info(f"Excel '{excel_path}' has {len(sheet_names)} sheet(s): {sheet_names}")

    all_rows: List[Dict[str, Any]] = []

    for sheet_name in sheet_names:
        logger.info(f"Processing sheet: '{sheet_name}'")
        try:
            _, img_base64 = convert_excel_to_image(excel_path, sheet_name=sheet_name)
        except Exception as e:
            logger.warning(f"Could not render sheet '{sheet_name}' to image: {e}")
            continue

        logger.info(f"Sheet '{sheet_name}' rendered ({len(img_base64)} base64 chars). Calling vision model…")
        raw = llm.analyze_image(_VISION_PROMPT, img_base64, timeout=timeout, max_tokens=4096).get("analysis", "")

        if not raw or raw.strip().lower() == "none":
            logger.warning(f"Vision model returned nothing for sheet '{sheet_name}'")
            continue

        try:
            rows = _parse_vision_rows(raw, sheet_name)
            logger.info(f"Sheet '{sheet_name}': extracted {len(rows)} data rows")
            all_rows.extend(rows)
        except Exception as e:
            logger.error(f"Failed to parse vision response for sheet '{sheet_name}': {e}")
            logger.debug(f"Raw response was: {raw[:500]}")
            continue

    logger.info(f"Total rows extracted across all sheets: {len(all_rows)}")
    return all_rows
