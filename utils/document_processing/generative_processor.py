"""
Generative Document Processor

Treats the uploaded Word document as a STYLE REFERENCE ONLY.

Architecture:
  1. Extract writing style from the full reference Word document (one LLM call).
  2. Extract all financial data from the Excel file via vision OCR — every sheet,
     every row, exactly as a human analyst would read the table.
  3. For each data row in the Excel, generate one paragraph in the reference style.

The number of output paragraphs equals the number of data rows in the Excel,
NOT the number of paragraphs in the reference Word document.

Usage:
    from utils.document_processing.generative_processor import generate_from_structure
    results = generate_from_structure("ref.docx", "data.xlsx", ["report.pdf"])

Input:
    docx_path: str          — style reference Word document (.docx)
    excel_path: str         — Excel file with financial data (.xlsx); all sheets processed
    qualitative_paths: list — optional PDF/.docx qualitative source files

Output:
    List of (style_sample, generated_paragraph, confidence, description) tuples.
    style_sample is the reference paragraph whose style was used as guidance.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from config import Config
from funnels.excel_to_vision import extract_excel_via_vision
from funnels.extract import read_docx
from funnels.llm_provider import get_llm_provider
from funnels.qualitative_extractor import extract_qualitative_sources
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _parse_json_response(text: str) -> Any:
    """Strip markdown fences and parse a JSON value from an LLM response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _extract_style_profile(paragraphs: Dict[int, str], llm, timeout: int) -> Dict[str, Any]:
    """Analyze the full reference document once and return a style profile.

    The profile captures how the author writes financial commentary:
    number formats, sentence rhythm, comparison language, tense, etc.
    """
    non_trivial = [s for s in paragraphs.values() if len(s.strip()) >= 30]
    sample = non_trivial[:10]
    combined = "\n\n".join(f'[{i+1}] "{s}"' for i, s in enumerate(sample))

    prompt = f"""You are a writing-style analyst.

The paragraphs below are from a financial document used ONLY as a style reference.
Extract the WRITING CONVENTIONS — not the content.

Paragraphs:
{combined}

Output ONLY a JSON object:
{{
  "number_format": "how numbers appear, e.g. '$X million', '$Xb', 'X%'",
  "sentence_structure": "typical shape, e.g. 'metric first, then period, then YoY change, then cause'",
  "comparison_pattern": "how changes are expressed, e.g. 'increased $X million, or X%, in the [period]'",
  "tense": "past / present",
  "vocabulary": ["list", "of", "characteristic", "verbs", "or", "phrases"],
  "paragraph_length": "short (1-2 sentences) / medium (2-3 sentences) / long (4+ sentences)",
  "sample_sentences": ["best representative sentence 1", "best representative sentence 2"]
}}"""

    raw = llm.analyze_text(prompt, timeout=timeout, max_tokens=800).get("analysis", "")
    assert raw and raw != "none", "Style profile extraction returned empty response"
    return _parse_json_response(raw)


def _find_best_style_sample(row_header: str, sheet: str, paragraphs: Dict[int, str]) -> str:
    """Return the reference paragraph most topically similar to this Excel row.

    Uses word-overlap. Falls back to first non-trivial paragraph.
    """
    query_words = set((row_header + " " + sheet).lower().split())
    best, best_score = "", 0
    for text in paragraphs.values():
        if len(text.strip()) < 30:
            continue
        score = len(query_words & set(text.lower().split()))
        if score > best_score:
            best_score = score
            best = text
    if not best:
        for text in paragraphs.values():
            if len(text.strip()) >= 30:
                return text
    return best


def _get_qualitative_context(row_header: str, passages: List[str], max_passages: int = 2) -> List[str]:
    """Return the most topically relevant qualitative passages."""
    if not passages:
        return []
    query_words = set(row_header.lower().split())
    scored = sorted(
        passages,
        key=lambda p: len(query_words & set(p.lower().split())),
        reverse=True,
    )
    return [p for p in scored[:max_passages] if len(query_words & set(p.lower().split())) > 0]


def _generate_paragraph(
    row: Dict[str, Any],
    style_profile: Dict[str, Any],
    style_sample: str,
    qualitative_context: List[str],
    llm,
    timeout: int,
) -> Optional[str]:
    """Generate one financial commentary paragraph for a single Excel data row."""
    qual_block = ""
    if qualitative_context:
        qual_block = "\nQualitative context (use only where clearly relevant):\n" + \
                     "\n".join(f"- {p}" for p in qualitative_context)

    values_json = json.dumps(row["values"], indent=2)
    sheet = row.get("sheet", "")

    prompt = f"""You are writing financial commentary for a document.

== TASK ==
Write ONE paragraph about: {row['row_header']}
Statement: {sheet}

== DATA (use ALL values shown — do not omit any period) ==
{values_json}

== STYLE REFERENCE (match the style, do NOT copy verbatim) ==
"{style_sample}"

== STYLE CONVENTIONS ==
- Number format: {style_profile.get('number_format', 'match sample')}
- Sentence structure: {style_profile.get('sentence_structure', 'match sample')}
- How changes are expressed: {style_profile.get('comparison_pattern', 'match sample')}
- Tense: {style_profile.get('tense', 'match sample')}
- Vocabulary: {', '.join(style_profile.get('vocabulary', []))}
- Paragraph length: {style_profile.get('paragraph_length', 'medium')}
{qual_block}

== RULES ==
1. Reference every time period in the data.
2. Compute and state the change (absolute and/or %) between periods where meaningful.
3. Match the sentence structure and number format of the style sample exactly.
4. Write only financial commentary — no headings, no bullets, no meta-commentary.
5. Output ONLY the paragraph — nothing else.

If the data is insufficient for meaningful commentary, output exactly: SKIP"""

    raw = llm.analyze_text(prompt, timeout=timeout, max_tokens=400).get("analysis", "").strip()
    if not raw or raw.upper() in ("NONE", "SKIP"):
        return None
    return raw


def generate_from_structure(
        docx_path: str,
        excel_path: str,
        qualitative_paths: List[str],
        timeout: int = None) -> List[Tuple[str, str, float, str]]:
    """Generate financial commentary for every Excel row, guided by the reference doc style.

    Args:
        docx_path: Path to style reference Word document (.docx)
        excel_path: Path to Excel data file (.xlsx); all sheets are processed
        qualitative_paths: Optional list of PDF/.docx qualitative source paths
        timeout: LLM timeout in seconds (uses config default if None)

    Returns:
        List of (style_sample, generated_paragraph, confidence, description) tuples.
    """
    logger.info("=" * 80)
    logger.info("GENERATIVE DOCUMENT PROCESSING")
    logger.info("=" * 80)
    t_start = time.time()

    config = Config()
    timeout = timeout or config.get_model_config()["llm"].get("timeout", 60)
    llm = get_llm_provider()

    paragraphs = read_docx(docx_path)
    qualitative_passages = extract_qualitative_sources(qualitative_paths) if qualitative_paths else []

    logger.info(f"Reference paragraphs: {len(paragraphs)}")
    logger.info(f"Qualitative passages: {len(qualitative_passages)}")

    # Phase 1 — extract writing style from reference doc
    logger.info("Phase 1: Extracting style profile …")
    style_profile = _extract_style_profile(paragraphs, llm, timeout)
    logger.info(f"Style: {json.dumps(style_profile, indent=2)[:300]}")

    # Phase 2 — extract all Excel data via vision OCR
    logger.info("Phase 2: Extracting Excel data via vision OCR …")
    excel_rows = extract_excel_via_vision(excel_path, llm, timeout=max(timeout, 90))
    assert excel_rows, "Vision extraction returned no data rows from Excel"
    logger.info(f"Excel rows extracted: {len(excel_rows)}")

    # Phase 3 — generate one paragraph per Excel row
    results: List[Tuple[str, str, float, str]] = []

    for row in tqdm(excel_rows, desc="Generating", unit="row"):
        row_header = row.get("row_header", "")
        sheet = row.get("sheet", "")
        logger.info(f"Generating: [{sheet}] {row_header}")

        style_sample = _find_best_style_sample(row_header, sheet, paragraphs)
        qual_context = _get_qualitative_context(row_header, qualitative_passages)

        try:
            paragraph = _generate_paragraph(row, style_profile, style_sample, qual_context, llm, timeout)
        except Exception as e:
            logger.warning(f"  Generation failed for '{row_header}': {e}")
            continue

        if not paragraph:
            logger.debug(f"  Skipped: '{row_header}'")
            continue

        len_ratio = len(paragraph) / max(len(style_sample), 1)
        confidence = min(0.70 + max(0.0, 1.0 - abs(1.0 - len_ratio)) * 0.25, 0.95)
        description = f"{sheet} — {row_header}"

        logger.info(f"  → '{paragraph[:100]}'")
        results.append((style_sample, paragraph, confidence, description))

    logger.info("=" * 80)
    logger.info(f"COMPLETE: {len(results)} paragraphs in {time.time() - t_start:.2f}s")
    logger.info("=" * 80)
    return results
