"""
Qualitative source text extractor.

Extracts plain-text passages from PDF and Word documents to serve as
qualitative context in the generative pipeline.

Usage:
    from funnels.qualitative_extractor import extract_qualitative_sources
    passages = extract_qualitative_sources(["/path/to/report.pdf", "/path/to/notes.docx"])

Input:
    paths: List[str] — file paths ending in .pdf or .docx

Output:
    List[str] — flat list of non-empty text passages across all source files
    Each passage is at least 20 characters (shorter fragments filtered out).
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def extract_from_pdf(path: str) -> List[str]:
    """Extract non-empty text paragraphs from a PDF file.

    Args:
        path: Path to the PDF file

    Returns:
        List of text passages, one per non-empty line/paragraph
    """
    from pypdf import PdfReader
    reader = PdfReader(path)
    passages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.split("\n"):
            line = line.strip()
            if len(line) >= 20:
                passages.append(line)
    logger.info(f"Extracted {len(passages)} passages from PDF: {path}")
    return passages


def extract_from_docx(path: str) -> List[str]:
    """Extract non-empty paragraphs from a Word document.

    Args:
        path: Path to the .docx file

    Returns:
        List of paragraph texts
    """
    import docx
    doc = docx.Document(path)
    passages = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) >= 20]
    logger.info(f"Extracted {len(passages)} passages from DOCX: {path}")
    return passages


def extract_qualitative_sources(paths: List[str]) -> List[str]:
    """Extract text passages from a list of PDF or Word source files.

    Dispatches to the appropriate extractor based on file extension.
    Unsupported extensions are logged and skipped.

    Args:
        paths: List of file paths (.pdf or .docx)

    Returns:
        Flat list of non-empty text passages across all files
    """
    all_passages: List[str] = []
    for path in paths:
        lower = path.lower()
        if lower.endswith(".pdf"):
            all_passages.extend(extract_from_pdf(path))
        elif lower.endswith(".docx") or lower.endswith(".doc"):
            all_passages.extend(extract_from_docx(path))
        else:
            logger.warning(f"Unsupported file type, skipping: {path}")
    logger.info(f"Total qualitative passages extracted from {len(paths)} file(s): {len(all_passages)}")
    return all_passages
