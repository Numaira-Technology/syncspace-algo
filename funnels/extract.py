"""
Word document paragraph extractor.

Reads a .docx file and returns a dict of non-empty paragraph texts,
indexed in document order.

Usage:
    from funnels.extract import read_docx
    paragraphs = read_docx("/path/to/document.docx")

Input:
    word_file: str — path to a .docx file

Output:
    Dict[int, str] — {0: "First paragraph text", 1: "Second paragraph text", ...}
    Empty paragraphs (whitespace-only) are skipped.
"""

import docx
import logging

logger = logging.getLogger(__name__)


def read_docx(word_file: str):
    """Read a Word document and return an ordered dict of paragraph texts."""
    logger.info(f"Reading Word document: {word_file}")
    doc = docx.Document(word_file)

    sentences = {}
    idx = 0
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            sentences[idx] = text
            idx += 1

    logger.info(f"Extracted {len(sentences)} paragraphs from {word_file}")
    return sentences
