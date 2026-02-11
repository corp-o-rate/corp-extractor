"""
PDF parser plugins for extracting text from PDF files.

Built-in parsers:
- pypdf_parser: Default PDF parser using PyMuPDF with optional OCR
- glm_ocr_parser: GLM-OCR 0.9B VLM for high-quality OCR (scans, tables, formulas)
"""

from .pypdf import PyPDFParserPlugin
from .glm_ocr import GlmOcrParserPlugin

__all__ = ["PyPDFParserPlugin", "GlmOcrParserPlugin"]
