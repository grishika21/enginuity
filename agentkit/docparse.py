"""Document parsing with Docling, with OCR only where it is needed.

A PDF whose pages carry no text layer (a scan) is detected per file and converted with Tesseract
OCR; digital PDFs never go through OCR. The result keeps page numbers, which the knowledge base
cites and the PageIndex tree uses for page ranges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .contract import AgentError

log = logging.getLogger("agentkit.docparse")

SUPPORTED = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".xlsx", ".csv"}
SCANNED_RATIO = 0.2  # OCR when at least this share of pages has no text layer


@dataclass
class ParsedDoc:
    title: str
    pages: list[tuple[int | None, str]]  # (page number or None, markdown)
    ocr_used: bool = False
    tables: int = 0
    pictures: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def markdown(self) -> str:
        return "\n\n".join(text for _page, text in self.pages)


def scanned_share(path: Path) -> float:
    """Share of PDF pages with (almost) no extractable text."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        total = len(pdf)
        if total == 0:
            return 0.0
        empty = 0
        for i in range(total):
            page = pdf[i]
            text = page.get_textpage().get_text_range()
            if len(text.strip()) < 20:
                empty += 1
        return empty / total
    finally:
        pdf.close()


@lru_cache(maxsize=2)
def _converter(ocr: bool):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    if ocr:
        opts.ocr_options = TesseractCliOcrOptions(lang=["eng"], force_full_page_ocr=True)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def parse(path: Path, display_name: str | None = None) -> ParsedDoc:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise AgentError(f"I can't read {suffix or 'that'} files. Supported: {', '.join(sorted(SUPPORTED))}.")
    ocr = False
    if suffix == ".pdf":
        try:
            ocr = scanned_share(path) >= SCANNED_RATIO
        except Exception as exc:
            raise AgentError("That PDF could not be opened. Is it damaged or password-protected?") from exc
    try:
        result = _converter(ocr).convert(str(path))
    except Exception as exc:
        log.warning("docling failed on %s: %s", path.name, exc)
        raise AgentError("The document could not be parsed.") from exc
    doc = result.document
    pages: list[tuple[int | None, str]] = []
    n_pages = doc.num_pages() if hasattr(doc, "num_pages") else 0
    if n_pages:
        for page_no in range(1, n_pages + 1):
            text = doc.export_to_markdown(page_no=page_no).strip()
            if text:
                pages.append((page_no, text))
    else:
        text = doc.export_to_markdown().strip()
        if text:
            pages.append((None, text))
    if not pages:
        raise AgentError("No readable text was found in the document.")
    name = display_name or path.name
    title = _guess_title(pages[0][1]) or Path(name).stem.replace("_", " ")
    return ParsedDoc(title=title, pages=pages, ocr_used=ocr, tables=len(getattr(doc, "tables", []) or []),
                     pictures=len(getattr(doc, "pictures", []) or []))


def _guess_title(first_page: str) -> str:
    for line in first_page.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if len(line) > 8:
            return line[:100]
    return ""
