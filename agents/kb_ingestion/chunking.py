"""Three chunking strategies, and the LLM call that picks one per document.

Every chunk keeps its page number and the heading it sits under, so answers can cite
"KP-200 manual, p.2, 4.1 Mechanical Seal Replacement".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from agentkit.docparse import ParsedDoc
from agentkit.llm import LLM, LLMError

STRATEGIES = ("structural", "semantic", "recursive")
MAX_CHARS = 1400
_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


@dataclass
class Chunk:
    text: str
    page: int | None
    heading: str


def sections(doc: ParsedDoc) -> list[tuple[str, int | None, str]]:
    """Split the document at headings: (heading, page where it starts, body)."""
    out: list[tuple[str, int | None, str]] = []
    heading, page, body = doc.title, doc.pages[0][0], []
    for page_no, text in doc.pages:
        for line in text.splitlines():
            m = _HEADING.match(line.strip())
            if m:
                if body:
                    out.append((heading, page, "\n".join(body).strip()))
                heading, page, body = m.group(1).strip(), page_no, []
            else:
                body.append(line)
    if body:
        out.append((heading, page, "\n".join(body).strip()))
    return [(h, p, b) for h, p, b in out if b]


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paras:
        while len(para) > size:  # a single huge paragraph
            pieces.append(para[:size])
            para = para[size - overlap:]
        if len(current) + len(para) + 2 > size and current:
            pieces.append(current)
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces


def chunk_structural(doc: ParsedDoc) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, page, body in sections(doc):
        for piece in _split_long(body, MAX_CHARS, 0):
            chunks.append(Chunk(piece, page, heading))
    return _merge_tiny(chunks)


def chunk_recursive(doc: ParsedDoc, size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, text in doc.pages:
        heading = doc.title
        for piece in _split_long(text, size, overlap):
            m = next((_HEADING.match(l.strip()) for l in piece.splitlines() if _HEADING.match(l.strip())), None)
            heading = m.group(1).strip() if m else heading
            chunks.append(Chunk(piece, page_no, heading))
    return chunks


def chunk_semantic(doc: ParsedDoc, embed_fn, threshold: float = 0.45) -> list[Chunk]:
    """Start a new chunk where consecutive sentences stop being similar."""
    chunks: list[Chunk] = []
    for heading, page, body in sections(doc):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", body) if s.strip()]
        if len(sentences) <= 2:
            chunks.append(Chunk(body, page, heading))
            continue
        vecs = embed_fn(sentences)
        current = [sentences[0]]
        for i in range(1, len(sentences)):
            sim = float(np.dot(vecs[i - 1], vecs[i]))
            if sim < threshold or sum(len(s) for s in current) > MAX_CHARS:
                chunks.append(Chunk(" ".join(current), page, heading))
                current = []
            current.append(sentences[i])
        if current:
            chunks.append(Chunk(" ".join(current), page, heading))
    return _merge_tiny(chunks)


def _merge_tiny(chunks: list[Chunk], min_chars: int = 120) -> list[Chunk]:
    merged: list[Chunk] = []
    for c in chunks:
        if merged and len(c.text) < min_chars and merged[-1].heading == c.heading:
            merged[-1].text += "\n" + c.text
        else:
            merged.append(c)
    return merged


def doc_stats(doc: ParsedDoc) -> dict:
    secs = sections(doc)
    lengths = [len(b) for _h, _p, b in secs] or [0]
    return {"pages": len(doc.pages), "headings": len(secs), "avg_section_chars": int(sum(lengths) / len(lengths)),
            "tables": doc.tables, "chars": len(doc.markdown), "ocr": doc.ocr_used}


def rule_strategy(stats: dict) -> str:
    if stats["headings"] >= 3 and stats["avg_section_chars"] <= 2500:
        return "structural"
    if stats["headings"] < 2 and stats["chars"] > 3000:
        return "semantic"
    return "recursive"


def choose_strategy(llm: LLM, doc: ParsedDoc) -> tuple[str, str]:
    """Let the model pick the chunking strategy from the document's shape; fall back to rules."""
    stats = doc_stats(doc)
    sample = doc.markdown[:1500]
    prompt = (
        "You choose how to split a document into chunks for a search index.\n"
        "- structural: the document has clear headings; keep each section together.\n"
        "- semantic: long running prose without useful headings; split where the topic changes.\n"
        "- recursive: short or irregular text (OCR output, notes, tables); split by size.\n"
        f"Document stats: {stats}\nStart of the document:\n---\n{sample}\n---\n"
        'Answer with JSON: {"strategy": "structural|semantic|recursive", "reason": "<one short sentence>"}'
    )
    try:
        out = llm.json([{"role": "user", "content": prompt}])
        strategy = str(out.get("strategy", "")).lower().strip()
        if strategy in STRATEGIES:
            return strategy, str(out.get("reason", "")).strip()[:200]
    except LLMError:
        pass
    strategy = rule_strategy(stats)
    return strategy, "chosen by rule (the model's choice was unavailable)"


def make_chunks(strategy: str, doc: ParsedDoc, embed_fn) -> list[Chunk]:
    if strategy == "structural":
        return chunk_structural(doc)
    if strategy == "semantic":
        return chunk_semantic(doc, embed_fn)
    return chunk_recursive(doc)
