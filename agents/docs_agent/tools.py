"""Deterministic tools: everything that must be exact is done in code, not by the model.

Section layout, step numbering, part-number checks against the catalogue, the parts table,
impact analysis over the document graph, the approval check and the publish with rollback.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path

from docx import Document
from docx.shared import Pt

from agentkit.approval import is_explicit_yes  # noqa: F401  (re-exported for the graph)
from agentkit.config import get_settings
from agentkit.contract import AgentError
from agentkit.sandbox import valid_sandbox

PART_RE = re.compile(r"\b([A-Z]{2}-\d{2,3}-\d{3}[A-Z]?)\b")
_HEADING = re.compile(r"^#{1,4}\s+(.+?)\s*$")
_STEP = re.compile(r"^\s*(?:[-*]|\d+[.)]|step\s+\d+:?)\s+(.*)$", re.I)

NOT_COVERED = "Not covered by the sources."


# ---- catalogue -----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def parts_catalogue() -> dict[str, dict]:
    path = get_settings().data_dir / "parts.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["part_number"]: row for row in csv.DictReader(fh)}


def equipment_models() -> list[str]:
    return sorted({row["equipment"] for row in parts_catalogue().values()})


def check_equipment(value: str | None) -> str | None:
    """Normalise an equipment reference ("kp200", "KP 200") to a catalogue model, or None."""
    if not value:
        return None
    squashed = re.sub(r"[^a-z0-9]", "", value.lower())
    for model in equipment_models():
        if re.sub(r"[^a-z0-9]", "", model.lower()) in squashed:
            return model
    return None


CHECKS = {"equipment": check_equipment}


# ---- layout --------------------------------------------------------------------------------


def split_sections(markdown: str, wanted: list[str]) -> dict[str, str]:
    """Map the model's output onto exactly the skill's sections, in order; unknown headings fold
    into the section before them and missing sections are marked as not covered."""
    found: dict[str, list[str]] = {name: [] for name in wanted}
    current: str | None = None
    lookup = {re.sub(r"[^a-z]", "", w.lower()): w for w in wanted}
    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            key = re.sub(r"[^a-z]", "", re.sub(r"^\d+(\.\d+)*\s*", "", m.group(1)).lower())
            match = lookup.get(key) or next((w for k, w in lookup.items() if k and (k in key or key in k)), None)
            if match:
                current = match
                continue
        if current:
            found[current].append(line)
    return {name: "\n".join(lines).strip() or NOT_COVERED for name, lines in found.items()}


def number_steps(text: str) -> str:
    """Renumber step lines 1..n in code, whatever numbering the model used."""
    out, n = [], 0
    for line in text.splitlines():
        m = _STEP.match(line)
        if m and m.group(1).strip():
            n += 1
            out.append(f"{n}. {m.group(1).strip()}")
        elif line.strip():
            out.append(line.strip())
    return "\n".join(out) if n else text


def check_parts(text: str, equipment: str) -> tuple[str, list[str], list[str]]:
    """Flag part numbers that are not in the catalogue for this equipment.

    Returns (text with unknown numbers flagged, verified parts, unknown parts)."""
    catalogue = parts_catalogue()
    verified, unknown = [], []

    def flag(m: re.Match) -> str:
        pn = m.group(1)
        row = catalogue.get(pn)
        if row and row["equipment"] == equipment:
            if pn not in verified:
                verified.append(pn)
            return pn
        if pn not in unknown:
            unknown.append(pn)
        return f"{pn} [unverified part number]"

    return PART_RE.sub(flag, text), verified, unknown


def _part_name(description: str) -> str:
    return re.split(r"[,(]", description)[0].strip().lower()


def annotate_part_names(text: str, equipment: str) -> str:
    """Add the part number after the first mention of a catalogue part name that has none.

    Only names of two or more words count, and a name shared by several parts (a part and its
    superseding version) is skipped: the code never guesses which one is meant."""
    rows = [r for r in parts_catalogue().values() if r["equipment"] == equipment]
    names: dict[str, list[str]] = {}
    for row in rows:
        names.setdefault(_part_name(row["description"]), []).append(row["part_number"])
    for name, numbers in sorted(names.items(), key=lambda kv: -len(kv[0])):
        if len(numbers) != 1 or len(name.split()) < 2 or numbers[0] in text:
            continue
        pattern = re.compile(rf"\b({re.escape(name)})\b(?!\s*\()", re.I)
        text = pattern.sub(lambda m: f"{m.group(1)} ({numbers[0]})", text, count=1)
    return text


def parts_table(parts: list[str]) -> str:
    if not parts:
        return NOT_COVERED
    catalogue = parts_catalogue()
    rows = ["| Part number | Description | Qty |", "|---|---|---|"]
    rows += [f"| {pn} | {catalogue[pn]['description']} | {catalogue[pn]['qty_per_unit']} |" for pn in parts]
    return "\n".join(rows)


GENERATORS = {"parts_table": parts_table}


def render(title: str, header: dict[str, str], sections: dict[str, str]) -> str:
    meta = "  \n".join(f"**{k}:** {v}" for k, v in header.items())
    body = "\n\n".join(f"## {i}. {name}\n\n{text}" for i, (name, text) in enumerate(sections.items(), 1))
    return f"# {title}\n\n{meta}\n\n{body}\n"


def to_docx(markdown: str) -> bytes:
    doc = Document()
    doc.styles["Normal"].font.size = Pt(10.5)
    table_rows: list[list[str]] = []

    def flush_table():
        if not table_rows:
            return
        table = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        table.style = "Light Grid Accent 1"
        for r, row in enumerate(table_rows):
            for c, cell in enumerate(row[: len(table_rows[0])]):
                table.cell(r, c).text = cell
        table_rows.clear()

    for line in markdown.splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                table_rows.append(cells)
            continue
        flush_table()
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
        if not s:
            continue
        if s.startswith("# "):
            doc.add_heading(clean[2:], level=0)
        elif s.startswith("## "):
            doc.add_heading(clean[3:], level=1)
        elif s.startswith("### "):
            doc.add_heading(clean[4:], level=2)
        elif re.match(r"^[-*]\s+", s):
            doc.add_paragraph(re.sub(r"^[-*]\s+", "", clean), style="List Bullet")
        else:
            doc.add_paragraph(clean)
    flush_table()
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---- library and graph ---------------------------------------------------------------------


def _sandbox_root(sandbox: str) -> Path:
    if not valid_sandbox(sandbox):
        raise ValueError("invalid sandbox id")
    return get_settings().storage_dir / "sandboxes" / sandbox


def library_dir(sandbox: str = "") -> Path:
    """The published-documents library. Each sandbox starts from its own copy of the sample
    library, so one visitor's publications never show up in another's impact analysis."""
    lib = _sandbox_root(sandbox) / "library" if sandbox else get_settings().storage_dir / "library"
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(get_settings().data_dir / "library", lib)
    return lib


def drop_library(sandbox: str) -> bool:
    root = _sandbox_root(sandbox)
    if not root.exists():
        return False
    shutil.rmtree(root)
    return True


def load_graph(sandbox: str = "") -> dict:
    return json.loads((library_dir(sandbox) / "graph.json").read_text(encoding="utf-8"))


def impact(parts: list[str], equipment: str | None, sandbox: str = "") -> list[dict]:
    """Published documents that use any of the same parts: they may need review."""
    out = []
    for doc_id, doc in load_graph(sandbox)["documents"].items():
        shared = sorted(set(parts) & set(doc.get("parts", [])))
        if shared:
            out.append({"id": doc_id, "title": doc["title"], "shared_parts": shared})
    return sorted(out, key=lambda d: (-len(d["shared_parts"]), d["id"]))


_APPROVE = {"approve", "approved", "looks good", "lgtm", "ok", "okay", "yes", "good", "fine", "continue", "approve it"}


def is_approval(reply: str) -> bool:
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", (reply or "").lower())).strip()
    return normalized in _APPROVE


DOC_PREFIX = {"KP": "TP-1", "KC": "TP-2"}  # other equipment, and documents about no equipment: TP-9xx


def next_doc_id(equipment: str | None, sandbox: str = "") -> str:
    graph = load_graph(sandbox)
    prefix = DOC_PREFIX.get((equipment or "")[:2], "TP-9")
    used = [int(d[3:]) for d in graph["documents"] if d.startswith(prefix)]
    return f"TP-{max(used, default=int(prefix[3:]) * 100) + 1}"


def publish(title: str, equipment: str | None, parts: list[str], markdown: str, sandbox: str = "") -> str:
    """Write the document to the library, then update the graph. If the graph update fails, the
    written file is removed again, so the library never holds a document the graph doesn't know."""
    lib = library_dir(sandbox)
    doc_id = next_doc_id(equipment, sandbox)
    doc_path = lib / f"{doc_id}.md"
    front = (f"---\nid: {doc_id}\ntitle: {title}\nequipment: {equipment or ''}\n"
             f"parts: [{', '.join(parts)}]\n---\n\n")
    doc_path.write_text(front + markdown, encoding="utf-8")
    try:
        if os.environ.get("DOCS_SIMULATE_GRAPH_FAILURE") == "1":
            raise OSError("simulated graph write failure")
        graph = load_graph(sandbox)
        graph["documents"][doc_id] = {"title": title, "equipment": equipment or "", "parts": parts,
                                      "file": doc_path.name}
        tmp = lib / "graph.json.tmp"
        tmp.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        os.replace(tmp, lib / "graph.json")
    except Exception as exc:
        doc_path.unlink(missing_ok=True)
        raise AgentError("Publishing failed while updating the document graph, so the library was "
                         "restored and nothing was published. The draft is kept.") from exc
    return doc_id
