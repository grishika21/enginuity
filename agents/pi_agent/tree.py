"""PageIndex-style document tree.

Instead of cutting a long instruction into embedded chunks, the document becomes a table of
contents: one node per section with its page range, a short summary and its own text. The model
reasons over that outline to pick sections, so the order and nesting of steps stay intact.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from agentkit.docparse import ParsedDoc

_HEADING = re.compile(r"^#{1,6}\s+(.*)$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")


@dataclass
class Node:
    node_id: str
    title: str
    level: int
    page_start: int | None
    page_end: int | None
    text: str = ""
    summary: str = ""
    children: list["Node"] = field(default_factory=list)

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def full_text(self) -> str:
        parts = [self.text] if self.text else []
        for child in self.children:
            parts.append(f"{child.title}\n{child.full_text()}")
        return "\n\n".join(p for p in parts if p)


def _level(title: str, hashes: int, default_level: int) -> int:
    m = _NUMBERED.match(title)
    if m:
        return m.group(1).count(".") + 1
    return default_level if hashes <= 2 else hashes - 1


def _summary(node: Node) -> str:
    text = re.sub(r"^[-*]\s*\d+\.\s*", "", node.text.strip(), flags=re.M)
    first = re.split(r"(?<=[.!?])\s|\n", text, maxsplit=1)[0] if text else ""
    if not first and node.children:
        first = "Covers: " + "; ".join(c.title for c in node.children[:5])
    return first[:160]


def build_tree(doc: ParsedDoc) -> list[Node]:
    roots: list[Node] = []
    stack: list[Node] = []
    counter = 0
    current: Node | None = None
    body: list[str] = []

    def flush():
        if current is not None:
            current.text = "\n".join(body).strip()

    for page_no, text in doc.pages:
        for line in text.splitlines():
            m = _HEADING.match(line.strip())
            if not m:
                if current is not None:
                    body.append(line)
                    current.page_end = page_no
                continue
            flush()
            body = []
            title = m.group(1).strip()
            hashes = len(line.strip()) - len(line.strip().lstrip("#"))
            level = _level(title, hashes, default_level=1)
            counter += 1
            node = Node(f"{counter:04d}", title, level, page_no, page_no)
            while stack and stack[-1].level >= level:
                stack.pop()
            (stack[-1].children if stack else roots).append(node)
            stack.append(node)
            current = node
    flush()
    for root in roots:
        for node in root.walk():
            node.summary = _summary(node)
            for child in node.walk():
                if child.page_end and (node.page_end is None or child.page_end > node.page_end):
                    node.page_end = child.page_end
    return roots


def iter_nodes(roots: list[Node]):
    for root in roots:
        yield from root.walk()


def toc(roots: list[Node]) -> str:
    lines = []
    for node in iter_nodes(roots):
        pages = f"p.{node.page_start}" if node.page_start == node.page_end else f"p.{node.page_start}-{node.page_end}"
        lines.append(f"{'  ' * (node.level - 1)}[{node.node_id}] {node.title} ({pages}) - {node.summary}")
    return "\n".join(lines)


def select(roots: list[Node], ids: list[str]) -> list[Node]:
    """Return the chosen nodes in document order, dropping any already inside a chosen parent."""
    wanted = set(ids)
    chosen: list[Node] = []

    def visit(node: Node, covered: bool):
        take = node.node_id in wanted and not covered
        if take:
            chosen.append(node)
        for child in node.children:
            visit(child, covered or take)

    for root in roots:
        visit(root, False)
    return chosen


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def resolve_sections(roots: list[Node], picked: list) -> list[str]:
    """Turn the model's picks into node ids. A title wins over an id, because models copy titles
    reliably but can be off by one on ids."""
    by_title = {_norm(n.title): n.node_id for n in iter_nodes(roots)}
    valid = set(by_title.values())

    def by_text(text: str) -> str | None:
        # Models often copy the whole outline line ("5.3 Impeller (p.3) - Fit the..."): take the
        # longest section title the text starts with.
        t = _norm(text)
        hits = [title for title in by_title if t == title or t.startswith(title + " ")]
        return by_title[max(hits, key=len)] if hits else None

    ids: list[str] = []
    for item in picked:
        node_id = None
        if isinstance(item, dict):
            node_id = by_text(str(item.get("title", "")))
            if node_id is None and str(item.get("id", "")).zfill(4) in valid:
                node_id = str(item["id"]).zfill(4)
        elif str(item).zfill(4) in valid:
            node_id = str(item).zfill(4)
        if node_id and node_id not in ids:
            ids.append(node_id)
    return ids


def lexical_pick(roots: list[Node], query: str, limit: int = 3) -> list[str]:
    """Fallback when the model's pick is unusable: overlap between query words and titles."""
    words = {w for w in re.findall(r"[a-z]{3,}", query.lower())} - {"the", "and", "for", "steps", "extract"}
    scored = []
    for node in iter_nodes(roots):
        title_words = set(re.findall(r"[a-z]{3,}", node.title.lower()))
        score = len(words & title_words)
        if score:
            scored.append((score, node.level, node.node_id))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [node_id for _s, _l, node_id in scored[:limit]]


def to_dicts(roots: list[Node]) -> list[dict]:
    return [asdict(r) for r in roots]


def from_dicts(items: list[dict]) -> list[Node]:
    def make(d: dict) -> Node:
        return Node(**{**d, "children": [make(c) for c in d.get("children", [])]})

    return [make(d) for d in items]
