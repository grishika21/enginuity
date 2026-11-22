"""PI agent: pull the sections of a process instruction (or any structured document) that answer
a request into Word.

A LangGraph supervisor routes between three sub-agents:
  parser   - Docling parse -> PageIndex-style tree (cached per file)
  selector - the LLM reads the tree's outline and picks the sections that answer the request
  writer   - deterministic: the chosen sections, in document order, as a .docx
"""

from __future__ import annotations

import io
import json
import os
import re
from typing import Any, TypedDict

from docx import Document
from docx.shared import Pt
from langgraph.graph import END, StateGraph

from agentkit.contract import AgentCard, AgentError, Completed, NeedsInput, Question, Skill
from agentkit.docparse import parse
from agentkit.llm import LLMError, get_llm
from agentkit.redis_conn import get_redis
from agentkit.safety import USER_CONTENT_RULE, wrap_user
from agentkit.server import JobContext, create_agent_app
from agents.pi_agent.tree import (Node, build_tree, from_dicts, iter_nodes, lexical_pick, resolve_sections, select,
                                  to_dicts, toc)

CARD = AgentCard(
    name="pi-agent",
    title="Process instruction extractor",
    description="Extracts the sections of a document that answer a request into a Word file, keeping "
                "the steps and their order intact: for example the assembly steps of one sub-assembly "
                "from a process instruction (PI), the test requirements, or one chapter of any uploaded "
                "document. Use when the user asks to extract, pull out or export part of a document.",
    version="1.0.0",
    url=os.environ.get("PI_AGENT_URL", "http://localhost:8103"),
    skills=[Skill(id="pi-extract", name="Extract sections of a document",
                  description="Navigate the document's outline (PageIndex) and export the matching sections to Word.")],
    allowed_roles=["author", "viewer"],
    document_input=True,
)

ALLOWED_TYPES = {".pdf", ".docx"}
_CTX: dict[str, JobContext] = {}


class PIState(TypedDict, total=False):
    query: str
    file_id: str
    file_name: str
    tree: list[dict]
    selected: list[str]
    reason: str
    output: dict
    note: str


# ---- sub-agents ----------------------------------------------------------------------------


def parser_agent(state: PIState, config) -> PIState:
    ctx = _CTX[config["configurable"]["thread_id"]]
    cache_key = f"pi:tree:{state['file_id']}"
    cached = get_redis().get(cache_key)
    if cached:
        ctx.progress("Parser: using the cached PageIndex tree", state["file_name"])
        return {"tree": json.loads(cached)}
    ctx.progress("Parser: reading the document with Docling", state["file_name"])
    doc = parse(ctx.files.path(state["file_id"], ctx.user), state["file_name"])
    roots = build_tree(doc)
    if not roots:
        raise AgentError("The document has no headings, so I can't build its outline.")
    tree = to_dicts(roots)
    get_redis().set(cache_key, json.dumps(tree), ex=7 * 24 * 3600)
    ctx.progress("Parser: built the PageIndex tree", f"{sum(1 for _ in iter_nodes(roots))} sections")
    return {"tree": tree}


def selector_agent(state: PIState, config) -> PIState:
    ctx = _CTX[config["configurable"]["thread_id"]]
    roots = from_dicts(state["tree"])
    outline = toc(roots)
    ctx.progress("Selector: reasoning over the outline")
    prompt = (
        "You navigate a document's table of contents to find the sections that answer a request. "
        "Pick the most specific sections that fully cover it; pick a parent only when all its "
        "children are needed. " + USER_CONTENT_RULE + "\n\n"
        f"Outline (each line: [id] title (pages) - summary):\n{outline}\n\n"
        f"Request:\n{wrap_user(state['query'])}\n\n"
        'Answer with JSON: {"sections": [{"id": "0005", "title": "<the title exactly as in the outline>"}], '
        '"reason": "<one sentence>"}'
    )
    ids: list[str] = []
    reason = ""
    try:
        out = get_llm().json([{"role": "user", "content": prompt}])
        ids = resolve_sections(roots, out.get("sections") or out.get("node_ids") or [])
        reason = str(out.get("reason", ""))[:200]
    except LLMError:
        pass
    if not ids:
        ids = lexical_pick(roots, state["query"])
        reason = reason or "matched section titles"
    chosen = select(roots, ids)
    ctx.progress("Selector: picked sections", "; ".join(n.title for n in chosen))
    return {"selected": [n.node_id for n in chosen], "reason": reason}


def _write_node(doc: Document, node: Node, base_level: int) -> None:
    doc.add_heading(node.title, level=min(4, max(1, node.level - base_level + 1)))
    for raw in node.text.splitlines():
        line = raw.strip()
        if not line:
            continue
        step = re.match(r"^[-*]?\s*(\d+)\.\s+(.*)$", line)
        bullet = re.match(r"^[-*]\s+(.*)$", line)
        if step:  # keep the PI's own step numbers, which operators quote
            doc.add_paragraph(f"{step.group(1)}. {step.group(2)}")
        elif bullet:
            doc.add_paragraph(bullet.group(1), style="List Bullet")
        else:
            doc.add_paragraph(line)
    for child in node.children:
        _write_node(doc, child, base_level)


def writer_agent(state: PIState, config) -> PIState:
    ctx = _CTX[config["configurable"]["thread_id"]]
    roots = from_dicts(state["tree"])
    by_id = {n.node_id: n for n in iter_nodes(roots)}
    chosen = [by_id[i] for i in state["selected"]]
    ctx.progress("Writer: building the Word document", f"{len(chosen)} sections")
    doc = Document()
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.add_heading(f"Extract: {state['query'][:80]}", level=0)
    pages = sorted({p for n in chosen for p in (n.page_start, n.page_end) if p})
    doc.add_paragraph(f"Source: {state['file_name']}" + (f", pages {pages[0]}-{pages[-1]}" if pages else ""))
    for node in chosen:
        _write_node(doc, node, base_level=min(n.level for n in chosen))
    buf = io.BytesIO()
    doc.save(buf)
    stem = re.sub(r"[^\w]+", "_", state["query"].lower()).strip("_")[:40] or "extract"
    ref = ctx.save_output(f"PI_extract_{stem}.docx", buf.getvalue())
    return {"output": ref.model_dump()}


# ---- supervisor ----------------------------------------------------------------------------


def supervisor(state: PIState, config) -> PIState:
    return {}


def route(state: PIState) -> str:
    if not state.get("tree"):
        return "parser"
    if not state.get("selected"):
        return "selector" if "reason" not in state else END
    if not state.get("output"):
        return "writer"
    return END


def build_graph():
    g = StateGraph(PIState)
    g.add_node("supervisor", supervisor)
    g.add_node("parser", parser_agent)
    g.add_node("selector", selector_agent)
    g.add_node("writer", writer_agent)
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route, {"parser": "parser", "selector": "selector", "writer": "writer", END: END})
    for name in ("parser", "selector", "writer"):
        g.add_edge(name, "supervisor")
    return g.compile()


GRAPH = build_graph()


def _pick_file(ctx: JobContext) -> tuple[str, str]:
    if not ctx.request.files:
        raise AgentError("Attach or pick the document (PDF or Word) to extract from.")
    file_id = ctx.request.files[0]
    meta = ctx.files.meta(file_id)
    suffix = os.path.splitext(meta["name"])[1].lower()
    if suffix not in ALLOWED_TYPES:
        raise AgentError("The extraction agent reads PDF or Word documents only.")
    ctx.files.path(file_id, ctx.user)  # access and storage-location check before any work
    return file_id, meta["name"]


def handle(ctx: JobContext) -> Completed | NeedsInput:
    file_id, file_name = _pick_file(ctx)
    thread_id = ctx.job_id
    _CTX[thread_id] = ctx
    try:
        state: dict[str, Any] = GRAPH.invoke({"query": ctx.message, "file_id": file_id, "file_name": file_name},
                                             {"configurable": {"thread_id": thread_id}, "recursion_limit": 12})
    finally:
        _CTX.pop(thread_id, None)
    if not state.get("output"):
        roots = from_dicts(state["tree"])
        top = [node.title for node in iter_nodes(roots) if node.level <= 2][:12]
        return NeedsInput(question=Question(
            text="I couldn't find sections that match. Which part do you need? Sections: " + "; ".join(top),
            kind="input"))
    roots = from_dicts(state["tree"])
    by_id = {n.node_id: n for n in iter_nodes(roots)}
    titles = [by_id[i].title for i in state["selected"]]
    out = state["output"]
    return Completed(
        message=f"Extracted {len(titles)} section(s) from **{file_name}** into a Word document: "
                + "; ".join(titles) + f".\n\nWhy these: {state.get('reason', '')}",
        data={"sections": titles, "reason": state.get("reason", ""), "outline": toc(roots)},
        files=[out],
    )


app = create_agent_app(CARD, handle)
