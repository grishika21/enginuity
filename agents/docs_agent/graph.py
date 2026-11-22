"""The docs agent's workflow as a LangGraph graph with human-in-the-loop interrupts.

pick_skill -> extract_inputs -> ask_input* -> gather_sources -> write_draft -> review <-> revise
           -> analyse_impact -> confirm_publish -> publish

Every pause is a LangGraph `interrupt()`; state is checkpointed to SQLite after every step, so a
paused run survives a restart and resumes when the user answers.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agentkit.config import get_settings
from agentkit.contract import AgentError
from agentkit.llm import LLMError, get_llm
from agentkit.safety import USER_CONTENT_RULE, wrap_user
from agentkit.sandbox import sandbox_of
from agentkit.server import JobContext
from agents.docs_agent import tools
from agents.docs_agent.skills_loader import SkillSpec, load_skills

log = logging.getLogger("docs_agent.graph")

SKILLS: dict[str, SkillSpec] = load_skills()
CTX: dict[str, JobContext] = {}


def reload_skills() -> dict[str, SkillSpec]:
    global SKILLS
    SKILLS = load_skills()
    return SKILLS


class DocState(TypedDict, total=False):
    request: str
    files: list[str]
    skill: str
    inputs: dict[str, str | None]
    input_error: str
    sources: list[dict]
    title: str
    draft: str
    version: int
    verified_parts: list[str]
    open_items: list[str]
    draft_file: dict
    feedback: str
    impact: list[dict]
    status: str
    doc_id: str


def _ctx(config) -> JobContext:
    return CTX[config["configurable"]["thread_id"]]


def _skill(state: DocState) -> SkillSpec:
    spec = SKILLS.get(state["skill"])
    if spec is None:
        raise AgentError(f"The '{state['skill']}' document type is no longer available.")
    return spec


# ---- nodes ---------------------------------------------------------------------------------


def pick_skill(state: DocState, config) -> DocState:
    ctx = _ctx(config)
    forced = ctx.parameters.get("skill")
    if forced in SKILLS:
        name = forced
    else:
        catalog = "\n".join(f"- {s.name}: {s.description}" for s in SKILLS.values())
        prompt = (
            "Pick the document type (skill) that fits the request, using only the descriptions. "
            "If none fits, answer \"none\". " + USER_CONTENT_RULE + f"\n\nSkills:\n{catalog}\n\n"
            f"Request:\n{wrap_user(state['request'])}\n\n"
            'Answer with JSON: {"skill": "<skill name or none>"}'
        )
        try:
            name = str(get_llm().json([{"role": "user", "content": prompt}]).get("skill", "none")).strip()
        except LLMError:
            name = "none"
        if name not in SKILLS:
            words = set(state["request"].lower().split())
            name = next((s.name for s in SKILLS.values() if set(s.title.lower().split()) <= words), "none")
    if name == "none":
        return {"status": "no_skill"}
    ctx.progress(f"Skill: {SKILLS[name].title}", f"read skills/{name}/SKILL.md")
    return {"skill": name}


def extract_inputs(state: DocState, config) -> DocState:
    ctx = _ctx(config)
    spec = _skill(state)
    keys = {i.key: i.ask for i in spec.inputs}
    prompt = (
        "Extract these fields from the request. Use null for anything the request does not state; "
        "never guess. " + USER_CONTENT_RULE + f"\n\nFields: {keys}\n\nRequest:\n{wrap_user(state['request'])}\n\n"
        "Answer with a JSON object with exactly those keys."
    )
    try:
        raw = get_llm().json([{"role": "user", "content": prompt}])
    except LLMError:
        raw = {}
    inputs: dict[str, str | None] = {}
    for item in spec.inputs:
        value = raw.get(item.key)
        value = str(value).strip() if value not in (None, "", "null") else None
        if value and item.check:
            value = tools.CHECKS[item.check](value)
        if not value and item.check == "equipment":  # a model code in the text is enough
            value = tools.check_equipment(state["request"])
        inputs[item.key] = value or item.default
    ctx.progress("Collected inputs", ", ".join(f"{k}: {v or 'missing'}" for k, v in inputs.items()))
    return {"inputs": inputs}


def _missing(state: DocState) -> str | None:
    return next((i.key for i in _skill(state).inputs if not (state.get("inputs") or {}).get(i.key)), None)


def ask_input(state: DocState, config) -> DocState:
    key = _missing(state)
    if key is None:
        return {}
    spec = next(i for i in _skill(state).inputs if i.key == key)
    options = tools.equipment_models() if spec.check == "equipment" else []
    text = (state.get("input_error", "") + " " + spec.ask).strip()
    reply = str(interrupt({"text": text, "options": options, "kind": "input"})).strip()
    value = tools.CHECKS[spec.check](reply) if spec.check else reply
    inputs = dict(state.get("inputs") or {})
    if not value:
        return {"input_error": f"\"{reply[:40]}\" is not a model I have data for."}
    inputs[key] = value
    return {"inputs": inputs, "input_error": ""}


def gather_sources(state: DocState, config) -> DocState:
    ctx = _ctx(config)
    spec = _skill(state)
    inputs = state["inputs"]
    query = spec.source_query.format(**{k: v or "" for k, v in inputs.items()})
    parameters = {"mode": "chunks", "top_k": 8, "equipment": inputs.get("equipment") or ""}
    attached = state.get("files") or []
    if spec.source_scope == "document" and attached:  # the whole attached document, in reading order
        parameters["file_id"] = attached[0]
    job = ctx.call_peer(query, skill="ask", parameters=parameters)
    if job.status != "completed":
        raise AgentError(job.message or "The knowledge base could not be searched.")
    if job.data.get("not_indexed"):
        return {"status": "not_indexed"}
    chunks = job.data.get("chunks", [])
    if not chunks:
        return {"status": "no_sources"}
    chunks.sort(key=lambda c: (c.get("title", ""), c.get("order", 0)))  # reading order within each document
    sources = [{"n": n, "title": c["title"], "page": c.get("page"), "heading": c.get("heading", ""),
                "text": c["text"]} for n, c in enumerate(chunks, 1)]
    docs = sorted({s["title"] for s in sources})
    ctx.progress("Sources from the knowledge base", f"{len(sources)} passages from: {'; '.join(docs)}")
    return {"sources": sources}


def _source_block(sources: list[dict]) -> str:
    return "\n\n".join(f"[S{s['n']}] {s['title']}, p.{s['page']}, \"{s['heading']}\":\n{s['text']}" for s in sources)


def _assemble(state: DocState, raw: str) -> DocState:
    """Deterministic post-processing of the model's draft. Part numbers are checked only for
    document types about one equipment model: without a model there is no catalogue to check."""
    spec = _skill(state)
    equipment = state["inputs"].get("equipment")
    sections = tools.split_sections(raw, spec.sections)
    for name in spec.numbered_sections:
        sections[name] = tools.number_steps(sections[name])
    verified, unknown = [], []
    for name, text in sections.items():
        if name in spec.generated_sections or not equipment:
            continue
        checked, ok, bad = tools.check_parts(tools.annotate_part_names(text, equipment), equipment)
        sections[name] = checked
        verified += [p for p in ok if p not in verified]
        unknown += [p for p in bad if p not in unknown]
    for name, generator in spec.generated_sections.items():
        sections[name] = tools.GENERATORS[generator](verified)
    open_items = [f"Part number {p} is not in the {equipment} catalogue." for p in unknown]
    open_items += [f"Section \"{n}\" has no source content." for n, t in sections.items() if t == tools.NOT_COVERED]
    gaps = sum(t.count("[value not in sources]") for t in sections.values())
    if gaps:
        open_items.append(f"{gaps} value(s) are not in the sources and are marked [value not in sources].")
    inputs = state["inputs"]
    subject = (inputs.get("task") or inputs.get("system") or inputs.get("topic") or "").strip()
    titles = sorted({s["title"] for s in state["sources"]})
    if spec.source_scope == "document" and state.get("files") and len(titles) == 1:
        subject = titles[0]  # the document's own title, not the file name the user typed
    subject = subject[:1].upper() + subject[1:]
    if not equipment:
        title = f"{subject} - {spec.title}" if subject else spec.title
    elif subject.lower() in ("", "whole unit", "complete unit", "whole equipment"):
        title = f"{equipment} {spec.title}"
    else:
        title = f"{equipment} {subject} - {spec.title}"
    version = state.get("version", 0) + 1
    header = {"Document type": spec.title, **({"Equipment": equipment} if equipment else {}),
              "Version": f"draft v{version}", "Sources": "; ".join(sorted({s['title'] for s in state['sources']}))}
    markdown = tools.render(title, header, sections)
    return {"title": title, "draft": markdown, "version": version, "verified_parts": verified,
            "open_items": open_items}


def _save_draft(ctx: JobContext, update: DocState) -> DocState:
    name = f"{update['title'].split(' - ')[0].replace(' ', '_')}_draft_v{update['version']}.docx"
    ref = ctx.save_output(name, tools.to_docx(update["draft"]))
    return {**update, "draft_file": ref.model_dump()}


def draft(state: DocState, config) -> DocState:
    ctx = _ctx(config)
    spec = _skill(state)
    headings = "\n".join(f"## {s}" for s in spec.sections)
    system = ("You are a technical author. Follow the skill's authoring rules exactly.\n\n"
              f"{spec.rules}\n\n{USER_CONTENT_RULE}")
    user = (f"Sources:\n{_source_block(state['sources'])}\n\n"
            f"Inputs: {state['inputs']}\nOriginal request:\n{wrap_user(state['request'])}\n\n"
            f"Write the document in Markdown using exactly these headings, in this order:\n{headings}")
    ctx.progress("Drafting", f"{len(spec.sections)} sections from {len(state['sources'])} passages")
    raw = get_llm().chat([{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.1)
    ctx.progress("Checking the draft", "section layout, step numbering, part numbers against the catalogue")
    return _save_draft(ctx, _assemble(state, raw))


def review(state: DocState, config) -> DocState:
    items = state.get("open_items") or []
    notes = ("\n\nOpen items:\n" + "\n".join(f"- {i}" for i in items)) if items else "\n\nNo open items."
    reply = str(interrupt({
        "text": f"Draft v{state['version']} of **{state['title']}** is ready.{notes}\n\n"
                "Reply **approve** to continue, or tell me what to change.",
        "options": ["approve"], "kind": "review", "files": [state["draft_file"]],
    })).strip()
    return {"feedback": "" if tools.is_approval(reply) else reply}


def revise(state: DocState, config) -> DocState:
    ctx = _ctx(config)
    spec = _skill(state)
    ctx.progress("Revising the draft", state["feedback"][:120])
    system = ("You are a technical author revising a document. Apply the change requested, keep every "
              "heading, and use only the sources.\n\n" + spec.rules + "\n\n" + USER_CONTENT_RULE)
    user = (f"Sources:\n{_source_block(state['sources'])}\n\nCurrent document:\n{state['draft']}\n\n"
            f"Requested change:\n{wrap_user(state['feedback'])}\n\nReturn the full revised document in Markdown.")
    raw = get_llm().chat([{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.1)
    return _save_draft(ctx, _assemble(state, raw))


def impact(state: DocState, config) -> DocState:
    ctx = _ctx(config)
    affected = tools.impact(state.get("verified_parts") or [], state["inputs"].get("equipment"),
                            sandbox_of(ctx.user))
    ctx.progress("Impact analysis", f"{len(affected)} published document(s) share parts with this draft")
    return {"impact": affected}


def confirm_publish(state: DocState, config) -> DocState:
    affected = state.get("impact") or []
    if affected:
        lines = "\n".join(f"- {d['id']} {d['title']} (shares {', '.join(d['shared_parts'])})" for d in affected)
        note = f"\n\nThese published documents use the same parts and may need review:\n{lines}"
    else:
        note = "\n\nNo published document shares parts with it."
    reply = str(interrupt({
        "text": f"Ready to publish **{state['title']}** (v{state['version']}) to the library.{note}\n\n"
                "Reply **yes** to publish. Any other reply keeps it as an unpublished draft.",
        "options": ["yes", "no"], "kind": "approval",
    }))
    # Server-side check: the model never decides whether the user approved.
    return {"status": "approved" if tools.is_explicit_yes(reply) else "not_published"}


def publish(state: DocState, config) -> DocState:
    ctx = _ctx(config)
    ctx.progress("Publishing", "library, then the document graph")
    doc_id = tools.publish(state["title"], state["inputs"].get("equipment"), state.get("verified_parts") or [],
                           state["draft"].replace(f"draft v{state['version']}", f"v{state['version']}"),
                           sandbox=sandbox_of(ctx.user))
    return {"status": "published", "doc_id": doc_id}


# ---- wiring --------------------------------------------------------------------------------


def _after_pick(state: DocState) -> str:
    return END if state.get("status") == "no_skill" else "extract_inputs"


def _after_ask(state: DocState) -> str:
    return "ask_input" if _missing(state) else "gather_sources"


def _after_sources(state: DocState) -> str:
    return END if state.get("status") in ("no_sources", "not_indexed") else "write_draft"


def _after_review(state: DocState) -> str:
    return "revise" if state.get("feedback") else "analyse_impact"


def _after_confirm(state: DocState) -> str:
    return "publish" if state.get("status") == "approved" else END


def build_graph(checkpointer):
    g = StateGraph(DocState)
    for name, fn in [("pick_skill", pick_skill), ("extract_inputs", extract_inputs), ("ask_input", ask_input),
                     ("gather_sources", gather_sources), ("write_draft", draft), ("review", review), ("revise", revise),
                     ("analyse_impact", impact), ("confirm_publish", confirm_publish), ("publish", publish)]:
        g.add_node(name, fn)
    g.set_entry_point("pick_skill")
    g.add_conditional_edges("pick_skill", _after_pick, ["extract_inputs", END])
    g.add_edge("extract_inputs", "ask_input")
    g.add_conditional_edges("ask_input", _after_ask, ["ask_input", "gather_sources"])
    g.add_conditional_edges("gather_sources", _after_sources, ["write_draft", END])
    g.add_edge("write_draft", "review")
    g.add_conditional_edges("review", _after_review, ["revise", "analyse_impact"])
    g.add_edge("revise", "review")
    g.add_edge("analyse_impact", "confirm_publish")
    g.add_conditional_edges("confirm_publish", _after_confirm, ["publish", END])
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer)


def open_checkpointer() -> SqliteSaver:
    path = get_settings().storage_dir / "checkpoints" / "docs_agent.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


GRAPH = build_graph(open_checkpointer())


def pending_interrupt(config: dict[str, Any]) -> dict | None:
    state = GRAPH.get_state(config)
    for task in state.tasks or ():
        if task.interrupts:
            return task.interrupts[0].value
    return None
