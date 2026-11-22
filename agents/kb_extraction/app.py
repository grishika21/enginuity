"""KB extraction agent: questions in, cited answers out.

Vector search over the index, then a GPT-style answer written only from the retrieved chunks,
with numbered citations checked in code. When a question matches several documents equally
well, the agent asks which one is meant instead of guessing. Other agents can call it in
`chunks` mode to get raw source content without an answer.
"""

from __future__ import annotations

import json
import os
import re

from agentkit.contract import AgentCard, Completed, NeedsInput, Question, Skill
from agentkit.llm import get_llm
from agentkit.redis_conn import get_redis
from agentkit.safety import USER_CONTENT_RULE, wrap_user
from agentkit.server import JobContext, create_agent_app
from agentkit.vectors import Hit, embed, get_store
from agents.kb_common import COLLECTION, all_docs, equipment_codes

CARD = AgentCard(
    name="kb-extraction-agent",
    title="Knowledge-base answers",
    description="Answers questions about the equipment from its manuals, service bulletins and "
                "procedures, with page-level sources. Use for EVERY question: how do I, how often, "
                "what, which part, what value, why, including step-by-step how-to questions.",
    version="1.0.0",
    url=os.environ.get("KB_EXTRACTION_URL", "http://localhost:8102"),
    skills=[Skill(id="ask", name="Answer from the knowledge base",
                  description="Find the relevant passages and answer with citations.")],
    allowed_roles=["viewer", "author"],
)

MIN_SCORE = 0.28
AMBIGUITY_MARGIN = 0.06
PENDING_TTL_S = 3600


def _pending_key(session_id: str) -> str:
    return f"kb:pending:{session_id}"


def _resolve_choice(reply: str, options: list[dict]) -> list[str] | None:
    """Map the user's reply (a number, a title or an equipment code) to that option's equipment."""
    reply = reply.strip().lower()
    m = re.match(r"^\D*(\d+)\b", reply)
    if m and 1 <= int(m.group(1)) <= len(options):
        return options[int(m.group(1)) - 1]["equipment"]
    for opt in options:
        if reply and (reply in opt["title"].lower() or any(code.lower() in reply for code in opt["equipment"])):
            return opt["equipment"]
    return None


def search(query: str, doc_id: str | None, top_k: int = 12) -> list[Hit]:
    return get_store().search(COLLECTION, embed([query])[0], top_k=top_k, doc_id=doc_id)


def scope_hits(hits: list[Hit], codes: set[str]) -> list[Hit]:
    """Keep only documents about the given equipment, when any is given."""
    if not codes:
        return hits
    scoped = [h for h in hits if codes & set(h.payload.get("equipment") or [])]
    return scoped or hits


def ambiguous_equipment(hits: list[Hit]) -> list[dict]:
    """When equally good matches are about different equipment, one option per equipment.

    Documents about the same equipment (a manual and its service bulletin) complement each other,
    so they never count as ambiguous.
    """
    best: dict[str, Hit] = {}
    for h in hits:
        d = h.payload["doc_id"]
        if d not in best or h.score > best[d].score:
            best[d] = h
    ranked = sorted(best.values(), key=lambda h: -h.score)
    if len(ranked) < 2:
        return []
    close = [h for h in ranked if h.score >= ranked[0].score - AMBIGUITY_MARGIN]
    options: dict[str, dict] = {}
    for h in close:
        equipment = h.payload.get("equipment") or []
        key = equipment[0] if equipment else h.payload["doc_id"]
        options.setdefault(key, {"title": key if equipment else h.payload["title"], "equipment": equipment or [key]})
    return list(options.values())[:4] if len(options) > 1 else []


def answer(query: str, hits: list[Hit]) -> tuple[str, list[dict]]:
    sources = []
    for n, h in enumerate(hits, 1):
        p = h.payload
        sources.append({"n": n, "title": p["title"], "page": p.get("page"), "heading": p.get("heading", ""),
                        "file_id": p.get("file_id"), "snippet": p["text"][:300], "score": round(h.score, 3)})
    context = "\n\n".join(f"[{s['n']}] {s['title']}, page {s['page']}, section \"{s['heading']}\":\n{h.payload['text']}"
                          for s, h in zip(sources, hits))
    system = (
        "You answer engineers' questions using only the numbered sources. Cite every fact with its "
        "source number like [1]. If the sources do not contain the answer, say so plainly and do not "
        "guess. Keep values (torques, lengths, intervals, part numbers) exactly as written. If a "
        "service bulletin changes a part or value, give the bulletin's version and say so. Use short "
        "numbered steps for procedures. " + USER_CONTENT_RULE
    )
    text = get_llm().chat([{"role": "system", "content": system},
                           {"role": "user", "content": f"Sources:\n{context}\n\nQuestion:\n{wrap_user(query)}"}])
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    text = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) <= len(sources) else "", text)
    used = [s for s in sources if s["n"] in cited] or sources[:3]
    return text, used


def handle(ctx: JobContext) -> Completed | NeedsInput:
    r = get_redis()
    query = ctx.message
    doc_id = ctx.parameters.get("doc_id")
    mode = ctx.parameters.get("mode", "answer")

    codes = set(equipment_codes(query.upper()))
    pending_raw = r.get(_pending_key(ctx.session_id))
    if pending_raw and mode == "answer":
        pending = json.loads(pending_raw)
        chosen = _resolve_choice(ctx.message, pending["options"])
        r.delete(_pending_key(ctx.session_id))
        if chosen:
            codes, query = set(chosen), pending["query"]

    ctx.progress("Searching the knowledge base")
    hits = [h for h in search(query, doc_id) if h.score >= MIN_SCORE]
    if mode == "chunks":
        codes |= set(equipment_codes(str(ctx.parameters.get("equipment", "")).upper()))
        hits = scope_hits(hits, codes)
        chunks = [{**h.payload, "score": round(h.score, 3)} for h in hits[: int(ctx.parameters.get("top_k", 8))]]
        return Completed(message=f"{len(chunks)} passages found.", data={"chunks": chunks})

    if not hits:
        names = ", ".join(sorted({d["title"] for d in all_docs().values()})) or "none yet"
        return Completed(message="I couldn't find that in the knowledge base. Documents I can search: " + names + ".",
                         data={"sources": []})

    hits = scope_hits(hits, codes)
    if not doc_id and not codes:
        options = ambiguous_equipment(hits)
        if options:
            r.set(_pending_key(ctx.session_id), json.dumps({"query": query, "options": options}), ex=PENDING_TTL_S)
            listing = "\n".join(f"{i}. {o['title']}" for i, o in enumerate(options, 1))
            return NeedsInput(question=Question(
                text=f"This is covered for more than one piece of equipment. Which one do you mean?\n{listing}",
                options=[o["title"] for o in options], kind="choice"))

    ctx.progress("Writing the answer", f"from {min(len(hits), 6)} passages")
    text, used = answer(query, hits[:6])
    refs = "\n".join(f"[{s['n']}] {s['title']}, p.{s['page']}: {s['heading']}" for s in used)
    return Completed(message=f"{text}\n\n**Sources**\n{refs}", data={"answer": text, "sources": used})


app = create_agent_app(CARD, handle)
