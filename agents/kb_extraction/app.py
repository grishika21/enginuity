"""KB extraction agent: questions in, cited answers out.

Vector search over the index, then a GPT-style answer written only from the retrieved chunks,
with numbered citations checked in code. When a question matches several documents equally
well, the agent asks which one is meant instead of guessing. Other agents can call it in
`chunks` mode to get raw source content without an answer, either the best matches for a query
or, with `file_id`, one whole document in reading order.

Searches only ever see the global sample documents and the caller's own sandbox's.
"""

from __future__ import annotations

import json
import os
import re

from fastapi import HTTPException, Request

from agentkit.auth import AuthError, bearer, verify
from agentkit.contract import AgentCard, AgentError, Completed, NeedsInput, Question, Skill
from agentkit.llm import get_llm
from agentkit.redis_conn import get_redis
from agentkit.safety import USER_CONTENT_RULE, wrap_user
from agentkit.sandbox import sandbox_of, visible_sandboxes
from agentkit.server import JobContext, create_agent_app
from agentkit.vectors import Hit, embed, get_store
from agents.kb_common import COLLECTION, equipment_codes, visible_docs

CARD = AgentCard(
    name="kb-extraction-agent",
    title="Knowledge-base answers",
    description="Answers questions from the documents in the knowledge base (equipment manuals, "
                "service bulletins, procedures and any document a user has added), with page-level "
                "sources. Use for EVERY question: how do I, how often, what, which, what value, why, "
                "what does it say about, what are the key points of, summarise, including step-by-step "
                "how-to questions.",
    version="1.0.0",
    url=os.environ.get("KB_EXTRACTION_URL", "http://localhost:8102"),
    skills=[Skill(id="ask", name="Answer from the knowledge base",
                  description="Find the relevant passages and answer with citations.")],
    allowed_roles=["viewer", "author"],
)

MIN_SCORE = 0.28
DOCUMENT_CHARS = 12000  # how much of one document a peer gets in whole-document mode
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
        title = opt["title"].lower()
        if reply and (reply in title or title in reply or any(code.lower() in reply for code in opt["equipment"])):
            return opt["equipment"]
    return None


def search(query: str, doc_id: str | None, user: str, top_k: int = 12) -> list[Hit]:
    return get_store().search(COLLECTION, embed([query])[0], top_k=top_k, doc_id=doc_id,
                              visible=visible_sandboxes(user))


def scope_hits(hits: list[Hit], codes: set[str]) -> list[Hit]:
    """Keep only the chosen equipment's documents (or the chosen document), when any is given."""
    if not codes:
        return hits
    scoped = [h for h in hits if codes & (set(h.payload.get("equipment") or []) | {h.payload.get("doc_id")})]
    return scoped or hits


def whole_document(file_id: str, ctx: JobContext) -> list[dict] | None:
    """One indexed document's chunks in reading order, up to DOCUMENT_CHARS. None if the file is
    not in the knowledge base this user can see."""
    if not ctx.files.can_read(file_id, ctx.user):
        raise AgentError("You do not have access to that file.")
    doc_id = next((d for d, m in visible_docs(ctx.user).items() if m.get("file_id") == file_id), None)
    if doc_id is None:
        return None
    chunks, used = [], 0
    for chunk in sorted(get_store().chunks(COLLECTION, doc_id), key=lambda c: c.get("order", 0)):
        if used + len(chunk["text"]) > DOCUMENT_CHARS and chunks:
            break
        chunks.append(chunk)
        used += len(chunk["text"])
    return chunks


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
        "You answer questions using only the numbered sources. Cite every fact with its "
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

    if mode == "chunks" and ctx.parameters.get("file_id"):
        ctx.progress("Reading the whole document from the knowledge base")
        chunks = whole_document(str(ctx.parameters["file_id"]), ctx)
        if chunks is None:
            return Completed(message="That document is not in the knowledge base yet.",
                             data={"chunks": [], "not_indexed": True})
        return Completed(message=f"{len(chunks)} passages, in reading order.", data={"chunks": chunks})

    ctx.progress("Searching the knowledge base")
    hits = [h for h in search(query, doc_id, ctx.user) if h.score >= MIN_SCORE]
    if mode == "chunks":
        codes |= set(equipment_codes(str(ctx.parameters.get("equipment", "")).upper()))
        hits = scope_hits(hits, codes)
        chunks = [{**h.payload, "score": round(h.score, 3)} for h in hits[: int(ctx.parameters.get("top_k", 8))]]
        return Completed(message=f"{len(chunks)} passages found.", data={"chunks": chunks})

    if not hits:
        names = ", ".join(sorted({d["title"] for d in visible_docs(ctx.user).values()})) or "none yet"
        return Completed(message="I couldn't find that in the knowledge base. Documents I can search: " + names + ".",
                         data={"sources": []})

    hits = scope_hits(hits, codes)
    if not doc_id and not codes:
        options = ambiguous_equipment(hits)
        if options:
            r.set(_pending_key(ctx.session_id), json.dumps({"query": query, "options": options}), ex=PENDING_TTL_S)
            listing = "\n".join(f"{i}. {o['title']}" for i, o in enumerate(options, 1))
            # equipment options are titled by their model code; document options by the document title
            what = "piece of equipment" if all(o["title"] in o["equipment"] for o in options) else "document"
            return NeedsInput(question=Question(
                text=f"This is covered for more than one {what}. Which one do you mean?\n{listing}",
                options=[o["title"] for o in options], kind="choice"))

    ctx.progress("Writing the answer", f"from {min(len(hits), 6)} passages")
    text, used = answer(query, hits[:6])
    refs = "\n".join(f"[{s['n']}] {s['title']}, p.{s['page']}: {s['heading']}" for s in used)
    return Completed(message=f"{text}\n\n**Sources**\n{refs}", data={"answer": text, "sources": used})


app = create_agent_app(CARD, handle)


@app.get("/catalog")
async def catalog(request: Request) -> list[dict]:
    """The documents this caller can search, for the UI: global samples plus their sandbox's."""
    try:
        ident = verify(bearer(request.headers.get("authorization")))
    except AuthError as exc:
        raise HTTPException(status_code=401, detail="Sign in again.") from exc
    mine = sandbox_of(ident.user)
    out = []
    for doc_id, meta in visible_docs(ident.user).items():
        out.append({"doc_id": doc_id, "title": meta.get("title", ""), "file_id": meta.get("file_id"),
                    "file_name": meta.get("file_name", ""), "pages": meta.get("pages"), "chunks": meta.get("chunks"),
                    "strategy": meta.get("strategy"), "ocr": bool(meta.get("ocr")),
                    "equipment": meta.get("equipment") or [],
                    "shared_sample": not meta.get("sandbox"), "added_by_you": bool(mine) and meta.get("sandbox") == mine})
    return sorted(out, key=lambda d: (not d["shared_sample"], d["title"]))
