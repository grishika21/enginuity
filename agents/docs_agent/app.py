"""Docs agent: authors equipment documentation end to end, with a person in control.

Each document type is a SKILL.md folder. The agent card lists one skill per folder, so a new
folder plus POST /admin/reload-skills makes a new document type available everywhere, including
the orchestrator's router, without a code change or restart.
"""

from __future__ import annotations

import os
import uuid

from fastapi import HTTPException, Request
from langgraph.types import Command

from agentkit.auth import AuthError, bearer, verify
from agentkit.contract import AgentCard, Completed, FileRef, NeedsInput, Question, Skill
from agentkit.redis_conn import get_redis
from agentkit.server import JobContext, create_agent_app
from agents.docs_agent import graph as g

THREAD_TTL_S = 7 * 24 * 3600
STOP_WORDS = {"cancel", "stop", "abort", "start over"}


def card_skills() -> list[Skill]:
    return [Skill(id=s.name, name=s.title, description=s.description) for s in g.SKILLS.values()]


CARD = AgentCard(
    name="docs-agent",
    title="Technical documentation author",
    description="Authors a new document (maintenance procedure, system description or another type it "
                "has a skill for) for review and publishing. Use ONLY when the user explicitly asks to "
                "write, draft, create, author or prepare a document. Never use it to answer a question, "
                "even a how-to question.",
    version="1.0.0",
    url=os.environ.get("DOCS_AGENT_URL", "http://localhost:8104"),
    skills=card_skills(),
    allowed_roles=["author"],
)


def _finished_message(values: dict) -> Completed:
    status = values.get("status")
    if status == "no_skill":
        names = ", ".join(s.title.lower() for s in g.SKILLS.values())
        return Completed(message=f"I can author these document types: {names}. Which one do you need?")
    if status == "no_sources":
        return Completed(message="The knowledge base has nothing on that yet, so I can't write it without "
                                 "inventing content. Add a source document first, then ask again.")
    files = [FileRef(**values["draft_file"])] if values.get("draft_file") else []
    if status == "published":
        return Completed(message=f"Published **{values['title']}** to the library as {values['doc_id']}.",
                         data={"doc_id": values["doc_id"], "title": values["title"]}, files=files)
    if status == "not_published":
        return Completed(message=f"Not published. **{values['title']}** stays a draft (v{values['version']}).",
                         files=files)
    return Completed(message="Done.", files=files)


def handle(ctx: JobContext) -> Completed | NeedsInput:
    r = get_redis()
    key = f"docs:thread:{ctx.session_id}"
    thread_id = r.get(key)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
    paused = bool(thread_id) and g.pending_interrupt(config) is not None

    if paused and ctx.message.strip().lower() in STOP_WORDS:
        r.delete(key)
        return Completed(message="Stopped. Nothing was published.")

    if not paused:
        thread_id = f"{ctx.session_id}:{uuid.uuid4().hex[:8]}"
        r.set(key, thread_id, ex=THREAD_TTL_S)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
    g.CTX[thread_id] = ctx
    try:
        if paused:
            ctx.progress("Resuming the paused run", "from its checkpoint")
            g.GRAPH.invoke(Command(resume=ctx.message), config)
        else:
            g.GRAPH.invoke({"request": ctx.message}, config)
    finally:
        g.CTX.pop(thread_id, None)

    question = g.pending_interrupt(config)
    if question:
        return NeedsInput(question=Question(text=question["text"], options=question.get("options", []),
                                            kind=question.get("kind", "input")),
                          files=[FileRef(**f) for f in question.get("files", [])])
    r.delete(key)
    return _finished_message(g.GRAPH.get_state(config).values)


app = create_agent_app(CARD, handle)


@app.post("/admin/reload-skills")
async def reload_skills(request: Request) -> dict:
    try:
        ident = verify(bearer(request.headers.get("authorization")))
    except AuthError as exc:
        raise HTTPException(status_code=401, detail="Sign in again.") from exc
    if "admin" not in ident.roles:
        raise HTTPException(status_code=403, detail="Only an admin can reload skills.")
    g.reload_skills()
    CARD.skills = card_skills()
    app.state.registrar.beat()  # republish the card so the orchestrator sees the new skills now
    return {"skills": [s.id for s in CARD.skills]}
