"""Enginuity orchestrator: one chat in front of many agents.

- Login issues a JWT with the user's roles; the token is forwarded on every agent call.
- The router (LLM tool calling) picks an agent from the live Redis registry, filtered by role.
- A step approval gate asks the user before calling an agent whose skill is marked sensitive.
- When an agent pauses with a question, the question goes to the chat and the next reply
  resumes the same agent session.
- Every turn runs as a background job streamed to the UI over SSE; chats are saved per user.
- Files are reached only through short-lived signed links; internal paths never reach the UI.
- On the hosted demo each visitor signs in inside a private sandbox (see agentkit.sandbox).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentkit.approval import is_explicit_yes
from agentkit.auth import AuthError, Identity, allowed, bearer, issue, verify
from agentkit.client import AsyncAgentClient
from agentkit.config import get_settings
from agentkit.contract import AgentCard, AgentError, InvokeRequest
from agentkit.files import FileStore, sign_link, verify_link
from agentkit.jobs import JobStore
from agentkit.llm import LLMError, get_llm
from agentkit.registry import find_agent, healthy_agents
from agentkit.safety import screen, scrub
from agentkit.sandbox import persona, sandbox_of, scoped, valid_sandbox
from orchestrator import sandbox as sandboxes
from orchestrator.history import HistoryStore
from orchestrator.router import route

log = logging.getLogger("orchestrator")

STATIC = Path(__file__).resolve().parent / "static"
UPLOAD_TYPES = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".md", ".html", ".htm", ".txt"}
MAX_UPLOAD = 20 * 1024 * 1024

settings = get_settings()
jobs = JobStore()
files = FileStore()
history = HistoryStore()
client = AsyncAgentClient()


def load_users() -> dict[str, dict]:
    return json.loads((settings.data_dir / "users.json").read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await client.close()


app = FastAPI(title="Enginuity orchestrator", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def identity(request: Request) -> Identity:
    try:
        return verify(bearer(request.headers.get("authorization")))
    except AuthError as exc:
        raise HTTPException(status_code=401, detail="Sign in again.") from exc


def readable_conv(conv_id: str, ident: Identity) -> dict:
    try:
        conv = history.get(conv_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown chat.") from exc
    if not history.can_read(conv, ident.user):
        raise HTTPException(status_code=404, detail="Unknown chat.")
    return conv


# ---- auth and catalog ----------------------------------------------------------------------


class Login(BaseModel):
    username: str
    password: str
    sandbox: str | None = None  # the hosted demo signs every visitor in to their own sandbox


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/login")
async def login(body: Login) -> dict:
    user = load_users().get(body.username)
    digest = hashlib.sha256(body.password.encode()).hexdigest()
    if not user or not hmac.compare_digest(digest, user["password_sha256"]):
        raise HTTPException(status_code=401, detail="Wrong user name or password.")
    if body.sandbox is not None and not valid_sandbox(body.sandbox):
        raise HTTPException(status_code=400, detail="Invalid sandbox id.")
    user_id = scoped(body.username, body.sandbox)
    return {"token": issue(user_id, user["roles"]), "user": user_id, "persona": body.username,
            "name": user["name"], "roles": user["roles"], "sandbox": body.sandbox or ""}


@app.get("/api/users")
async def demo_users() -> list[dict]:
    return [{"user": u, "name": d["name"], "roles": d["roles"]} for u, d in load_users().items()]


@app.get("/api/agents")
async def list_agents(request: Request) -> list[dict]:
    ident = identity(request)
    out = []
    for card, age in await asyncio.to_thread(healthy_agents):
        out.append({"name": card.name, "title": card.title, "version": card.version,
                    "skills": [{"id": s.id, "name": s.name, "approval": s.requires_approval} for s in card.skills],
                    "heartbeat_age_s": round(age, 1), "allowed": allowed(ident.roles, card.allowed_roles),
                    "routable": card.routable})
    return out


@app.get("/api/knowledge")
async def knowledge(request: Request) -> list[dict]:
    """What the knowledge base holds for this user, from the Q&A agent that owns the index."""
    ident = identity(request)
    card = find_agent("kb-extraction-agent")
    if card is None:
        return []
    resp = await client._client.get(f"{card.url}/catalog", headers={"Authorization": f"Bearer {ident.token}"})
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="The knowledge base did not answer.")
    return resp.json()


@app.post("/api/sandbox/reset")
async def reset_sandbox(request: Request) -> dict:
    """Delete everything created in the caller's sandbox. The shared sample data stays."""
    ident = identity(request)
    sandbox = sandbox_of(ident.user)
    if not sandbox:
        raise HTTPException(status_code=400, detail="Only a sandboxed demo session can be reset.")
    removed = await asyncio.to_thread(sandboxes.reset, sandbox, list(load_users()), files, history)
    log.info("sandbox %s reset by %s: %s", sandbox, persona(ident.user), removed)
    return {"sandbox": sandbox, "removed": removed}


@app.get("/api/files")
async def list_files(request: Request) -> list[dict]:
    ident = identity(request)
    return await asyncio.to_thread(files.list_for, ident.user)


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...), conversation_id: str = Form("")) -> dict:
    ident = identity(request)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in UPLOAD_TYPES:
        raise HTTPException(status_code=400, detail=f"{suffix or 'That'} files are not supported.")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=400, detail="Files up to 20 MB only.")
    ref = files.put(ident.user, file.filename or "upload", data)
    if conversation_id:
        conv = readable_conv(conversation_id, ident)
        conv["files"].append(ref.model_dump())
        history.save(conv)
    return ref.model_dump()


@app.get("/api/files/{file_id}/link")
async def file_link(file_id: str, request: Request) -> dict:
    ident = identity(request)
    if not files.can_read(file_id, ident.user):
        raise HTTPException(status_code=404, detail="Unknown file.")
    return {"url": sign_link(file_id, ident.user), "expires_in_s": settings.link_ttl_s}


@app.get("/files/{file_id}")
async def download(file_id: str, u: str, exp: int, sig: str) -> FileResponse:
    if not verify_link(file_id, u, exp, sig) or not files.can_read(file_id, u):
        raise HTTPException(status_code=403, detail="This link has expired. Open the file from the chat again.")
    path = files.path(file_id, u)
    return FileResponse(path, filename=files.meta(file_id)["name"])


# ---- chats ---------------------------------------------------------------------------------


@app.get("/api/conversations")
async def conversations(request: Request) -> list[dict]:
    return history.list_for(identity(request).user)


@app.get("/api/conversations/{conv_id}")
async def conversation(conv_id: str, request: Request) -> dict:
    return readable_conv(conv_id, identity(request))


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str, request: Request) -> dict:
    ident = identity(request)
    conv = readable_conv(conv_id, ident)
    if conv["owner"] != ident.user:
        raise HTTPException(status_code=403, detail="Only the owner can delete a chat.")
    history.delete(conv_id)
    return {"deleted": conv_id}


class Share(BaseModel):
    user: str


@app.post("/api/conversations/{conv_id}/share")
async def share(conv_id: str, body: Share, request: Request) -> dict:
    ident = identity(request)
    conv = readable_conv(conv_id, ident)
    if conv["owner"] != ident.user:
        raise HTTPException(status_code=403, detail="Only the owner can share a chat.")
    if body.user not in load_users() or body.user == ident.user:
        raise HTTPException(status_code=400, detail="Unknown user.")
    if body.user not in conv["members"]:
        conv["members"].append(body.user)
        for ref in conv["files"]:
            files.share(ref["file_id"], body.user)
        history.save(conv)
    return {"members": conv["members"]}


# ---- turns ---------------------------------------------------------------------------------


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None
    files: list[str] = Field(default_factory=list, max_length=5)


@app.post("/api/chat", status_code=202)
async def chat(body: ChatIn, request: Request) -> dict:
    ident = identity(request)
    reason = screen(body.message)
    conv = readable_conv(body.conversation_id, ident) if body.conversation_id else history.create(ident.user, body.message)
    for file_id in body.files:
        if not files.can_read(file_id, ident.user):
            raise HTTPException(status_code=403, detail="You do not have access to one of the files.")
    job = jobs.create("orchestrator", conv["id"], ident.user)
    asyncio.create_task(run_turn(job.job_id, conv["id"], ident, body.message, body.files, reason))
    return {"job_id": job.job_id, "conversation_id": conv["id"]}


@app.get("/api/turns/{job_id}/stream")
async def turn_stream(job_id: str, request: Request) -> StreamingResponse:
    ident = identity(request)
    job = jobs.get(job_id)
    if job is None or job.owner != ident.user:
        raise HTTPException(status_code=404, detail="Unknown turn.")

    async def gen():
        last = "0"
        while True:
            batch = await asyncio.to_thread(jobs.events, job_id, last, 1000)
            if not batch:
                yield ": keepalive\n\n"
                continue
            for entry_id, body in batch:
                last = entry_id
                yield f"id: {entry_id}\nevent: {body['event']}\ndata: {json.dumps(body, default=str)}\n\n"
                if body["event"] == "done":
                    return

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _documents(conv: dict, user: str) -> dict[str, str]:
    """Document names the router may pick, mapped to file ids (newest wins)."""
    names: dict[str, str] = {}
    for meta in sorted(files.list_for(user), key=lambda m: m["created"]):
        if meta["owner"] == "library" or meta["kind"] == "upload":
            names[meta["name"]] = meta["file_id"]
    for ref in conv.get("files", []):
        names[ref["name"]] = ref["file_id"]
    return names


class Turn:
    """One user message: route, maybe ask for approval, call the agent, relay its events."""

    def __init__(self, job_id: str, conv: dict, ident: Identity):
        self.job_id = job_id
        self.conv = conv
        self.ident = ident
        self.reply: dict[str, Any] = {"role": "assistant", "text": "", "agent": None, "files": [], "sources": [],
                                      "question": None, "steps": [], "ts": time.time()}

    def step(self, stage: str, detail: str = "", agent: str = "orchestrator") -> None:
        item = {"stage": scrub(stage), "detail": scrub(detail), "agent": agent, "ts": time.time()}
        self.reply["steps"].append(item)
        jobs.emit(self.job_id, "progress", **item)

    def ask(self, text: str, options: list[str], kind: str) -> None:
        self.reply["question"] = {"text": text, "options": options, "kind": kind}
        self.reply["text"] = text

    async def call(self, card: AgentCard, request: str, file_ids: list[str]) -> None:
        self.reply["agent"] = card.name
        req = InvokeRequest(message=request, session_id=f"{self.conv['id']}:{card.name}", files=file_ids,
                            call_chain=["orchestrator"])
        self.step(f"Calling {card.title}", agent=card.name)
        agent_job = await client.start(card, req, self.ident.token)
        try:
            async for event in client.events(card, agent_job, self.ident.token):
                if event.get("event") == "progress":
                    self.step(event.get("stage", ""), event.get("detail", ""), event.get("agent", card.name))
        except Exception as exc:  # the job record below still has the outcome
            log.warning("event stream from %s broke: %s", card.name, exc)
        job = await client.job(card, agent_job, self.ident.token)
        self.reply["files"] = [f.model_dump() for f in job.files]
        known = {f["file_id"] for f in self.conv["files"]}
        self.conv["files"] += [f for f in self.reply["files"] if f["file_id"] not in known]
        if job.status == "input_required" and job.question:
            self.conv["pending"] = {"kind": "agent", "agent": card.name}
            self.ask(job.question.text, job.question.options, job.question.kind)
        elif job.status == "completed":
            self.reply["text"] = job.message
            self.reply["sources"] = job.data.get("sources", [])
        else:
            self.reply["text"] = f"{card.title} couldn't finish: {job.message or job.status}"

    async def run(self, message: str, file_ids: list[str], refused: str | None) -> None:
        if refused:
            self.reply["text"] = f"I can't help with that request because {refused}."
            return
        pending = self.conv.get("pending")
        self.conv["pending"] = None
        if pending and pending["kind"] == "approval":
            card = find_agent(pending["agent"])
            if not is_explicit_yes(message):
                self.reply["text"] = f"OK, I did not call {pending['agent']}."
            elif card is None:
                self.reply["text"] = f"{pending['agent']} is offline right now."
            else:
                self.step("Step approved by the user", pending["agent"])
                await self.call(card, pending["request"], pending["files"])
            return
        if pending and pending["kind"] == "agent":
            card = find_agent(pending["agent"])
            if card is None:
                self.reply["text"] = f"{pending['agent']} went offline before it could continue. Please ask again."
                return
            self.step("Your answer goes back to the waiting agent", card.name)
            await self.call(card, message, file_ids)
            return

        documents = _documents(self.conv, self.ident.user)
        cards = [c for c, _age in await asyncio.to_thread(healthy_agents)]
        self.step("Routing", f"{len(cards)} agents online")
        recent = [{"role": m["role"], "text": m["text"]} for m in self.conv["messages"][:-1] if m.get("text")]
        attached = [files.meta(f)["name"] for f in file_ids]
        decision = await asyncio.to_thread(route, get_llm(), cards, self.ident.roles, message, recent, list(documents),
                                           attached)
        if decision.card is None:
            self.reply["text"] = decision.reply
            return
        card = decision.card
        if not decision.allowed:  # enforced in code, whatever the model chose
            self.step(f"Refused: {card.name} is not allowed for your role", ", ".join(self.ident.roles))
            self.reply["text"] = (f"That needs the **{card.title}**, which your role "
                                  f"({', '.join(self.ident.roles)}) can't use. Ask someone with the "
                                  f"{' or '.join(card.allowed_roles)} role.")
            return
        chosen = list(file_ids)  # files the user attached to this message always win
        if card.document_input and not chosen:
            if decision.document:
                chosen = [documents[decision.document]]
            elif self.conv["files"]:
                chosen = [self.conv["files"][-1]["file_id"]]
        self.step(f"Routed to {card.name}", ", ".join(files.meta(f)["name"] for f in chosen))
        if any(s.requires_approval for s in card.skills):
            names = ", ".join(files.meta(f)["name"] for f in chosen) or "no document"
            self.conv["pending"] = {"kind": "approval", "agent": card.name, "request": decision.request, "files": chosen}
            self.reply["agent"] = card.name
            self.ask(f"**Approval needed.** The {card.title} agent changes shared data: {names}. "
                     "Nothing happens until you say yes. Go ahead?", ["yes", "no"], "approval")
            return
        await self.call(card, decision.request, chosen)


async def run_turn(job_id: str, conv_id: str, ident: Identity, message: str, file_ids: list[str],
                   refused: str | None) -> None:
    conv = history.get(conv_id)
    attached = [{"file_id": f, "name": files.meta(f)["name"]} for f in file_ids]
    conv["messages"].append({"role": "user", "text": message, "files": attached, "user": ident.user, "ts": time.time()})
    known = {f["file_id"] for f in conv["files"]}
    conv["files"] += [f for f in attached if f["file_id"] not in known]
    turn = Turn(job_id, conv, ident)
    jobs.update(job_id, status="running")
    try:
        await turn.run(message, file_ids, refused)
    except (AgentError, LLMError) as exc:
        turn.reply["text"] = str(exc)
    except Exception:
        log.exception("turn %s failed", job_id)
        turn.reply["text"] = "Something went wrong while handling that. Please try again."
    turn.reply["text"] = scrub(turn.reply["text"])
    conv["messages"].append(turn.reply)
    history.save(conv)
    jobs.update(job_id, status="completed")
    jobs.emit(job_id, "message", message=turn.reply, conversation_id=conv_id)
    jobs.emit(job_id, "done", status="completed")


@app.post("/api/admin/reload-skills")
async def admin_reload(request: Request, agent: str = "docs-agent") -> dict:
    ident = identity(request)
    if "admin" not in ident.roles:
        raise HTTPException(status_code=403, detail="Admins only.")
    card = find_agent(agent)
    if card is None:
        raise HTTPException(status_code=404, detail=f"{agent} is offline.")
    resp = await client._client.post(f"{card.url}/admin/reload-skills",
                                     headers={"Authorization": f"Bearer {ident.token}"})
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Reload failed.")
    return resp.json()
