"""Build a FastAPI service for one agent.

Every agent gets the same job API, auth check, input screening, event stream, registry
heartbeat and agent card. The agent itself only supplies a handler: a function that receives a
`JobContext` and returns `Completed` or `NeedsInput`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import AuthError, Identity, allowed, bearer, verify
from .client import call_agent
from .config import get_settings
from .contract import (MAX_CALL_CHAIN, AgentCard, AgentError, Completed, FileRef, InvokeRequest, Job,
                       NeedsInput)
from .files import FileStore
from .jobs import JobStore
from .llm import LLMError
from .registry import Registrar, find_agent, find_by_skill
from .safety import screen, scrub

log = logging.getLogger("agentkit.server")


class Cancelled(Exception):
    pass


class JobContext:
    def __init__(self, job: Job, req: InvokeRequest, ident: Identity, card: AgentCard, store: JobStore,
                 files: FileStore):
        self.job_id = job.job_id
        self.session_id = job.session_id
        self.request = req
        self.message = req.message
        self.parameters = req.parameters
        self.user = ident.user
        self.roles = ident.roles
        self.token = ident.token
        self.card = card
        self.store = store
        self.files = files

    def progress(self, stage: str, detail: str = "", agent: str | None = None) -> None:
        self.check_cancel()
        self.store.update(self.job_id, stage=stage)
        self.store.emit(self.job_id, "progress", stage=stage, detail=scrub(detail), agent=agent or self.card.name)

    def check_cancel(self) -> None:
        if self.store.is_cancelled(self.job_id):
            raise Cancelled()

    def save_output(self, name: str, data: bytes) -> FileRef:
        return self.files.put(self.user, name, data, kind=f"output:{self.card.name}")

    def call_peer(self, message: str, *, name: str | None = None, skill: str | None = None,
                  parameters: dict[str, Any] | None = None, files: list[str] | None = None,
                  session_id: str | None = None) -> Job:
        """Call another agent found in the registry, forwarding this caller's token."""
        card = find_agent(name) if name else find_by_skill(skill or "")
        if card is None:
            raise AgentError(f"The {name or skill} agent is not available right now.")
        req = InvokeRequest(message=message, parameters=parameters or {}, files=files or [],
                            session_id=session_id or f"{self.session_id}:{card.name}",
                            call_chain=[*self.request.call_chain, self.card.name])
        self.progress(f"Calling {card.name}", agent=self.card.name)
        return call_agent(card, req, self.token,
                          on_event=lambda e: self.store.emit(self.job_id, "progress", stage=e.get("stage", ""),
                                                             detail=e.get("detail", ""), agent=e.get("agent", card.name)))


Handler = Callable[[JobContext], Completed | NeedsInput]


def create_agent_app(card: AgentCard, handler: Handler) -> FastAPI:
    settings = get_settings()
    store = JobStore()
    files = FileStore()
    registrar = Registrar(card, settings.heartbeat_s)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await registrar.start()
        yield
        await registrar.stop()

    app = FastAPI(title=card.title, version=card.version, lifespan=lifespan)
    app.state.registrar = registrar
    app.state.card = card

    def identity(request: Request) -> Identity:
        try:
            ident = verify(bearer(request.headers.get("authorization")))
        except AuthError as exc:
            raise HTTPException(status_code=401, detail="Sign in again.") from exc
        if not allowed(ident.roles, card.allowed_roles):
            raise HTTPException(status_code=403, detail=f"Your role cannot use {card.name}.")
        return ident

    def owned_job(job_id: str, ident: Identity) -> Job:
        job = store.get(job_id)
        if job is None or (job.owner != ident.user and "admin" not in ident.roles):
            raise HTTPException(status_code=404, detail="Unknown job.")
        return job

    @app.get("/v1/health")
    async def health() -> dict:
        return {"status": "ok", "agent": card.name, "version": card.version}

    @app.get("/.well-known/agent-card.json")
    async def agent_card() -> dict:
        return card.model_dump()

    @app.post("/v1/invoke", status_code=202)
    async def invoke(req: InvokeRequest, request: Request) -> dict:
        ident = identity(request)
        if card.name in req.call_chain or len(req.call_chain) >= MAX_CALL_CHAIN:
            raise HTTPException(status_code=400, detail="Refused: the agent call chain loops or is too long.")
        reason = screen(req.message) or next(
            (screen(v) for v in req.parameters.values() if isinstance(v, str) and screen(v)), None)
        if reason:
            raise HTTPException(status_code=400, detail=f"I can't help with that request because {reason}.")
        for file_id in req.files:
            if not files.can_read(file_id, ident.user):
                raise HTTPException(status_code=403, detail="You do not have access to one of the files.")
        session_id = req.session_id or f"sess-{uuid.uuid4().hex[:12]}"
        job = store.create(card.name, session_id, ident.user)
        store.emit(job.job_id, "status", status="pending", agent=card.name)
        threading.Thread(target=_run, args=(job, req, ident), daemon=True).start()
        return {"job_id": job.job_id, "session_id": session_id}

    def _run(job: Job, req: InvokeRequest, ident: Identity) -> None:
        ctx = JobContext(job, req, ident, card, store, files)
        store.update(job.job_id, status="running")
        store.emit(job.job_id, "status", status="running", agent=card.name)
        try:
            outcome = handler(ctx)
            if isinstance(outcome, NeedsInput):
                store.update(job.job_id, status="input_required", question=outcome.question,
                             data=outcome.data, files=outcome.files, message=scrub(outcome.question.text))
            else:
                store.update(job.job_id, status="completed", message=scrub(outcome.message),
                             data=outcome.data, files=outcome.files)
        except Cancelled:
            store.update(job.job_id, status="cancelled", message="Cancelled.")
        except (AgentError, LLMError) as exc:
            store.update(job.job_id, status="failed", message=scrub(str(exc)))
        except Exception:
            log.exception("job %s failed", job.job_id)
            store.update(job.job_id, status="failed", message=f"Something went wrong in {card.name}.")
        final = store.get(job.job_id)
        store.emit(job.job_id, "done", status=final.status if final else "failed", agent=card.name)

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, request: Request) -> dict:
        return owned_job(job_id, identity(request)).model_dump()

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, request: Request) -> dict:
        owned_job(job_id, identity(request))
        store.request_cancel(job_id)
        return {"status": "cancelling"}

    @app.get("/v1/jobs/{job_id}/stream")
    async def stream_job(job_id: str, request: Request) -> StreamingResponse:
        owned_job(job_id, identity(request))
        after = request.headers.get("last-event-id", "0")

        async def gen():
            last = after
            while True:
                batch = await asyncio.to_thread(store.events, job_id, last, 1000)
                if not batch:
                    job = store.get(job_id)
                    if job is None:
                        return
                    yield ": keepalive\n\n"
                    continue
                for entry_id, body in batch:
                    last = entry_id
                    yield f"id: {entry_id}\nevent: {body['event']}\ndata: {json.dumps(body, default=str)}\n\n"
                    if body["event"] == "done":
                        return

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.exception_handler(AgentError)
    async def agent_error(_request: Request, exc: AgentError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app
