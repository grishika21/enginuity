"""Job records and per-job event streams in Redis.

One JSON record per job (status, stage, result, owner) and one Redis Stream per job holding
every progress and terminal event, so a client that reconnects can replay what it missed.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator

from .contract import TERMINAL, Job
from .redis_conn import get_redis

JOB_TTL_S = 24 * 3600


def _key(job_id: str) -> str:
    return f"job:{job_id}"


def _stream(job_id: str) -> str:
    return f"job:{job_id}:events"


class JobStore:
    def __init__(self, redis_client=None):
        self.r = redis_client or get_redis()

    def create(self, agent: str, session_id: str, owner: str) -> Job:
        now = time.time()
        job = Job(job_id=f"job-{uuid.uuid4().hex[:12]}", session_id=session_id, agent=agent,
                  owner=owner, created_at=now, updated_at=now)
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        job.updated_at = time.time()
        self.r.set(_key(job.job_id), job.model_dump_json(), ex=JOB_TTL_S)

    def get(self, job_id: str) -> Job | None:
        raw = self.r.get(_key(job_id))
        return Job.model_validate_json(raw) if raw else None

    def update(self, job_id: str, **fields: Any) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        for name, value in fields.items():
            setattr(job, name, value)
        self.save(job)
        return job

    def emit(self, job_id: str, event: str, **payload: Any) -> None:
        body = {"event": event, "ts": time.time(), **payload}
        self.r.xadd(_stream(job_id), {"data": json.dumps(body, default=str)}, maxlen=2000)
        self.r.expire(_stream(job_id), JOB_TTL_S)

    def events(self, job_id: str, after: str = "0", block_ms: int = 1000) -> list[tuple[str, dict]]:
        res = self.r.xread({_stream(job_id): after}, block=block_ms, count=100)
        out: list[tuple[str, dict]] = []
        for _stream_name, entries in res or []:
            for entry_id, fields in entries:
                out.append((entry_id, json.loads(fields["data"])))
        return out

    def follow(self, job_id: str, after: str = "0", timeout_s: float = 900) -> Iterator[tuple[str, dict]]:
        """Yield events until the job reaches a terminal state (blocking; run in a thread)."""
        deadline = time.time() + timeout_s
        last = after
        while time.time() < deadline:
            batch = self.events(job_id, last)
            for entry_id, body in batch:
                last = entry_id
                yield entry_id, body
                if body.get("event") == "done":
                    return
            if not batch:
                job = self.get(job_id)
                if job is None:
                    return
                if job.status in TERMINAL and not self.events(job_id, last, block_ms=1):
                    return

    def request_cancel(self, job_id: str) -> None:
        self.r.set(f"{_key(job_id)}:cancel", "1", ex=JOB_TTL_S)

    def is_cancelled(self, job_id: str) -> bool:
        return bool(self.r.exists(f"{_key(job_id)}:cancel"))
