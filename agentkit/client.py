"""Calling an agent: start a job, follow its SSE stream, read the final job.

Used by the orchestrator (async) and by agents that call peers (sync, from a worker thread).
The caller's bearer token and the call chain are forwarded on every call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Callable, Iterator

import httpx

from .contract import AgentCard, AgentError, InvokeRequest, Job

log = logging.getLogger("agentkit.client")

RETRY_STATUS = {429, 502, 503, 504}


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _parse_sse(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                try:
                    yield json.loads("\n".join(data))
                except json.JSONDecodeError:
                    pass
                data = []
            continue
        if line.startswith("data:"):
            data.append(line[5:].strip())


def _refusal(resp: httpx.Response, agent: str) -> AgentError:
    try:
        detail = resp.json().get("detail")
    except Exception:
        detail = None
    if resp.status_code == 403:
        return AgentError(f"You are not allowed to use {agent}.")
    if isinstance(detail, str):
        return AgentError(detail)
    return AgentError(f"{agent} refused the request ({resp.status_code}).")


# ---- sync (agents calling peers) ----------------------------------------------------------


def call_agent(card: AgentCard, req: InvokeRequest, token: str,
               on_event: Callable[[dict[str, Any]], None] | None = None, timeout_s: float = 600) -> Job:
    with httpx.Client(base_url=card.url, headers=_headers(token), timeout=30) as client:
        resp = None
        for attempt in range(3):
            resp = client.post("/v1/invoke", json=req.model_dump())
            if resp.status_code not in RETRY_STATUS:
                break
            time.sleep(1.5 * (attempt + 1))
        if resp is None or resp.status_code != 202:
            raise _refusal(resp, card.name)
        job_id = resp.json()["job_id"]
        try:
            with client.stream("GET", f"/v1/jobs/{job_id}/stream", timeout=httpx.Timeout(30, read=timeout_s)) as stream:
                for event in _parse_sse(stream.iter_lines()):
                    if on_event and event.get("event") == "progress":
                        on_event(event)
                    if event.get("event") == "done":
                        break
        except httpx.HTTPError as exc:
            log.warning("stream from %s broke, falling back to the job record: %s", card.name, exc)
        final = client.get(f"/v1/jobs/{job_id}")
        final.raise_for_status()
        return Job.model_validate(final.json())


# ---- async (orchestrator) ----------------------------------------------------------------


class AsyncAgentClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=900))

    async def close(self) -> None:
        await self._client.aclose()

    async def start(self, card: AgentCard, req: InvokeRequest, token: str) -> str:
        resp = None
        for attempt in range(3):
            resp = await self._client.post(f"{card.url}/v1/invoke", json=req.model_dump(), headers=_headers(token))
            if resp.status_code not in RETRY_STATUS:
                break
            await asyncio.sleep(1.5 * (attempt + 1))
        if resp is None or resp.status_code != 202:
            raise _refusal(resp, card.name)
        return resp.json()["job_id"]

    async def events(self, card: AgentCard, job_id: str, token: str) -> AsyncIterator[dict[str, Any]]:
        async with self._client.stream("GET", f"{card.url}/v1/jobs/{job_id}/stream", headers=_headers(token)) as stream:
            data: list[str] = []
            async for line in stream.aiter_lines():
                if line == "":
                    if data:
                        try:
                            event = json.loads("\n".join(data))
                        except json.JSONDecodeError:
                            event = None
                        data = []
                        if event:
                            yield event
                            if event.get("event") == "done":
                                return
                    continue
                if line.startswith("data:"):
                    data.append(line[5:].strip())

    async def job(self, card: AgentCard, job_id: str, token: str) -> Job:
        resp = await self._client.get(f"{card.url}/v1/jobs/{job_id}", headers=_headers(token))
        resp.raise_for_status()
        return Job.model_validate(resp.json())

    async def cancel(self, card: AgentCard, job_id: str, token: str) -> None:
        await self._client.post(f"{card.url}/v1/jobs/{job_id}/cancel", headers=_headers(token))
