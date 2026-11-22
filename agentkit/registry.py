"""Agent self-registration and discovery through Redis.

Each agent writes its card into the `agents:registry` hash and refreshes a heartbeat key with a
short TTL. Discovery returns only agents whose heartbeat is alive, so a new agent appears without
a redeploy of anything else and a dead one drops out on its own.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from .contract import AgentCard
from .redis_conn import get_redis

log = logging.getLogger("agentkit.registry")

REGISTRY = "agents:registry"
EVENTS = "registry:events"


def _hb(name: str) -> str:
    return f"agents:heartbeat:{name}"


class Registrar:
    def __init__(self, card: AgentCard, heartbeat_s: int, redis_client=None):
        self.card = card
        self.heartbeat_s = heartbeat_s
        self.r = redis_client or get_redis()
        self._task: asyncio.Task | None = None

    def beat(self) -> None:
        self.r.hset(REGISTRY, self.card.name, self.card.model_dump_json())
        self.r.set(_hb(self.card.name), str(time.time()), ex=self.heartbeat_s * 3)

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.beat)
            except Exception as exc:
                log.warning("heartbeat for %s failed: %s", self.card.name, exc)
            await asyncio.sleep(self.heartbeat_s)

    async def start(self) -> None:
        try:
            await asyncio.to_thread(self.beat)
            self.r.publish(EVENTS, json.dumps({"event": "registered", "agent": self.card.name}))
            log.info("registered %s at %s", self.card.name, self.card.url)
        except Exception as exc:
            log.warning("registration of %s failed, will retry: %s", self.card.name, exc)
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        try:
            self.r.hdel(REGISTRY, self.card.name)
            self.r.delete(_hb(self.card.name))
            self.r.publish(EVENTS, json.dumps({"event": "deregistered", "agent": self.card.name}))
        except Exception as exc:
            log.warning("deregistration of %s failed: %s", self.card.name, exc)


def healthy_agents(redis_client=None) -> list[tuple[AgentCard, float]]:
    """Return (card, seconds since last heartbeat) for every live agent."""
    r = redis_client or get_redis()
    out: list[tuple[AgentCard, float]] = []
    now = time.time()
    for name, raw in sorted(r.hgetall(REGISTRY).items()):
        beat = r.get(_hb(name))
        if beat is None:
            continue
        try:
            out.append((AgentCard.model_validate_json(raw), max(0.0, now - float(beat))))
        except Exception as exc:
            log.warning("skipping unreadable card for %s: %s", name, exc)
    return out


def find_agent(name: str, redis_client=None) -> AgentCard | None:
    for card, _age in healthy_agents(redis_client):
        if card.name == name:
            return card
    return None


def find_by_skill(skill_id: str, redis_client=None) -> AgentCard | None:
    for card, _age in healthy_agents(redis_client):
        if any(s.id == skill_id for s in card.skills):
            return card
    return None
