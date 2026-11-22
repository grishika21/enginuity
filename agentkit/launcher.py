"""Start the services: one process each (local), or one HTTP server per thread in a single process.

The single-process mode is for small hosts such as Streamlit Community Cloud: every agent still
has its own port, registry entry, heartbeat and HTTP API, but they share one copy of torch.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import shutil
import subprocess
import threading
import time
from urllib.parse import urlparse

import httpx
import redis

log = logging.getLogger("agentkit.launcher")

SERVICES = [
    ("orchestrator", "orchestrator.app:app", 8000),
    ("kb-ingestion-agent", "agents.kb_ingestion.app:app", 8101),
    ("kb-extraction-agent", "agents.kb_extraction.app:app", 8102),
    ("pi-agent", "agents.pi_agent.app:app", 8103),
    ("docs-agent", "agents.docs_agent.app:app", 8104),
]
AGENT_COUNT = len(SERVICES) - 1


def ensure_redis(url: str) -> subprocess.Popen | None:
    """Use the Redis at `url`, or start a throwaway one on that port when none answers."""
    client = redis.Redis.from_url(url)
    try:
        client.ping()
        return None
    except redis.RedisError:
        pass
    binary = shutil.which("redis-server")
    if not binary:
        raise RuntimeError("Redis is not reachable and redis-server is not installed.")
    port = str(urlparse(url).port or 6379)
    proc = subprocess.Popen([binary, "--port", port, "--bind", "127.0.0.1", "--save", "", "--appendonly", "no"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        try:
            client.ping()
            return proc
        except redis.RedisError:
            time.sleep(0.2)
    raise RuntimeError("redis-server did not start.")


def _load(target: str):
    module, attr = target.split(":")
    return getattr(importlib.import_module(module), attr)


def start_in_process() -> list[threading.Thread]:
    import uvicorn

    threads = []
    for name, target, port in SERVICES:
        app = _load(target)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
        server = uvicorn.Server(config)  # uvicorn skips signal handlers off the main thread
        thread = threading.Thread(target=lambda s=server: asyncio.run(s.serve()), name=name, daemon=True)
        thread.start()
        threads.append(thread)
    return threads


def wait_ready(base_url: str = "http://127.0.0.1:8000", timeout_s: float = 180) -> None:
    from agentkit.registry import healthy_agents

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{base_url}/api/users", timeout=2).status_code == 200 and \
                    len(healthy_agents()) >= AGENT_COUNT:
                return
        except (httpx.HTTPError, redis.RedisError):
            pass
        time.sleep(0.5)
    raise RuntimeError("The services did not come up in time.")
