"""Start the orchestrator and all agents locally, each as its own process.

    python scripts/run_all.py            # everything
    python scripts/run_all.py --qdrant   # also start Qdrant in Docker and use it as the vector index

Logs go to storage/logs/<service>.log. Ctrl+C stops everything.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import redis

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentkit.config import get_settings  # noqa: E402
from agentkit.launcher import SERVICES  # noqa: E402


def check_prereqs() -> None:
    s = get_settings()
    try:
        redis.Redis.from_url(s.redis_url).ping()
    except redis.RedisError:
        sys.exit(f"Redis is not reachable at {s.redis_url}. Start it with: redis-server --daemonize yes")
    try:
        httpx.get(s.llm_base_url.rstrip("/") + "/models", timeout=5,
                  headers={"Authorization": f"Bearer {s.llm_api_key}"}).raise_for_status()
    except httpx.HTTPError:
        print(f"warning: the LLM endpoint {s.llm_base_url} did not answer; routing and drafting will fail.")


def start_qdrant() -> str:
    url = "http://localhost:6333"
    try:
        httpx.get(url + "/readyz", timeout=2)
        return url
    except httpx.HTTPError:
        pass
    subprocess.run(["docker", "rm", "-f", "fielddesk-qdrant"], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", "fielddesk-qdrant", "-p", "6333:6333",
                    "-v", f"{ROOT / 'storage' / 'qdrant'}:/qdrant/storage", "qdrant/qdrant:v1.19.0"], check=True)
    for _ in range(30):
        try:
            httpx.get(url + "/readyz", timeout=2).raise_for_status()
            return url
        except httpx.HTTPError:
            time.sleep(1)
    sys.exit("Qdrant did not start.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant", action="store_true", help="use Qdrant (Docker) as the vector index")
    args = parser.parse_args()
    check_prereqs()
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    if args.qdrant:
        env["QDRANT_URL"] = start_qdrant()
    logs = ROOT / "storage" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for name, target, port in SERVICES:
        log = open(logs / f"{name}.log", "a", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", target, "--host", "127.0.0.1", "--port", str(port)],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((name, port, proc))
        print(f"started {name:22} on :{port}  (log: storage/logs/{name}.log)")
    print("\nFieldDesk: http://localhost:8000   users: asha (author), ravi (viewer), admin; password: demo")
    print("Ctrl+C to stop.\n")

    def stop(*_args):
        for _name, _port, proc in procs:
            proc.terminate()
        for _name, _port, proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while True:
        for name, port, proc in procs:
            if proc.poll() is not None:
                print(f"{name} exited with code {proc.returncode}; see storage/logs/{name}.log")
                procs.remove((name, port, proc))
        time.sleep(2)


if __name__ == "__main__":
    main()
