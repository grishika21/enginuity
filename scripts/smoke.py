"""Drive the running stack through the orchestrator API, exactly as the UI does.

    python scripts/smoke.py              # all scenarios
    python scripts/smoke.py qa pi        # some of them

Scenarios: qa, ambiguous, pi, author, viewer, approval-gate.
"""

from __future__ import annotations

import json
import sys
import textwrap

import httpx

BASE = "http://localhost:8000"


class Session:
    def __init__(self, user: str):
        self.http = httpx.Client(base_url=BASE, timeout=httpx.Timeout(30, read=900))
        token = self.http.post("/api/login", json={"username": user, "password": "demo"}).json()["token"]
        self.http.headers["Authorization"] = f"Bearer {token}"
        self.user = user
        self.conv: str | None = None

    def file_id(self, name: str) -> str:
        return next(f["file_id"] for f in self.http.get("/api/files").json() if f["name"] == name)

    def say(self, text: str, files: list[str] | None = None) -> dict:
        print(f"\n\033[1m{self.user}>\033[0m {text}")
        res = self.http.post("/api/chat", json={"message": text, "conversation_id": self.conv, "files": files or []})
        res.raise_for_status()
        body = res.json()
        self.conv = body["conversation_id"]
        message = {}
        with self.http.stream("GET", f"/api/turns/{body['job_id']}/stream") as stream:
            for line in stream.iter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:])
                if event["event"] == "progress":
                    detail = f": {event['detail']}" if event.get("detail") else ""
                    print(f"   \033[2m[{event['agent']}] {event['stage']}{detail[:150]}\033[0m")
                elif event["event"] == "message":
                    message = event["message"]
                elif event["event"] == "done":
                    break
        print(textwrap.indent(message.get("text", ""), "   "))
        if message.get("files"):
            print("   files:", ", ".join(f["name"] for f in message["files"]))
        if message.get("question"):
            print("   options:", message["question"].get("options"))
        return message


def qa():
    Session("asha").say("How do I replace the mechanical seal on the KP-200?")


def ambiguous():
    s = Session("asha")
    s.say("How often should the bearings be lubricated?")
    s.say("1")


def pi():
    s = Session("asha")
    s.say("Extract the mechanical seal and impeller fitting steps from PI-0042",
          files=[s.file_id("PI-0042_KP-200_Pump_Assembly.pdf")])


def author():
    s = Session("asha")
    s.say("Write a maintenance procedure for the KP-200 mechanical seal replacement")
    s.say("approve")
    s.say("yes")


def viewer():
    Session("ravi").say("Write a maintenance procedure for the KP-200 seal")


def approval_gate():
    s = Session("asha")
    s.say("Add this document to the knowledge base", files=[s.file_id("PI-0042_KP-200_Pump_Assembly.pdf")])
    s.say("no")


SCENARIOS = {"qa": qa, "ambiguous": ambiguous, "pi": pi, "author": author, "viewer": viewer,
             "approval-gate": approval_gate}

if __name__ == "__main__":
    for name in sys.argv[1:] or SCENARIOS:
        print(f"\n===== {name} =====")
        SCENARIOS[name]()
