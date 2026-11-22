"""Conversations saved per user (JSON files standing in for blob storage), with sharing."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from agentkit.config import get_settings

_LOCK = threading.Lock()


class HistoryStore:
    def __init__(self, root: Path | None = None):
        self.root = root or get_settings().storage_dir / "history"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, conv_id: str) -> Path:
        if not re.fullmatch(r"c-[0-9a-f]{12}", conv_id or ""):
            raise KeyError(conv_id)
        return self.root / f"{conv_id}.json"

    def create(self, owner: str, title: str) -> dict[str, Any]:
        conv = {"id": f"c-{uuid.uuid4().hex[:12]}", "owner": owner, "members": [], "title": title[:60] or "New chat",
                "created": time.time(), "updated": time.time(), "pending": None, "files": [], "messages": []}
        self.save(conv)
        return conv

    def get(self, conv_id: str) -> dict[str, Any]:
        path = self._path(conv_id)
        if not path.exists():
            raise KeyError(conv_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, conv: dict[str, Any]) -> None:
        conv["updated"] = time.time()
        path = self._path(conv["id"])
        with _LOCK:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(conv, indent=1), encoding="utf-8")
            os.replace(tmp, path)

    def can_read(self, conv: dict[str, Any], user: str) -> bool:
        return conv["owner"] == user or user in conv.get("members", [])

    def list_for(self, user: str) -> list[dict[str, Any]]:
        out = []
        for path in self.root.glob("c-*.json"):
            try:
                conv = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if self.can_read(conv, user):
                out.append({"id": conv["id"], "title": conv["title"], "owner": conv["owner"],
                            "updated": conv["updated"], "shared": conv["owner"] != user})
        return sorted(out, key=lambda c: -c["updated"])

    def delete(self, conv_id: str) -> None:
        self._path(conv_id).unlink(missing_ok=True)
