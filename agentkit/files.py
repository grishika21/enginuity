"""File storage: a local stand-in for blob storage.

Files are addressed by opaque ids, never by path. Every read checks the owner (files owned by
`library` are readable by everyone), and links handed to the browser are short-lived HMAC-signed
URLs, so an internal path never leaves the system.
"""

from __future__ import annotations

import hashlib
import hmac
import mimetypes
import re
import time
import uuid
from pathlib import Path

from .config import get_settings
from .contract import AgentError, FileRef
from .redis_conn import get_redis

LIBRARY = "library"
_SAFE_NAME = re.compile(r"[^\w.\- ()]+")


def safe_name(name: str) -> str:
    base = Path(name or "file").name
    cleaned = _SAFE_NAME.sub("_", base).strip(" .") or "file"
    return cleaned[:120]


class FileStore:
    def __init__(self, root: Path | None = None, redis_client=None):
        self.root = (root or get_settings().storage_dir / "files").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.r = redis_client or get_redis()

    def put(self, owner: str, name: str, data: bytes, kind: str = "upload") -> FileRef:
        file_id = f"f-{uuid.uuid4().hex[:16]}"
        name = safe_name(name)
        rel = Path(owner) / file_id / name
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.r.hset(f"file:{file_id}", mapping={
            "owner": owner, "name": name, "rel": str(rel), "kind": kind, "size": len(data),
            "mime": mimetypes.guess_type(name)[0] or "application/octet-stream", "created": time.time(),
        })
        self.r.sadd(f"files:of:{owner}", file_id)
        return FileRef(file_id=file_id, name=name)

    def meta(self, file_id: str) -> dict:
        if not re.fullmatch(r"f-[0-9a-f]{16}", file_id or ""):
            raise AgentError("Unknown file.")
        meta = self.r.hgetall(f"file:{file_id}")
        if not meta:
            raise AgentError("Unknown file.")
        return meta

    def can_read(self, file_id: str, user: str) -> bool:
        try:
            owner = self.meta(file_id)["owner"]
        except AgentError:
            return False
        return owner in (user, LIBRARY) or self.r.sismember(f"files:shared:{file_id}", user)

    def path(self, file_id: str, user: str) -> Path:
        """Resolve a file for `user`, refusing other users' files and anything outside storage."""
        meta = self.meta(file_id)
        if not self.can_read(file_id, user):
            raise AgentError("You do not have access to that file.")
        path = (self.root / meta["rel"]).resolve()
        if self.root not in path.parents:
            raise AgentError("You do not have access to that file.")
        if not path.is_file():
            raise AgentError("That file is no longer available.")
        return path

    def share(self, file_id: str, with_user: str) -> None:
        self.r.sadd(f"files:shared:{file_id}", with_user)

    def list_for(self, user: str, include_library: bool = True) -> list[dict]:
        ids = set(self.r.smembers(f"files:of:{user}"))
        if include_library:
            ids |= set(self.r.smembers(f"files:of:{LIBRARY}"))
        out = []
        for file_id in ids:
            meta = self.r.hgetall(f"file:{file_id}")
            if meta:
                out.append({"file_id": file_id, "name": meta["name"], "owner": meta["owner"],
                            "kind": meta.get("kind", ""), "size": int(meta.get("size", 0)),
                            "created": float(meta.get("created", 0))})
        return sorted(out, key=lambda m: (m["owner"] != user, -m["created"]))


# ---- signed links --------------------------------------------------------------------------


def _sig(file_id: str, user: str, exp: int) -> str:
    key = get_settings().jwt_secret.encode()
    return hmac.new(key, f"{file_id}:{user}:{exp}".encode(), hashlib.sha256).hexdigest()[:32]


def sign_link(file_id: str, user: str, ttl_s: int | None = None) -> str:
    exp = int(time.time()) + (ttl_s or get_settings().link_ttl_s)
    return f"/files/{file_id}?u={user}&exp={exp}&sig={_sig(file_id, user, exp)}"


def verify_link(file_id: str, user: str, exp: int, sig: str) -> bool:
    if exp < int(time.time()):
        return False
    return hmac.compare_digest(_sig(file_id, user, exp), sig)
