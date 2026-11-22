"""A snapshot of the seeded sample data, so a fresh host starts with a ready knowledge base.

`scripts/snapshot.py export` writes `demo_snapshot/` from a seeded local stack: the library files,
the vector index, the PI agent's cached PageIndex tree and the Redis records that point at them.
`restore()` puts it back on an empty host in a second, instead of running Docling at every start.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import ROOT, get_settings
from .redis_conn import get_redis

SNAPSHOT = ROOT / "demo_snapshot"
MARKER = "kb:docs:manuals"


def export(redis_client=None) -> dict:
    r = redis_client or get_redis()
    storage = get_settings().storage_dir
    library_ids = sorted(r.smembers("files:of:library"))
    if not library_ids or not r.exists(MARKER):
        raise RuntimeError("Seed the stack first (scripts/seed.py).")
    if SNAPSHOT.exists():
        shutil.rmtree(SNAPSHOT)
    (SNAPSHOT / "files").mkdir(parents=True)
    keys: dict[str, dict] = {"files:of:library": {"type": "set", "value": library_ids},
                             MARKER: {"type": "hash", "value": r.hgetall(MARKER)}}
    for file_id in library_ids:
        keys[f"file:{file_id}"] = {"type": "hash", "value": r.hgetall(f"file:{file_id}")}
        shutil.copytree(storage / "files" / "library" / file_id, SNAPSHOT / "files" / "library" / file_id)
        tree = r.get(f"pi:tree:{file_id}")
        if tree:
            keys[f"pi:tree:{file_id}"] = {"type": "string", "value": tree}
    shutil.copytree(storage / "vectors" / "manuals", SNAPSHOT / "vectors" / "manuals",
                    ignore=shutil.ignore_patterns(".lock", "*.tmp*"))
    (SNAPSHOT / "redis.json").write_text(json.dumps(keys, indent=1), encoding="utf-8")
    return {"files": len(library_ids), "keys": len(keys)}


def restore(redis_client=None) -> bool:
    """Load the snapshot when this host has no knowledge base yet. Returns True if it loaded."""
    r = redis_client or get_redis()
    if r.exists(MARKER) or not (SNAPSHOT / "redis.json").exists():
        return False
    storage = get_settings().storage_dir
    shutil.copytree(SNAPSHOT / "files", storage / "files", dirs_exist_ok=True)
    shutil.copytree(SNAPSHOT / "vectors", storage / "vectors", dirs_exist_ok=True)
    for key, item in json.loads((SNAPSHOT / "redis.json").read_text(encoding="utf-8")).items():
        if item["type"] == "hash":
            r.hset(key, mapping=item["value"])
        elif item["type"] == "set":
            r.sadd(key, *item["value"])
        else:
            r.set(key, item["value"])
    return True
