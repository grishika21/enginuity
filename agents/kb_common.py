"""What the two knowledge-base agents share: the document catalog and equipment detection.

Each catalog entry and each chunk records the sandbox it was ingested in (see agentkit.sandbox);
the global sample documents have none. Readers only ever see the global entries and their own
sandbox's.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from agentkit.redis_conn import get_redis
from agentkit.sandbox import visible_sandboxes

COLLECTION = "manuals"
_MODEL_RE = re.compile(r"\b([A-Z]{2}-\d{2,3})(?![-\d])")


def equipment_codes(text: str) -> list[str]:
    """Equipment model codes such as KP-200 mentioned in a text, most frequent first."""
    counts = Counter(m.group(1) for m in _MODEL_RE.finditer(text or ""))
    return [code for code, _n in counts.most_common()]


def save_doc(doc_id: str, meta: dict) -> None:
    get_redis().hset(f"kb:docs:{COLLECTION}", doc_id, json.dumps(meta))


def get_doc(doc_id: str) -> dict | None:
    raw = get_redis().hget(f"kb:docs:{COLLECTION}", doc_id)
    return json.loads(raw) if raw else None


def all_docs() -> dict[str, dict]:
    return {k: json.loads(v) for k, v in get_redis().hgetall(f"kb:docs:{COLLECTION}").items()}


def visible_docs(user: str) -> dict[str, dict]:
    """The catalog entries this user may search: the global ones and their own sandbox's."""
    visible = visible_sandboxes(user)
    return {k: v for k, v in all_docs().items() if v.get("sandbox", "") in visible}


def purge_sandbox(sandbox: str) -> int:
    """Remove every document ingested in a sandbox from the index and the catalog."""
    from agentkit.vectors import get_store

    if not sandbox:
        raise ValueError("the global sample knowledge base is never purged")
    gone = [doc_id for doc_id, meta in all_docs().items() if meta.get("sandbox") == sandbox]
    for doc_id in gone:
        get_store().delete_doc(COLLECTION, doc_id)
        get_redis().hdel(f"kb:docs:{COLLECTION}", doc_id)
    return len(gone)
