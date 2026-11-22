"""What the two knowledge-base agents share: the document catalog and equipment detection."""

from __future__ import annotations

import json
import re
from collections import Counter

from agentkit.redis_conn import get_redis

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
