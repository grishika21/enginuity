"""The one rule every approval uses: only a clear, unqualified yes counts."""

from __future__ import annotations

import re

_YES = {"yes", "y", "yes please", "yes go ahead", "go ahead", "yes do it", "confirm", "yes confirm",
        "approve", "approved", "yes publish", "yes publish it", "publish"}


def is_explicit_yes(reply: str) -> bool:
    """"yes" approves; "yes but change X", "ok?", "maybe", "not yet" do not."""
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", (reply or "").lower())).strip()
    return normalized in _YES
