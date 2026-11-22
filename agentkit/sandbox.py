"""Per-visitor sandboxes for the public demo.

On the hosted demo every browser session gets a random sandbox id, and each demo persona signs in
as `<persona>~<sandbox>` (for example `asha~3f9a1c2b7d4e`). Everything already keyed by user
(chats, uploads, outputs, jobs) is then private to that visitor without further changes. The
knowledge base and the published-documents library are shared *within* a sandbox, so Asha's
uploads become searchable for Ravi in the same session, and never for another visitor.

A user name without `~` (the local web UI, the seed script) is in the global sandbox "", which
holds the sample knowledge base that every sandbox can read.
"""

from __future__ import annotations

import re
import uuid

SEP = "~"
_SANDBOX = re.compile(r"[a-z0-9]{6,32}")


def new_sandbox() -> str:
    return uuid.uuid4().hex[:12]


def valid_sandbox(value: str | None) -> bool:
    return bool(value) and bool(_SANDBOX.fullmatch(value or ""))


def scoped(persona: str, sandbox: str | None) -> str:
    """The user id a persona signs in with inside a sandbox."""
    if not sandbox:
        return persona
    if not valid_sandbox(sandbox):
        raise ValueError("invalid sandbox id")
    return f"{persona}{SEP}{sandbox}"


def sandbox_of(user: str) -> str:
    """The sandbox a user id belongs to; "" for the global one."""
    if SEP not in user:
        return ""
    sandbox = user.split(SEP, 1)[1]
    if not valid_sandbox(sandbox):  # ids come from signed tokens, so this means tampering or a bug
        raise ValueError("invalid sandbox in user id")
    return sandbox


def persona(user: str) -> str:
    return user.split(SEP, 1)[0]


def visible_sandboxes(user: str) -> set[str]:
    """Shared data a user may read: the global sample data plus their own sandbox's."""
    return {"", sandbox_of(user)}
