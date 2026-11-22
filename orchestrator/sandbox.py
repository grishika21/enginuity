"""Reset one demo sandbox: delete everything its visitor created, keep the shared samples.

The orchestrator is the entry point the visitor talks to, so it owns the reset. It removes the
sandbox personas' chats and files, the documents they added to the knowledge base and their copy
of the published-documents library. The global sample data is never touched.
"""

from __future__ import annotations

from agentkit.files import FileStore
from agentkit.redis_conn import get_redis
from agentkit.sandbox import scoped, valid_sandbox
from agents.docs_agent.tools import drop_library
from agents.kb_common import purge_sandbox
from orchestrator.history import HistoryStore


def reset(sandbox: str, personas: list[str], files: FileStore, history: HistoryStore) -> dict:
    if not valid_sandbox(sandbox):
        raise ValueError("invalid sandbox id")
    owners = {scoped(p, sandbox) for p in personas}
    removed_files: list[str] = []
    for owner in owners:
        removed_files += files.purge_owner(owner)
    r = get_redis()
    for file_id in removed_files:  # outlines the extraction agent cached for those files
        r.delete(f"pi:tree:{file_id}")
    return {"chats": history.purge_owners(owners), "files": len(removed_files),
            "kb_documents": purge_sandbox(sandbox), "library": drop_library(sandbox)}
