"""Settings shared by the orchestrator and every agent.

Values come from environment variables, optionally loaded from a `.env` file at the repo root.
Nothing here holds a real secret: the defaults are for a local demo only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ without overriding values already set."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    redis_url: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    jwt_secret: str
    storage_dir: Path
    data_dir: Path
    qdrant_url: str | None
    embed_model: str
    heartbeat_s: int
    orchestrator_url: str
    link_ttl_s: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    storage = Path(os.environ.get("STORAGE_DIR", ROOT / "storage")).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    return Settings(
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/5"),
        llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_api_key=os.environ.get("LLM_API_KEY", "ollama"),
        llm_model=os.environ.get("LLM_MODEL", "qwen3:4b-instruct-2507-q4_K_M"),
        jwt_secret=os.environ.get("JWT_SECRET", "local-demo-secret-change-me"),
        storage_dir=storage,
        data_dir=Path(os.environ.get("DATA_DIR", ROOT / "data")).resolve(),
        qdrant_url=os.environ.get("QDRANT_URL") or None,
        embed_model=os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        heartbeat_s=int(os.environ.get("HEARTBEAT_S", "10")),
        orchestrator_url=os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000"),
        link_ttl_s=int(os.environ.get("LINK_TTL_S", "600")),
    )
