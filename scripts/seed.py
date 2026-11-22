"""Load the sample documents: put them in the shared library and index the manuals.

Run after `scripts/run_all.py` is up:  python scripts/seed.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentkit.auth import issue  # noqa: E402
from agentkit.client import call_agent  # noqa: E402
from agentkit.contract import InvokeRequest  # noqa: E402
from agentkit.files import LIBRARY, FileStore  # noqa: E402
from agentkit.registry import find_agent  # noqa: E402
from agents.docs_agent.tools import library_dir  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
TO_INDEX = ["KP-200_Maintenance_Manual.pdf", "KC-50_Maintenance_Manual.pdf", "SB-07_KP-200_Seal_Bulletin_scanned.pdf"]


def main() -> int:
    if not SAMPLES.exists():
        import make_samples

        make_samples.main()
    store = FileStore()
    existing = {m["name"]: m["file_id"] for m in store.list_for(LIBRARY) if m["owner"] == LIBRARY}
    ids = {}
    for path in sorted(SAMPLES.iterdir()):
        ids[path.name] = existing.get(path.name) or store.put(LIBRARY, path.name, path.read_bytes(), kind="library").file_id
        print(f"library: {path.name} ({ids[path.name]})")
    library_dir()  # copy the published-documents library into storage on first run

    card = None
    for _ in range(30):
        card = find_agent("kb-ingestion-agent")
        if card:
            break
        time.sleep(1)
    if card is None:
        print("kb-ingestion-agent is not running. Start scripts/run_all.py first.")
        return 1
    token = issue("seed", ["admin"])
    for name in TO_INDEX:
        print(f"indexing {name} ...", flush=True)
        job = call_agent(card, InvokeRequest(message=f"Add {name} to the knowledge base", files=[ids[name]]), token,
                         on_event=lambda e: print(f"   {e.get('stage')}: {e.get('detail', '')}"))
        print(f"   -> {job.status}: {job.message}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
