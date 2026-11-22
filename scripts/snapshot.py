"""Export the seeded sample data to demo_snapshot/ (used by the hosted Streamlit demo).

Run against a seeded local stack:  python scripts/snapshot.py
It also builds the PI agent's PageIndex tree for the sample process instruction, so the hosted
demo never has to run Docling for the samples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentkit.docparse import parse  # noqa: E402
from agentkit.files import LIBRARY, FileStore  # noqa: E402
from agentkit.redis_conn import get_redis  # noqa: E402
from agentkit.snapshot import export  # noqa: E402
from agents.pi_agent.tree import build_tree, to_dicts  # noqa: E402


def main() -> int:
    store, r = FileStore(), get_redis()
    for meta in store.list_for(LIBRARY):
        if meta["owner"] == LIBRARY and meta["name"].startswith("PI-") and not r.exists(f"pi:tree:{meta['file_id']}"):
            doc = parse(store.path(meta["file_id"], LIBRARY), meta["name"])
            r.set(f"pi:tree:{meta['file_id']}", json.dumps(to_dicts(build_tree(doc))))
            print(f"built the PageIndex tree for {meta['name']}")
    print("snapshot:", export())
    return 0


if __name__ == "__main__":
    sys.exit(main())
