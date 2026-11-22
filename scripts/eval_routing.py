"""Routing evaluation: does the router send each request to the right agent?

Runs labelled prompts against the configured LLM, using the agent cards from the code (no agents
need to be running). Prints accuracy and every miss.

    python scripts/eval_routing.py            # one pass
    python scripts/eval_routing.py --runs 3   # repeat to see how stable the routing is
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentkit.llm import get_llm  # noqa: E402
from agents.docs_agent.app import CARD as DOCS  # noqa: E402
from agents.kb_extraction.app import CARD as KBE  # noqa: E402
from agents.kb_ingestion.app import CARD as KBI  # noqa: E402
from agents.pi_agent.app import CARD as PI  # noqa: E402
from orchestrator.router import route  # noqa: E402

DOCUMENTS = ["KP-200_Maintenance_Manual.pdf", "KC-50_Maintenance_Manual.pdf", "PI-0042_KP-200_Pump_Assembly.pdf",
             "SB-07_KP-200_Seal_Bulletin_scanned.pdf", "Travel_Policy_2026.pdf", "KV-80_Control_Valve_Manual.pdf"]

CASES = [
    ("How do I replace the mechanical seal on the KP-200?", "kb-extraction-agent"),
    ("How often should the bearings be lubricated?", "kb-extraction-agent"),
    ("What torque does the KP-200 impeller nut need?", "kb-extraction-agent"),
    ("Which part number is the KC-50 air filter?", "kb-extraction-agent"),
    ("What does service bulletin SB-07 change?", "kb-extraction-agent"),
    ("How do I change the oil on the KC-50?", "kb-extraction-agent"),
    ("Write a maintenance procedure for the KP-200 mechanical seal replacement", "docs-agent"),
    ("Draft a system description of the KC-50 compressor", "docs-agent"),
    ("Create a task card for lubricating the KP-200 bearings", "docs-agent"),
    ("Prepare a procedure document for the KC-50 filter change", "docs-agent"),
    ("Extract the mechanical seal and impeller fitting steps from PI-0042", "pi-agent"),
    ("Pull out the testing requirements from the PI-0042 process instruction into Word", "pi-agent"),
    ("Add the KC-50 manual to the knowledge base", "kb-ingestion-agent"),
    ("Index this document so I can search it", "kb-ingestion-agent", ["KV-80_Control_Valve_Manual.pdf"]),
    # visitors' own documents, outside the equipment domain
    ("What does the travel policy say about hotel costs?", "kb-extraction-agent"),
    ("What are the key points of the KV-80 control valve manual?", "kb-extraction-agent"),
    ("Write a brief of the KP-200 maintenance manual", "docs-agent"),
    ("Draft a one-page overview document of the travel policy", "docs-agent"),
    ("Extract the expense claim section from Travel_Policy_2026.pdf into Word", "pi-agent"),
    ("Add my travel policy to the knowledge base", "kb-ingestion-agent"),
    ("Hi! What can you do?", None),
    ("Thanks, that's all", None),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    llm = get_llm()
    cards = sorted([KBI, KBE, PI, DOCS], key=lambda c: c.name)  # the order the registry returns them in
    total, correct, misses = 0, 0, Counter()
    started = time.time()
    for _ in range(args.runs):
        for prompt, expected, *attached in CASES:  # a third item: the files attached to the message
            decision = route(llm, cards, ("author",), prompt, [], DOCUMENTS, attached[0] if attached else None)
            got = decision.card.name if decision.card else None
            total += 1
            if got == expected:
                correct += 1
            else:
                misses[(prompt, expected, got)] += 1
    print(f"model: {llm.model}")
    print(f"routing accuracy: {correct}/{total} ({100 * correct / total:.0f}%) in {time.time() - started:.0f} s")
    for (prompt, expected, got), n in misses.most_common():
        print(f"  miss x{n}: {prompt!r} expected {expected}, got {got}")
    return 0 if correct == total else 1


if __name__ == "__main__":
    sys.exit(main())
