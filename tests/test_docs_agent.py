import json
import shutil

import pytest

from agentkit.config import get_settings
from agentkit.contract import AgentError
from agents.docs_agent import tools
from agents.docs_agent.skills_loader import SKILLS_DIR, load_skills, parse_skill


def test_every_shipped_skill_loads(tmp_path):
    skills = load_skills()
    assert {"maintenance-procedure", "system-description"} <= set(skills)
    proc = skills["maintenance-procedure"]
    assert proc.sections[-1] == "Close-up and test"
    assert proc.generated_sections == {"Parts required": "parts_table"}


def test_example_skill_is_valid_so_it_can_be_dropped_in():
    spec = parse_skill(SKILLS_DIR.parent.parent.parent / "examples" / "skills" / "troubleshooting-guide" / "SKILL.md")
    assert spec.name == "troubleshooting-guide"


def test_broken_skill_is_skipped_not_fatal(tmp_path):
    shutil.copytree(SKILLS_DIR / "maintenance-procedure", tmp_path / "maintenance-procedure")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text("---\nname: broken\n---\nno description")
    long = tmp_path / "too-long"
    long.mkdir()
    text = (SKILLS_DIR / "maintenance-procedure" / "SKILL.md").read_text()
    long.joinpath("SKILL.md").write_text(text.replace("name: maintenance-procedure", "name: too-long")
                                         .replace("description: >-", "description: >-\n  " + "x" * 1100))
    assert list(load_skills(tmp_path)) == ["maintenance-procedure"]


def test_split_sections_maps_headings_and_marks_missing():
    md = "## 1. Purpose\nReplace the seal.\n### Safety\n- Lock out.\n## Extra notes\nmore safety\n## Procedure\n- Do it"
    out = tools.split_sections(md, ["Purpose", "Safety precautions", "Procedure", "Close-up and test"])
    assert out["Purpose"] == "Replace the seal."
    assert "Lock out" in out["Safety precautions"] and "more safety" in out["Safety precautions"]
    assert out["Close-up and test"] == tools.NOT_COVERED


def test_steps_are_numbered_in_code():
    assert tools.number_steps("- Remove the guard\n3. Remove bolts\nStep 7: Refit") == \
        "1. Remove the guard\n2. Remove bolts\n3. Refit"


def test_part_numbers_are_checked_against_the_catalogue():
    text, ok, bad = tools.check_parts("Fit seal KP-200-114B, O-ring KP-200-115, filter KC-50-210, bolt KP-200-999.",
                                      "KP-200")
    assert ok == ["KP-200-114B", "KP-200-115"]
    assert bad == ["KC-50-210", "KP-200-999"]  # other equipment's part, and an invented one
    assert "KP-200-999 [unverified part number]" in text


def test_part_names_get_their_numbers_but_shared_names_are_not_guessed():
    text = tools.annotate_part_names("Fit a new casing O-ring and the shaft sleeve. Fit the mechanical seal assembly.",
                                     "KP-200")
    assert "casing O-ring (KP-200-115)" in text
    assert "shaft sleeve (KP-200-111)" in text
    assert "mechanical seal assembly." in text  # KP-200-114 and -114B share the name: left alone
    assert tools.annotate_part_names("Fit the shaft sleeve (KP-200-111).", "KP-200").count("KP-200-111") == 1


def test_parts_table_comes_from_the_catalogue():
    table = tools.parts_table(["KP-200-115"])
    assert "Casing O-ring, 210 x 4 mm" in table and tools.parts_table([]) == tools.NOT_COVERED


def test_equipment_check_normalises():
    assert tools.check_equipment("the kp200 pump") == "KP-200"
    assert tools.check_equipment("KX-9") is None


def test_impact_finds_documents_sharing_parts():
    affected = tools.impact(["KP-200-115", "KP-200-114"], "KP-200")
    assert affected[0]["id"] == "TP-101" and affected[0]["shared_parts"] == ["KP-200-114", "KP-200-115"]
    assert {"TP-101", "TP-102"} <= {d["id"] for d in affected}


def test_review_approval_words():
    assert tools.is_approval("Approve") and tools.is_approval("looks good")
    assert not tools.is_approval("change step 2")


def test_publish_writes_library_and_graph(monkeypatch):
    monkeypatch.delenv("DOCS_SIMULATE_GRAPH_FAILURE", raising=False)
    doc_id = tools.publish("KP-200 Test - Maintenance procedure", "KP-200", ["KP-200-115"], "# body\n")
    lib = get_settings().storage_dir / "library"
    assert (lib / f"{doc_id}.md").exists()
    assert json.loads((lib / "graph.json").read_text())["documents"][doc_id]["parts"] == ["KP-200-115"]


def test_publish_rolls_back_when_the_graph_write_fails(monkeypatch):
    lib = tools.library_dir()
    before_files = sorted(p.name for p in lib.iterdir())
    before_graph = (lib / "graph.json").read_text()
    monkeypatch.setenv("DOCS_SIMULATE_GRAPH_FAILURE", "1")
    with pytest.raises(AgentError, match="nothing was published"):
        tools.publish("KP-200 Fail - Maintenance procedure", "KP-200", ["KP-200-115"], "# body\n")
    assert sorted(p.name for p in lib.iterdir()) == before_files
    assert (lib / "graph.json").read_text() == before_graph


def test_docx_export_handles_headings_lists_and_tables():
    data = tools.to_docx("# T\n\n## 1. A\n\n- bullet\n1. step\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert data[:2] == b"PK"  # a .docx is a zip file


def test_brief_skill_reads_the_attached_document(tmp_path):
    brief = load_skills()["document-brief"]
    assert brief.source_scope == "document" and [i.key for i in brief.inputs] == ["topic"]
    assert load_skills()["maintenance-procedure"].source_scope == "knowledge_base"
    (tmp_path / "bad").mkdir()
    text = (SKILLS_DIR / "document-brief" / "SKILL.md").read_text().replace("name: document-brief", "name: bad")
    (tmp_path / "bad" / "SKILL.md").write_text(text.replace("source_scope: document", "source_scope: web"))
    assert load_skills(tmp_path) == {}  # an unknown scope is rejected, not guessed


def test_a_draft_about_no_equipment_skips_part_checks():
    from agents.docs_agent import graph

    state = {"skill": "document-brief", "inputs": {"topic": "the travel policy"},
             "sources": [{"n": 1, "title": "Travel policy", "page": 1, "heading": "Scope", "text": "..."}]}
    raw = ("## Purpose and scope\nCovers business travel.\n## Key points\n- Book via the portal (ref KP-200-999).\n"
           "## Values, limits and requirements\n- Hotel cap 150 EUR per night.\n")
    out = graph._assemble(state, raw)
    assert out["title"] == "The travel policy - Document brief"
    assert "KP-200-999 [unverified" not in out["draft"] and out["verified_parts"] == []
    assert "**Equipment:**" not in out["draft"]
    assert 'Section "Warnings and caveats" has no source content.' in out["open_items"]


def test_a_brief_of_an_attached_document_is_titled_after_the_document():
    from agents.docs_agent import graph

    state = {"skill": "document-brief", "inputs": {"topic": "KV-80_Control_Valve_Manual.pdf"}, "files": ["f-1"],
             "sources": [{"n": 1, "title": "Kestrel KV-80 Pneumatic Control Valve", "page": 1, "heading": "1", "text": "."}]}
    out = graph._assemble(state, "## Purpose and scope\nA valve.\n")
    assert out["title"] == "Kestrel KV-80 Pneumatic Control Valve - Document brief"
