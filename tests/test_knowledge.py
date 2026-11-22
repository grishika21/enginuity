from agentkit.docparse import ParsedDoc
from agentkit.vectors import Hit
from agents.kb_common import equipment_codes
from agents.kb_extraction.app import _resolve_choice, ambiguous_equipment, scope_hits
from agents.kb_ingestion.chunking import chunk_recursive, chunk_structural, rule_strategy, sections, doc_stats
from agents.pi_agent.tree import build_tree, iter_nodes, lexical_pick, resolve_sections, select, toc

DOC = ParsedDoc(title="Pump manual", pages=[
    (1, "## 1 General\n\nThe pump moves water.\n\n## 1.1 Components\n\nCasing (KP-200-101) and impeller (KP-200-120)."),
    (2, "## 4 Procedures\n\n## 4.1 Seal Replacement\n\n- 1. Isolate the motor.\n- 2. Remove the seal (KP-200-114).\n"
        "\n## 4.2 Impeller\n\n- 1. Remove the impeller nut."),
    (3, "- 3. Torque the nut to 95 Nm.\n\n## 5 Troubleshooting\n\nCheck the seal."),
])


def test_sections_keep_heading_and_start_page():
    secs = sections(DOC)
    by_heading = {h: p for h, p, _b in secs}
    assert by_heading["1.1 Components"] == 1
    assert by_heading["4.1 Seal Replacement"] == 2
    assert "Torque the nut" in next(b for h, _p, b in secs if h == "4.2 Impeller")  # body runs across the page break


def test_structural_and_recursive_chunks_carry_pages():
    chunks = chunk_structural(DOC)
    assert {c.heading for c in chunks} >= {"4.1 Seal Replacement", "5 Troubleshooting"}
    assert all(c.page in (1, 2, 3) for c in chunks)
    assert all(c.page for c in chunk_recursive(DOC, size=80, overlap=10))


def test_rule_strategy():
    assert rule_strategy(doc_stats(DOC)) == "structural"
    assert rule_strategy({"headings": 0, "avg_section_chars": 0, "chars": 9000}) == "semantic"
    assert rule_strategy({"headings": 1, "avg_section_chars": 100, "chars": 500}) == "recursive"


def test_equipment_codes_ignore_part_numbers():
    assert equipment_codes("KP-200 seal KP-200-114 and KC-50") == ["KP-200", "KC-50"]


def hit(doc, score, equipment, title="t"):
    return Hit(score, {"doc_id": doc, "title": title, "equipment": equipment, "text": "x"})


def test_same_equipment_documents_are_never_ambiguous():
    hits = [hit("manual", 0.71, ["KP-200"]), hit("bulletin", 0.70, ["KP-200"])]
    assert ambiguous_equipment(hits) == []


def test_close_matches_on_different_equipment_ask_which():
    hits = [hit("kp", 0.62, ["KP-200"]), hit("kc", 0.60, ["KC-50"]), hit("far", 0.30, ["XX-10"])]
    options = ambiguous_equipment(hits)
    assert [o["title"] for o in options] == ["KP-200", "KC-50"]
    assert _resolve_choice("2", options) == ["KC-50"]
    assert _resolve_choice("the kp-200 one", options) == ["KP-200"]
    assert _resolve_choice("neither", options) is None


def test_scope_hits_filters_by_equipment():
    hits = [hit("kp", 0.6, ["KP-200"]), hit("kc", 0.6, ["KC-50"])]
    assert [h.payload["doc_id"] for h in scope_hits(hits, {"KC-50"})] == ["kc"]
    assert len(scope_hits(hits, set())) == 2


def test_pageindex_tree_levels_and_page_ranges():
    roots = build_tree(DOC)
    titles = [(n.title, n.level) for n in iter_nodes(roots)]
    assert ("4.1 Seal Replacement", 2) in titles and ("4 Procedures", 1) in titles
    procedures = next(n for n in roots if n.title == "4 Procedures")
    assert [c.title for c in procedures.children] == ["4.1 Seal Replacement", "4.2 Impeller"]
    impeller = procedures.children[1]
    assert (impeller.page_start, impeller.page_end) == (2, 3)
    assert "[0004]" in toc(roots)


def test_select_drops_children_of_a_chosen_parent_and_keeps_order():
    roots = build_tree(DOC)
    ids = {n.title: n.node_id for n in iter_nodes(roots)}
    chosen = select(roots, [ids["4.2 Impeller"], ids["4 Procedures"], ids["1.1 Components"]])
    assert [n.title for n in chosen] == ["1.1 Components", "4 Procedures"]


def test_section_titles_win_over_off_by_one_ids():
    roots = build_tree(DOC)
    ids = {n.title: n.node_id for n in iter_nodes(roots)}
    wrong_id = ids["4.2 Impeller"]  # the model copied the title right but the id of the next section
    picked = resolve_sections(roots, [{"id": wrong_id, "title": "4.1 Seal Replacement"}, "9999", ids["5 Troubleshooting"]])
    assert picked == [ids["4.1 Seal Replacement"], ids["5 Troubleshooting"]]
    copied_line = {"id": wrong_id, "title": "4.1 Seal Replacement (p.2) - Isolate the motor."}
    assert resolve_sections(roots, [copied_line]) == [ids["4.1 Seal Replacement"]]


def test_lexical_fallback_finds_sections_by_title():
    roots = build_tree(DOC)
    picked = select(roots, lexical_pick(roots, "extract the seal replacement steps"))
    assert picked[0].title == "4.1 Seal Replacement"
