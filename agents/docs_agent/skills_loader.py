"""Load SKILL.md playbooks: every document type is a folder, not code.

A folder under `skills/` with a valid SKILL.md becomes a document type: its frontmatter declares
the inputs to collect, the section layout and the source query; its body holds the authoring
rules the model follows. Adding a folder and calling /admin/reload-skills adds a document type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger("docs_agent.skills")

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
MAX_DESCRIPTION = 1024


@dataclass
class InputSpec:
    key: str
    ask: str
    check: str | None = None
    default: str | None = None


@dataclass
class SkillSpec:
    name: str
    title: str
    version: str
    description: str
    inputs: list[InputSpec]
    sections: list[str]
    source_query: str
    rules: str
    generated_sections: dict[str, str] = field(default_factory=dict)
    numbered_sections: list[str] = field(default_factory=list)


def parse_skill(path: Path) -> SkillSpec:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError("missing frontmatter")
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front) or {}
    for required in ("name", "description", "sections", "inputs", "source_query"):
        if not meta.get(required):
            raise ValueError(f"missing '{required}'")
    description = " ".join(str(meta["description"]).split())
    if len(description) > MAX_DESCRIPTION:
        raise ValueError(f"description is {len(description)} characters; the limit is {MAX_DESCRIPTION}")
    if meta["name"] != path.parent.name:
        log.warning("skill %s: name '%s' does not match its folder", path.parent.name, meta["name"])
    return SkillSpec(
        name=str(meta["name"]),
        title=str(meta.get("title") or meta["name"]),
        version=str(meta.get("version", "0.0.0")),
        description=description,
        inputs=[InputSpec(**i) for i in meta["inputs"]],
        sections=[str(s) for s in meta["sections"]],
        source_query=str(meta["source_query"]),
        rules=body.strip(),
        generated_sections={str(k): str(v) for k, v in (meta.get("generated_sections") or {}).items()},
        numbered_sections=[str(s) for s in meta.get("numbered_sections") or []],
    )


def load_skills(root: Path = SKILLS_DIR) -> dict[str, SkillSpec]:
    skills: dict[str, SkillSpec] = {}
    for path in sorted(root.glob("*/SKILL.md")):
        try:
            spec = parse_skill(path)
        except Exception as exc:  # a broken skill is skipped, never fatal
            log.error("skipping skill %s: %s", path.parent.name, exc)
            continue
        skills[spec.name] = spec
    log.info("loaded skills: %s", ", ".join(skills) or "none")
    return skills
