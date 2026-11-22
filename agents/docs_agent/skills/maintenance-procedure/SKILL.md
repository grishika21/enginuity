---
name: maintenance-procedure
title: Maintenance procedure
version: 1.0.0
description: >-
  Authors a step-by-step maintenance procedure (task card) for ONE task on ONE piece of
  equipment, such as replacing a seal, changing the oil or lubricating bearings, written only
  from the knowledge base. Use it to create, draft, write or update a procedure, even when the
  word "procedure" is not used. Not for: describing how a system works (system-description),
  answering a question, or extracting sections from a process instruction.
inputs:
  - key: equipment
    ask: Which equipment model is the procedure for?
    check: equipment
  - key: task
    ask: Which maintenance task should the procedure cover (for example "mechanical seal replacement")?
sections:
  - Purpose
  - Safety precautions
  - Tools and consumables
  - Parts required
  - Procedure
  - Close-up and test
generated_sections:
  Parts required: parts_table
numbered_sections:
  - Procedure
source_query: "{task} on the {equipment}: steps, torque values, settings, parts, safety warnings, tests"
---

# Maintenance procedure

## Authoring rules

- Write only from the numbered sources. Never invent a value (torque, length, interval,
  temperature, quantity). If a value the step needs is not in the sources, write
  `[value not in sources]` in its place.
- **Procedure**: one action per step, in the imperative ("Remove...", "Torque..."), in the
  order the sources give. Do not number the steps yourself; write one step per line starting
  with "- ". The steps are numbered in code.
- Name each part once with its part number in brackets, exactly as written in the sources,
  for example "mechanical seal assembly (KP-200-114)".
- If a service bulletin in the sources supersedes a part or a value, use the bulletin's part
  or value and say "per SB-xx" in that step.
- **Safety precautions**: the sources' warnings first, as bullet points.
- **Tools and consumables**: only what the sources mention.
- Leave **Parts required** empty: it is generated from the parts catalogue.
- **Close-up and test**: the acceptance test and its limits, from the sources.
