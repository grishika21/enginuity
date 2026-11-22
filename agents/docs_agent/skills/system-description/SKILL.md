---
name: system-description
title: System description
version: 1.0.0
description: >-
  Authors a system description for one piece of equipment or one of its systems: what it is,
  its components, controls and indicating, interfaces and power supply, written only from the
  knowledge base. Use it when the user asks to describe how equipment or a system works, or for
  a "system description" or "description section". Not for: step-by-step procedures
  (maintenance-procedure), answering a single question, or extracting from a process instruction.
inputs:
  - key: equipment
    ask: Which equipment model should the description cover?
    check: equipment
  - key: system
    ask: Which system should it describe? Reply "whole unit" for the complete equipment.
    default: whole unit
sections:
  - General
  - Components
  - Controls and indicating
  - Interfaces
  - Power supply
source_query: "{equipment} {system}: general description, components, controls and indicating, interfaces, power supply"
---

# System description

## Authoring rules

- Write only from the numbered sources, in plain descriptive prose (present tense, no steps).
- **Components**: one bullet per component with its part number in brackets, exactly as in
  the sources.
- Keep every value exactly as written in the sources. Never add a value the sources do not give.
- If a section has no source content, write `Not covered by the sources.` and nothing else.
