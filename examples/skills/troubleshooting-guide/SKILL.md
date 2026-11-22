---
name: troubleshooting-guide
title: Troubleshooting guide
version: 1.0.0
description: >-
  Authors a troubleshooting guide for one piece of equipment: symptoms, likely causes and the
  corrective action for each, written only from the knowledge base. Use it when the user asks
  for a troubleshooting guide, fault-finding table or "what to check when" document. Not for:
  a single maintenance procedure (maintenance-procedure) or a system description.
inputs:
  - key: equipment
    ask: Which equipment model is the troubleshooting guide for?
    check: equipment
sections:
  - Scope
  - Symptoms, causes and actions
  - When to escalate
source_query: "{equipment} troubleshooting: symptoms, causes, corrective actions, limits"
---

# Troubleshooting guide

## Authoring rules

- Write only from the numbered sources.
- **Symptoms, causes and actions**: a Markdown table with the columns Symptom | Likely cause |
  Action, one row per symptom in the sources. Refer to procedures by their section number.
- **When to escalate**: only conditions the sources give (limits, temperatures, leak rates).
