---
name: document-brief
title: Document brief
version: 1.0.0
description: >-
  Writes a one-page brief of a document or a topic in the knowledge base, on any subject (a
  manual, a policy, a report, a paper): its purpose and scope, the key points, the values,
  limits and requirements it sets, and its warnings, written only from the sources. Use it when
  the user asks to write, draft, create or prepare a brief, summary document or overview of a
  document or topic. Not for: step-by-step maintenance procedures (maintenance-procedure),
  system descriptions of a piece of equipment (system-description), or answering a question.
inputs:
  - key: topic
    ask: Which document or topic should the brief cover?
sections:
  - Purpose and scope
  - Key points
  - Values, limits and requirements
  - Warnings and caveats
source_query: "{topic}: purpose, scope, key points, requirements, values, limits, warnings"
source_scope: document
---

# Document brief

## Authoring rules

- Write only from the numbered sources, in plain language for a reader who has not seen the
  document. Never add facts, numbers or advice the sources do not give.
- **Purpose and scope**: two or three sentences: what the document is, who it is for and what
  it covers.
- **Key points**: five to eight bullets, the most important things a reader must know, in the
  order the document gives them.
- **Values, limits and requirements**: one bullet per value, limit, interval, deadline or rule,
  exactly as written in the sources (numbers and units unchanged), with what it applies to.
- **Warnings and caveats**: the sources' warnings, cautions and exceptions, as bullets.
- If a section has no source content, write `Not covered by the sources.` and nothing else.
