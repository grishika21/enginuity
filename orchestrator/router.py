"""The router: an LLM tool call that picks one agent per request.

Tools are built at request time from the live registry, so a new agent becomes routable as soon
as it registers. The router sees every routable agent so it can recognise what a request needs;
whether this user may use that agent is decided afterwards in code (`Route.allowed`), never by
the model. The router never answers domain questions itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agentkit.auth import allowed
from agentkit.contract import AgentCard
from agentkit.llm import LLM
from agentkit.safety import USER_CONTENT_RULE, wrap_user

DIRECT = "reply_directly"

SYSTEM = (
    "You are the router of FieldDesk, an assistant for equipment documentation. For every user "
    "message, call exactly one tool.\n"
    "- Call an agent tool for any request about the equipment, its documents, procedures, parts or "
    "values. Pass the request in full, resolving references like \"it\" or \"that pump\" from the "
    "conversation.\n"
    "- A question (how, how often, what, which, why) goes to the agent that answers questions. Only "
    "an explicit request to write, draft or create a document goes to the documentation author.\n"
    "- Call reply_directly only for greetings, thanks, questions about what you can do, or when no "
    "agent fits the request. Never answer an equipment question yourself.\n"
    "- When the request is about a specific document, set `document` to its exact name from the list.\n"
    + USER_CONTENT_RULE
)


def tool_name(card: AgentCard) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", card.name)


def build_tools(cards: list[AgentCard], documents: list[str]) -> list[dict[str, Any]]:
    tools = []
    for card in cards:
        if not card.routable:
            continue
        skills = "; ".join(f"{s.name}: {s.description}" for s in card.skills)
        props: dict[str, Any] = {"request": {"type": "string", "description": "The full request for the agent."}}
        if card.document_input and documents:
            props["document"] = {"type": "string", "enum": documents,
                                 "description": "The document the request is about."}
        tools.append({"type": "function", "function": {
            "name": tool_name(card), "description": f"{card.description} Skills: {skills}"[:1000],
            "parameters": {"type": "object", "properties": props, "required": ["request"]}}})
    tools.append({"type": "function", "function": {
        "name": DIRECT,
        "description": "Reply without calling an agent: greetings, thanks, what you can do, or no agent fits.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}})
    return tools


def resolve_tool(name: str, tools: list[dict[str, Any]]) -> str | None:
    """Map a tool name the model made up (e.g. "authors_a_new_document") to a real one by the words
    it shares with the start of each tool's description. None when nothing matches clearly."""
    offered = {t["function"]["name"] for t in tools}
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", name or "")
    if normalized in offered:
        return normalized
    words = set(re.findall(r"[a-z]{3,}", name.lower())) - {"the", "and", "agent", "tool"}
    best, best_score = None, 0
    for t in tools:
        lead = set(re.findall(r"[a-z]{3,}", " ".join(t["function"]["description"].lower().split()[:12])))
        score = len(words & (lead | set(re.findall(r"[a-z]{3,}", t["function"]["name"].lower()))))
        if score > best_score:
            best, best_score = t["function"]["name"], score
    return best if best_score >= 2 else None


@dataclass
class Route:
    card: AgentCard | None
    request: str
    document: str | None
    reply: str
    allowed: bool = True


def route(llm: LLM, cards: list[AgentCard], roles: tuple[str, ...], message: str,
          recent: list[dict[str, str]], documents: list[str]) -> Route:
    tools = build_tools(cards, documents)
    by_tool = {tool_name(c): c for c in cards}
    offered = {t["function"]["name"] for t in tools}
    doc_line = ("Documents available: " + "; ".join(documents)) if documents else "No documents are attached."
    messages = [{"role": "system", "content": f"{SYSTEM}\n\n{doc_line}"}]
    for m in recent[-6:]:
        messages.append({"role": m["role"], "content": m["text"][:1200]})
    messages.append({"role": "user", "content": wrap_user(message)})
    name, args, text = llm.tool_call(messages, tools)
    if name and name not in offered:
        name = resolve_tool(name, tools)
        if name is None:  # one more try; a made-up tool is never called
            name, args, text = llm.tool_call(messages, tools)
            name = resolve_tool(name, tools) if name else None
    if name and name in offered and name != DIRECT:
        card = by_tool[name]
        doc = args.get("document")
        return Route(card, str(args.get("request") or message), doc if doc in documents else None, "",
                     allowed=allowed(roles, card.allowed_roles))
    reply = str(args.get("text") or text or "").strip()
    return Route(None, message, None, reply or "I can help with equipment questions, procedures and documents.")
