"""The wire contract every agent speaks.

An A2A-style job API: the caller starts a job, follows its event stream, and answers a question
when the job pauses with `input_required`. A reply goes back as a new invoke with the same
`session_id`, which resumes the paused work.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["pending", "running", "input_required", "completed", "failed", "cancelled"]
TERMINAL = {"input_required", "completed", "failed", "cancelled"}

MAX_CALL_CHAIN = 5


class InvokeRequest(BaseModel):
    message: str = Field(..., max_length=8000)
    session_id: str | None = Field(default=None, max_length=200)
    parameters: dict[str, Any] = Field(default_factory=dict)
    files: list[str] = Field(default_factory=list, max_length=10)
    call_chain: list[str] = Field(default_factory=list)


class Question(BaseModel):
    text: str
    options: list[str] = Field(default_factory=list)
    kind: str = "input"  # input | choice | approval


class FileRef(BaseModel):
    file_id: str
    name: str


class Job(BaseModel):
    job_id: str
    session_id: str
    agent: str
    owner: str
    status: JobStatus = "pending"
    stage: str = ""
    message: str = ""
    question: Question | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    files: list[FileRef] = Field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0


class Skill(BaseModel):
    id: str
    name: str
    description: str
    requires_approval: bool = False


class AgentCard(BaseModel):
    name: str
    title: str
    description: str
    version: str
    url: str
    skills: list[Skill]
    allowed_roles: list[str]
    routable: bool = True  # False: only other agents may call it, never the orchestrator's router
    document_input: bool = False  # the agent works on a document the user picks


# ---- results a handler returns -------------------------------------------------------------


class Completed(BaseModel):
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    files: list[FileRef] = Field(default_factory=list)


class NeedsInput(BaseModel):
    question: Question
    data: dict[str, Any] = Field(default_factory=dict)
    files: list[FileRef] = Field(default_factory=list)


class AgentError(Exception):
    """A failure whose message is safe to show to the user."""
