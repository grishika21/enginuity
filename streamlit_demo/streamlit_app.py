"""FieldDesk on Streamlit Community Cloud.

Deploy at share.streamlit.io: Create app -> this repository, branch main, main file path
`streamlit_demo/streamlit_app.py`. Advanced settings: Python 3.11, and the secret
GROQ_API_KEY = "...". That's all: the app starts its own Redis (`redis-server` in packages.txt),
loads the sample knowledge base from demo_snapshot/, and runs the orchestrator and the four agents
as separate HTTP servers inside this process (one copy of torch fits the free tier's memory).
This page is a client of the orchestrator's API, exactly like the web UI in orchestrator/static.

Locally (stop scripts/run_all.py first; it uses the same ports):
    GROQ_API_KEY=... streamlit run streamlit_demo/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file locally
        return None
    return str(value) if value else None


# Hosted defaults. The environment or Streamlit secrets win, so the model can change without code.
DEMO_ENV = {
    "LLM_BASE_URL": "https://api.groq.com/openai/v1",
    "LLM_MODEL": "openai/gpt-oss-20b",
    "LLM_REASONING_EFFORT": "low",
    "REDIS_URL": "redis://127.0.0.1:6380/5",
    "TOKENIZERS_PARALLELISM": "false",
}
for _key in ("LLM_BASE_URL", "LLM_MODEL", "LLM_REASONING_EFFORT", "LLM_API_KEY", "JWT_SECRET"):
    if _secret(_key):
        os.environ[_key] = _secret(_key)
for _key, _value in DEMO_ENV.items():
    os.environ.setdefault(_key, _value)
if not os.environ.get("LLM_API_KEY"):
    os.environ["LLM_API_KEY"] = _secret("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")

import httpx  # noqa: E402

from agentkit.config import get_settings  # noqa: E402
from agentkit.launcher import ensure_redis, start_in_process, wait_ready  # noqa: E402
from agentkit.snapshot import restore  # noqa: E402

BASE = "http://127.0.0.1:8000"
USERS = {"asha": "Asha Rao (author)", "ravi": "Ravi Kumar (viewer)", "admin": "Admin"}
SAMPLES = [
    "How do I replace the mechanical seal on the KP-200?",
    "How often should the bearings be lubricated?",
    "Extract the mechanical seal and impeller fitting steps from PI-0042",
    "Write a maintenance procedure for the KP-200 mechanical seal replacement",
]
MAX_TURNS = 40  # per browser session, to keep the shared API key usable for everyone
EXTRA_SKILL = ROOT / "examples" / "skills" / "troubleshooting-guide"
LIVE_SKILL = ROOT / "agents" / "docs_agent" / "skills" / "troubleshooting-guide"

st.set_page_config(page_title="FieldDesk", page_icon=":material/engineering:", layout="wide")


@st.cache_resource(show_spinner=False)
def boot() -> bool:
    ensure_redis(get_settings().redis_url)
    restore()
    start_in_process()
    wait_ready(BASE)
    from agentkit.vectors import embed

    embed(["warm up"])  # load the embedding model now, not on the first question
    return True


if not os.environ.get("LLM_API_KEY"):
    st.error("Add a GROQ_API_KEY secret (app settings, Secrets) to run the demo.")
    st.stop()
with st.spinner("Starting the orchestrator and four agents. The first start takes about a minute."):
    boot()

ss = st.session_state
ss.setdefault("token", None)
ss.setdefault("user", None)
ss.setdefault("conv", None)
ss.setdefault("turns", 0)
ss.setdefault("uploaded", set())
if ss.pop("reset_attach", False):
    ss["attach_pick"] = ""


def api(method: str, path: str, **kwargs):
    headers = {"Authorization": f"Bearer {ss.token}"} if ss.token else {}
    resp = httpx.request(method, BASE + path, headers=headers, timeout=60, **kwargs)
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = resp.text
        raise RuntimeError(detail or resp.reason_phrase)
    return resp.json()


def sign_in(user: str) -> None:
    res = httpx.post(BASE + "/api/login", json={"username": user, "password": "demo"}, timeout=10).json()
    ss.token, ss.user, ss.roles, ss.conv = res["token"], user, res["roles"], None


@st.cache_data(ttl=600, show_spinner=False)
def file_bytes(file_id: str, user: str, token: str) -> bytes:
    link = httpx.get(f"{BASE}/api/files/{file_id}/link", headers={"Authorization": f"Bearer {token}"}, timeout=10)
    link.raise_for_status()
    resp = httpx.get(BASE + link.json()["url"], timeout=30)
    resp.raise_for_status()
    return resp.content


def send(text: str, files: list[str]) -> None:
    if ss.turns >= MAX_TURNS:
        st.warning("This browser session reached the demo's message limit. Reload the page to start again.")
        return
    ss.turns += 1
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"):
        with st.status("Routing...", expanded=True) as status:
            try:
                res = api("POST", "/api/chat", json={"message": text, "conversation_id": ss.conv, "files": files})
                ss.conv = res["conversation_id"]
                with httpx.stream("GET", f"{BASE}/api/turns/{res['job_id']}/stream",
                                  headers={"Authorization": f"Bearer {ss.token}"}, timeout=httpx.Timeout(30, read=900)) as stream:
                    for line in stream.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        event = json.loads(line[5:])
                        if event["event"] == "progress":
                            detail = f": {event['detail']}" if event.get("detail") else ""
                            status.write(f"`{event['agent']}` {event['stage']}{detail}")
                            status.update(label=event["stage"])
                        elif event["event"] == "done":
                            break
                status.update(label="Done", state="complete", expanded=False)
            except Exception as exc:  # shown to the user; the page keeps working
                status.update(label=f"Failed: {exc}", state="error")
                return
    ss.reset_attach = True
    st.rerun()


def render(message: dict, index: int) -> None:
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        if role == "assistant" and message.get("agent"):
            st.caption(f"via `{message['agent']}`")
        if role == "user" and message.get("files"):
            st.caption("Attached: " + ", ".join(f["name"] for f in message["files"]))
        st.markdown(message.get("text", ""))
        steps = message.get("steps") or []
        if steps:
            with st.expander(f"{len(steps)} steps"):
                for s in steps:
                    detail = f": {s['detail']}" if s.get("detail") else ""
                    st.markdown(f"`{s['agent']}` {s['stage']}{detail}")
        for f in message.get("files") or [] if role == "assistant" else []:
            try:
                data = file_bytes(f["file_id"], ss.user, ss.token)
            except Exception:
                st.caption(f"{f['name']} (not available)")
                continue
            st.download_button(f["name"], data=data, file_name=f["name"], key=f"dl-{index}-{f['file_id']}",
                               icon=":material/description:")


# ---- sidebar -------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## FieldDesk")
    st.caption("A multi-agent assistant for equipment documentation. Demo data only.")
    picked = st.radio("Signed in as", list(USERS), format_func=USERS.get, key="user_pick")
    if ss.user != picked:
        sign_in(picked)
    if st.button("New chat", icon=":material/add:", use_container_width=True):
        ss.conv = None
        st.rerun()
    chats = api("GET", "/api/conversations")
    if chats:
        st.markdown("**Chats**")
        for c in chats[:8]:
            label = c["title"][:42] + (f" (from {c['owner']})" if c["shared"] else "")
            if st.button(label, key=f"conv-{c['id']}", use_container_width=True,
                         type="primary" if c["id"] == ss.conv else "secondary"):
                ss.conv = c["id"]
                st.rerun()

    st.markdown("**Agents online**")
    st.caption("From the Redis registry: an agent that stops sending heartbeats drops out.")
    for a in api("GET", "/api/agents"):
        mark = ":green[●]" if a["allowed"] else ":gray[●]"
        lock = "" if a["allowed"] else " · *not for your role*"
        skills = ", ".join(s["name"] + (" (asks approval)" if s["approval"] else "") for s in a["skills"])
        st.markdown(f"{mark} **{a['name']}**{lock}  \n<small>{skills}</small>", unsafe_allow_html=True)

    upload = st.file_uploader("Upload a document", type=["pdf", "docx", "pptx", "xlsx", "csv", "md", "html", "txt"])
    if upload is not None and (upload.name, upload.size) not in ss.uploaded:
        ref = httpx.post(BASE + "/api/upload", headers={"Authorization": f"Bearer {ss.token}"},
                         files={"file": (upload.name, upload.getvalue())},
                         data={"conversation_id": ss.conv or ""}, timeout=60).json()
        ss.uploaded.add((upload.name, upload.size))
        ss["attach_pick"] = ref["file_id"]

    if "admin" in ss.roles:
        st.markdown("**Admin**")
        enabled = st.toggle("Troubleshooting-guide skill", value=LIVE_SKILL.exists(),
                            help="Adds a new document type by copying a SKILL.md folder, then reloads the docs agent. No restart.")
        if enabled != LIVE_SKILL.exists():
            if enabled:
                shutil.copytree(EXTRA_SKILL, LIVE_SKILL)
            else:
                shutil.rmtree(LIVE_SKILL)
            st.toast("docs-agent skills: " + ", ".join(api("POST", "/api/admin/reload-skills")["skills"]))
            st.rerun()

# ---- chat ----------------------------------------------------------------------------------

conversation = api("GET", f"/api/conversations/{ss.conv}") if ss.conv else None
messages = conversation["messages"] if conversation else []

if not messages:
    st.markdown("### What do you need?")
    st.markdown(
        "Ask about the fictional **KP-200 pump** or **KC-50 compressor**, extract from a process "
        "instruction, or have a document drafted for your review. Every request is routed to one of "
        "four agents; open **steps** under an answer to see which ones worked on it.")
    cols = st.columns(2)
    for i, sample in enumerate(SAMPLES):
        if cols[i % 2].button(sample, key=f"sample-{i}", use_container_width=True):
            ss.queued = sample
            if "PI-0042" in sample:
                ss.queued_files = [f["file_id"] for f in api("GET", "/api/files") if f["name"].startswith("PI-0042")]
            st.rerun()
    st.caption("Signed in as Ravi (viewer)? Try asking for a procedure: the role check refuses it in code.")

for i, message in enumerate(messages):
    render(message, i)

last = messages[-1] if messages else None
if last and last["role"] == "assistant" and (last.get("question") or {}).get("options"):
    options = last["question"]["options"]
    cols = st.columns(min(len(options), 4) or 1)
    for i, option in enumerate(options):
        if cols[i % len(cols)].button(option, key=f"opt-{len(messages)}-{i}",
                                      type="primary" if option in ("yes", "approve") else "secondary"):
            ss.queued = option
            st.rerun()

documents = [f for f in api("GET", "/api/files") if f["owner"] == "library" or f["kind"] == "upload"]
names = {f["file_id"]: ("Library: " if f["owner"] == "library" else "Mine: ") + f["name"] for f in documents}
attach = st.selectbox("Attach a document (optional)", ["", *names], key="attach_pick",
                      format_func=lambda fid: names.get(fid, "No document"))

prompt = st.chat_input("Message FieldDesk")
text = prompt or ss.pop("queued", None)
if text:
    files = ss.pop("queued_files", None) or ([attach] if attach else [])
    send(text, files)
