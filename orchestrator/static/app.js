"use strict";

const $ = (id) => document.getElementById(id);
const FILE_ICON = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>';
const state = { token: null, me: null, conv: null, attached: [], busy: false, files: [] };

const SAMPLES = {
  author: [
    "How do I replace the mechanical seal on the KP-200?",
    "How often should the bearings be lubricated?",
    "Extract the mechanical seal and impeller fitting steps from PI-0042",
    "Write a maintenance procedure for the KP-200 mechanical seal replacement",
  ],
  viewer: [
    "What torque does the KP-200 impeller nut need?",
    "How often should the bearings be lubricated?",
    "Write a maintenance procedure for the KP-200 seal",
  ],
  admin: ["What can you do?", "How do I change the oil on the KC-50?"],
};

// ---- api ---------------------------------------------------------------------------------

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (opts.json) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
  const res = await fetch(path, { ...opts, headers });
  if (res.status === 401 && state.token) { logout(); throw new Error("Signed out"); }
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(body?.detail || res.statusText);
  return body;
}

async function* sse(path) {
  const res = await fetch(path, { headers: { Authorization: `Bearer ${state.token}` } });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const data = raw.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trim()).join("\n");
      if (data) yield JSON.parse(data);
    }
  }
}

// ---- markdown (small, escapes first) -----------------------------------------------------

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function inline(s) {
  return esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
}
function md(text) {
  const lines = String(text || "").split("\n");
  let html = "", list = null, table = [];
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  const closeTable = () => {
    if (!table.length) return;
    const rows = table.filter((r) => !r.every((c) => /^:?-{2,}:?$/.test(c)));
    html += "<table>" + rows.map((r, i) => "<tr>" + r.map((c) => (i ? `<td>${inline(c)}</td>` : `<th>${inline(c)}</th>`)).join("") + "</tr>").join("") + "</table>";
    table = [];
  };
  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith("|")) { closeList(); table.push(line.replace(/^\||\|$/g, "").split("|").map((c) => c.trim())); continue; }
    closeTable();
    let m;
    if (!line) continue; // a blank line inside a list keeps the list (and its numbering) going
    if ((m = line.match(/^(#{1,3})\s+(.*)$/))) { closeList(); html += `<h${m[1].length + 1}>${inline(m[2])}</h${m[1].length + 1}>`; }
    else if ((m = line.match(/^\d+[.)]\s+(.*)$/))) { if (list !== "ol") { closeList(); html += "<ol>"; list = "ol"; } html += `<li>${inline(m[1])}</li>`; }
    else if ((m = line.match(/^[-*]\s+(.*)$/))) { if (list !== "ul") { closeList(); html += "<ul>"; list = "ul"; } html += `<li>${inline(m[1])}</li>`; }
    else { closeList(); html += `<p>${inline(line)}</p>`; }
  }
  closeList(); closeTable();
  return html;
}

// ---- login -------------------------------------------------------------------------------

async function showLogin() {
  $("app").classList.add("hidden");
  $("login").classList.remove("hidden");
  const users = await api("/api/users");
  $("user-list").innerHTML = "";
  for (const u of users) {
    const b = document.createElement("button");
    b.className = "btn user-card";
    b.innerHTML = `<span>${esc(u.name)}</span><span class="pill">${esc(u.roles.join(", "))}</span>`;
    b.onclick = () => login(u.user);
    $("user-list").appendChild(b);
  }
}

async function login(user) {
  try {
    const res = await api("/api/login", { method: "POST", json: { username: user, password: $("password").value } });
    state.token = res.token;
    state.me = res;
    try { sessionStorage.setItem("fd-auth", JSON.stringify(res)); } catch (e) { /* storage may be blocked */ }
    start();
  } catch (e) { $("login-error").textContent = e.message; }
}

function logout() {
  state.token = null;
  try { sessionStorage.removeItem("fd-auth"); } catch (e) { /* ignore */ }
  showLogin();
}

// ---- app ---------------------------------------------------------------------------------

function start() {
  $("login").classList.add("hidden");
  $("app").classList.remove("hidden");
  $("me-name").textContent = state.me.name;
  $("me-roles").textContent = "Roles: " + state.me.roles.join(", ");
  $("reload-skills").classList.toggle("hidden", !state.me.roles.includes("admin"));
  newChat();
  loadConversations();
  loadAgents();
  loadFiles();
  clearInterval(state.agentTimer);
  state.agentTimer = setInterval(loadAgents, 5000);
}

function renderSamples() {
  const role = state.me.roles.includes("admin") ? "admin" : state.me.roles.includes("author") ? "author" : "viewer";
  $("samples").innerHTML = "";
  for (const text of SAMPLES[role]) {
    const b = document.createElement("button");
    b.className = "btn sample";
    b.textContent = text;
    b.onclick = () => send(text);
    $("samples").appendChild(b);
  }
}

async function loadConversations() {
  const list = await api("/api/conversations");
  $("conv-list").innerHTML = "";
  for (const c of list) {
    const li = document.createElement("li");
    li.innerHTML = esc(c.title) + (c.shared ? `<span class="tag">shared by ${esc(c.owner)}</span>` : "");
    li.classList.toggle("active", state.conv?.id === c.id);
    li.onclick = () => openConversation(c.id);
    $("conv-list").appendChild(li);
  }
}

async function loadAgents() {
  if (!state.token) return;
  try {
    const agents = await api("/api/agents");
    $("agent-count").textContent = agents.length;
    $("agent-list").innerHTML = "";
    for (const a of agents) {
      const li = document.createElement("li");
      li.className = "agent" + (a.allowed ? "" : " locked");
      const skills = a.skills.map((s) => esc(s.name) + (s.approval ? " (asks approval)" : "")).join("<br>");
      li.innerHTML = `<div class="top"><span><span class="dot"></span>${esc(a.name)}</span>` +
        `<span class="lock">${a.allowed ? "beat " + a.heartbeat_age_s + "s ago" : "not for your role"}</span></div>` +
        `<div class="skills">${skills}</div>`;
      $("agent-list").appendChild(li);
    }
  } catch (e) { /* the next poll retries */ }
}

async function loadFiles() {
  state.files = await api("/api/files");
  const sel = $("library");
  sel.innerHTML = '<option value="">Attach a document...</option>';
  for (const f of state.files.filter((f) => f.owner === "library" || f.kind === "upload")) {
    const o = document.createElement("option");
    o.value = f.file_id;
    o.textContent = (f.owner === "library" ? "Library: " : "Mine: ") + f.name;
    sel.appendChild(o);
  }
}

function newChat() {
  state.conv = null;
  $("chat-title").textContent = "New chat";
  $("messages").innerHTML = "";
  $("messages").appendChild(emptyState());
  renderSamples();
}

function emptyState() {
  const tpl = document.createElement("div");
  tpl.id = "empty";
  tpl.className = "empty";
  tpl.innerHTML = '<h2>What do you need?</h2><p class="muted">Ask about the KP-200 pump or KC-50 compressor, extract from a process instruction, or have a document drafted for review.</p><div id="samples" class="samples"></div>';
  return tpl;
}

async function openConversation(id) {
  const conv = await api(`/api/conversations/${id}`);
  state.conv = conv;
  $("chat-title").textContent = conv.title + (conv.owner !== state.me.user ? ` (shared by ${conv.owner})` : "");
  $("messages").innerHTML = "";
  for (const m of conv.messages) renderMessage(m, false);
  scrollDown();
  loadConversations();
}

function renderMessage(m, live) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + m.role;
  const who = m.role === "user" ? esc(m.user || state.me.user) : "FieldDesk" + (m.agent ? ` <span class="agent-tag">${esc(m.agent)}</span>` : "");
  wrap.innerHTML = `<div class="who">${who}</div><div class="bubble"></div>`;
  const bubble = wrap.querySelector(".bubble");
  if (m.role === "assistant") {
    const steps = document.createElement("details");
    steps.className = "steps";
    steps.open = live;
    steps.innerHTML = `<summary>${live ? "Working" : (m.steps?.length || 0) + " steps"}</summary>`;
    for (const s of m.steps || []) steps.appendChild(stepRow(s));
    bubble.appendChild(steps);
    if (!live && !(m.steps || []).length) steps.remove();
  }
  const body = document.createElement("div");
  body.className = "body";
  body.innerHTML = md(m.text);
  bubble.appendChild(body);
  if (m.files?.length) bubble.appendChild(fileChips(m.files));
  if (m.question && m.question.options?.length) bubble.appendChild(questionBox(m.question));
  $("messages").appendChild(wrap);
  return wrap;
}

function stepRow(s) {
  const d = document.createElement("div");
  d.className = "step";
  d.innerHTML = `<span class="ag">${esc(s.agent)}</span><span>${esc(s.stage)}${s.detail ? ": " + esc(s.detail) : ""}</span>`;
  return d;
}

function fileChips(list) {
  const box = document.createElement("div");
  box.className = "files";
  for (const f of list) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "file-chip";
    chip.innerHTML = FILE_ICON + " " + esc(f.name);
    chip.onclick = async () => {
      const { url } = await api(`/api/files/${f.file_id}/link`);
      window.open(url, "_blank");
    };
    box.appendChild(chip);
  }
  return box;
}

function questionBox(q) {
  const box = document.createElement("div");
  box.className = "question";
  const label = { approval: "Needs your approval", review: "Review the draft", choice: "Pick one", input: "Needs input" }[q.kind] || "Needs input";
  box.innerHTML = `<div class="kind">${label}</div>`;
  const opts = document.createElement("div");
  opts.className = "options";
  for (const o of q.options) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn small" + (o === "yes" || o === "approve" ? " primary" : "");
    b.textContent = o;
    b.onclick = () => send(o);
    opts.appendChild(b);
  }
  box.appendChild(opts);
  return box;
}

function scrollDown() { $("messages").scrollTop = $("messages").scrollHeight; }

async function send(text) {
  text = (text ?? $("input").value).trim();
  if (!text || state.busy) return;
  state.busy = true;
  $("send").disabled = true;
  $("empty")?.remove();
  const files = state.attached.map((f) => f.file_id);
  renderMessage({ role: "user", text, files: state.attached, user: state.me.user }, false);
  $("input").value = "";
  state.attached = [];
  renderAttached();
  const live = renderMessage({ role: "assistant", text: "", steps: [] }, true);
  const steps = live.querySelector(".steps");
  scrollDown();
  try {
    const res = await api("/api/chat", { method: "POST", json: { message: text, conversation_id: state.conv?.id, files } });
    for await (const ev of sse(`/api/turns/${res.job_id}/stream`)) {
      if (ev.event === "progress") { steps.appendChild(stepRow(ev)); scrollDown(); }
      if (ev.event === "message") {
        live.remove();
        const m = renderMessage(ev.message, false);
        m.querySelector(".steps")?.removeAttribute("open");
        scrollDown();
      }
    }
    if (!state.conv || state.conv.id !== res.conversation_id) {
      state.conv = await api(`/api/conversations/${res.conversation_id}`);
      $("chat-title").textContent = state.conv.title;
    }
    loadConversations();
    loadFiles();
  } catch (e) {
    live.querySelector(".body").innerHTML = `<p class="error">${esc(e.message)}</p>`;
  } finally {
    state.busy = false;
    $("send").disabled = false;
  }
}

function renderAttached() {
  $("attached").innerHTML = "";
  for (const f of state.attached) {
    const chip = document.createElement("span");
    chip.className = "file-chip";
    chip.textContent = f.name;
    chip.onclick = () => { state.attached = state.attached.filter((x) => x !== f); renderAttached(); };
    $("attached").appendChild(chip);
  }
}

// ---- wiring ------------------------------------------------------------------------------

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("new-chat").onclick = newChat;
$("logout").onclick = logout;
$("library").onchange = (e) => {
  const f = state.files.find((x) => x.file_id === e.target.value);
  if (f && !state.attached.some((a) => a.file_id === f.file_id)) state.attached.push({ file_id: f.file_id, name: f.name });
  e.target.value = "";
  renderAttached();
};
$("upload").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  if (state.conv) form.append("conversation_id", state.conv.id);
  try {
    const ref = await api("/api/upload", { method: "POST", body: form });
    state.attached.push(ref);
    renderAttached();
    loadFiles();
  } catch (err) { alert(err.message); }
  e.target.value = "";
};
$("share").onclick = async () => {
  if (!state.conv) return alert("Send a message first.");
  const user = prompt("Share this chat with which user? (asha, ravi, admin)");
  if (!user) return;
  try { await api(`/api/conversations/${state.conv.id}/share`, { method: "POST", json: { user } }); alert(`Shared with ${user}.`); }
  catch (err) { alert(err.message); }
};
$("reload-skills").onclick = async () => {
  try { const r = await api("/api/admin/reload-skills", { method: "POST" }); alert("docs-agent skills: " + r.skills.join(", ")); loadAgents(); }
  catch (err) { alert(err.message); }
};

(function boot() {
  let saved = null;
  try { saved = JSON.parse(sessionStorage.getItem("fd-auth") || "null"); } catch (e) { saved = null; }
  if (saved?.token) { state.token = saved.token; state.me = saved; start(); } else { showLogin(); }
})();
