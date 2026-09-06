/* ==========================================================================
   SINA AI - UI controller
   --------------------------------------------------------------------------
   Security note: every string that reaches the DOM from the model, from the
   server, or from localStorage goes through DOMPurify (for Markdown) or
   textContent (for plain text). The previous version passed `marked.parse()`
   output straight to innerHTML, and built the voice bubble with
   `${text.replace(/'/g, "\\'")}` inside an inline onclick - a retrieved
   document containing HTML could execute script in every visitor's page.
   ========================================================================== */

import { SinaAvatar, WaveBackground } from "./avatar.js";

/* ------------------------------------------------------------------ util */
const $ = (id) => document.getElementById(id);
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const SANITIZE_CONFIG = {
  ALLOWED_TAGS: [
    "p", "br", "hr", "strong", "em", "del", "code", "pre", "blockquote",
    "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tr", "th", "td", "a", "span",
  ],
  ALLOWED_ATTR: ["href", "title", "target", "rel", "align"],
  ALLOW_DATA_ATTR: false,
};

/** Render Markdown safely. Never returns unsanitised HTML. */
function renderMarkdown(text) {
  const raw = window.marked
    ? window.marked.parse(text, { breaks: true, gfm: true })
    : escapeHtml(text).replace(/\n/g, "<br>");
  if (!window.DOMPurify) {
    // Sanitiser missing (blocked CDN): degrade to plain text rather than
    // injecting HTML we have not checked.
    return escapeHtml(text).replace(/\n/g, "<br>");
  }
  return window.DOMPurify.sanitize(raw, SANITIZE_CONFIG);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

/**
 * Turn a thrown fetch error into something a student can act on.
 *
 * A dead or restarting backend makes `fetch` reject with the bare browser
 * string "Failed to fetch", which tells the user nothing about what to do.
 */
function friendlyError(error) {
  if (error?.name === "AbortError") return "Stopped.";
  const raw = String(error?.message || error || "");
  if (
    error instanceof TypeError ||
    /failed to fetch|networkerror|network error|load failed/i.test(raw)
  ) {
    return "Can't reach the server. Check that it's running, then try again.";
  }
  return raw || "Something went wrong. Please try again.";
}

/** Make wide Markdown tables scroll instead of breaking the layout. */
function wrapTables(root) {
  root.querySelectorAll("table").forEach((table) => {
    if (table.parentElement?.classList.contains("table-wrap")) return;
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    table.replaceWith(wrap);
    wrap.appendChild(table);
  });
  root.querySelectorAll('a[href]').forEach((a) => {
    a.setAttribute("target", "_blank");
    a.setAttribute("rel", "noopener noreferrer");
  });
}

const icons = {
  user: '<svg viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
  bot: '<svg viewBox="0 0 24 24"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>',
  copy: '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
  flag: '<svg viewBox="0 0 24 24"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/></svg>',
  speak: '<svg viewBox="0 0 24 24"><path d="M11 5 6 9H2v6h4l5 4V5z"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>',
};

/* ----------------------------------------------------------------- toast */
const toastHost = $("toasts");
export function toast(message, kind = "info", duration = 3200) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.setAttribute("role", kind === "error" ? "alert" : "status");
  el.textContent = message;
  toastHost.appendChild(el);
  setTimeout(() => {
    el.classList.add("leaving");
    setTimeout(() => el.remove(), 320);
  }, duration);
}

/* ----------------------------------------------------------------- theme */
const THEME_KEY = "sina.theme";

function systemTheme() {
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

let theme = localStorage.getItem(THEME_KEY) || systemTheme();

function applyTheme(next) {
  theme = next;
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem(THEME_KEY, next); } catch { /* private mode */ }
  const isLight = next === "light";
  $("themeBtn").setAttribute("aria-label", isLight ? "Switch to dark theme" : "Switch to light theme");
  $("themeIconMoon").hidden = isLight;
  $("themeIconSun").hidden = !isLight;
  wave?.setColor(isLight ? 0x6c5ce7 : 0xa29bfe);
  avatar?.setTheme(next);
}

/* ------------------------------------------------------------- 3d scenes */
const avatar = new SinaAvatar($("sinaCanvas"), {
  onProgress: (fraction) => setLoadingProgress(0.15 + fraction * 0.8),
});
const wave = new WaveBackground($("waveLayer"));

// Handle for debugging from the console (window.sina.avatar.setState("speaking"), etc).
window.sina = { avatar, wave };

let activeTab = "sina";
let lastFrame = performance.now();

function tick(now) {
  requestAnimationFrame(tick);
  const dt = Math.min((now - lastFrame) / 1000, 1 / 20);
  lastFrame = now;
  wave.update(dt);
}
requestAnimationFrame(tick);
avatar.start();

window.addEventListener("resize", () => {
  avatar.resize();
  wave.resize();
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) lastFrame = performance.now();
});

/* --------------------------------------------------------- loading screen */
const loadingScreen = $("loadingScreen");
const loadingFill = $("loadingFill");
const loadingStatus = $("loadingStatus");
let loadingDone = false;

function setLoadingProgress(fraction, label) {
  loadingFill.style.width = `${Math.round(clamp01(fraction) * 100)}%`;
  if (label) loadingStatus.textContent = label;
}
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

function finishLoading(message) {
  if (loadingDone) return;
  loadingDone = true;
  setLoadingProgress(1, message || "Ready");
  setTimeout(() => {
    loadingScreen.classList.add("hidden");
    // Fully remove it so it can never trap focus or clicks.
    setTimeout(() => loadingScreen.remove(), 800);
  }, 320);
}

setLoadingProgress(0.12, "Waking Sina up");
avatar
  .load("/vrm-model")
  .then(() => finishLoading("Ready"))
  .catch((error) => {
    console.error("[SINA] VRM failed to load", error);
    // A missing avatar must not block the chat, which is the actual product.
    finishLoading("Avatar unavailable - chat still works");
    toast("Sina's 3D avatar could not load, but chat works normally.", "error", 5000);
  });
// Never let a stalled asset trap the user behind the splash.
setTimeout(() => finishLoading("Ready"), 12000);

/* ------------------------------------------------------------ chat state */
const HISTORY_KEY = "sina.history.v1";
const HISTORY_LIMIT = 40;

/** @type {{role:'user'|'ai', text:string, ts:number}[]} */
let history = [];
try {
  const stored = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  if (Array.isArray(stored)) history = stored.slice(-HISTORY_LIMIT);
} catch { history = []; }

function saveHistory() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-HISTORY_LIMIT)));
  } catch { /* quota or private mode - the transcript is not critical */ }
}

function pushHistory(role, text) {
  history.push({ role, text, ts: Date.now() });
  if (history.length > HISTORY_LIMIT) history = history.slice(-HISTORY_LIMIT);
  saveHistory();
  renderSidebar();
}

/* --------------------------------------------------------------- sidebar */
const sidebar = $("sidebar");
const sidebarOverlay = $("sidebarOverlay");
const sidebarBody = $("sidebarBody");

function openSidebar() {
  sidebar.classList.add("open");
  sidebarOverlay.classList.add("open");
  sidebar.setAttribute("aria-hidden", "false");
  $("sidebarClose").focus();
}
function closeSidebar() {
  sidebar.classList.remove("open");
  sidebarOverlay.classList.remove("open");
  sidebar.setAttribute("aria-hidden", "true");
}

function renderSidebar() {
  sidebarBody.textContent = "";
  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "sidebar-empty";
    empty.innerHTML =
      '<svg viewBox="0 0 24 24"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>' +
      "<p>No conversations yet.<br>Start talking to Sina!</p>";
    sidebarBody.appendChild(empty);
    return;
  }
  for (const entry of history) {
    const el = document.createElement("div");
    el.className = `sidebar-msg ${entry.role}`;
    const label = document.createElement("div");
    label.className = "sidebar-msg-label";
    label.textContent = entry.role === "user" ? "You" : "Sina";
    const body = document.createElement("div");
    body.textContent = entry.text; // plain text, never HTML
    el.append(label, body);
    sidebarBody.appendChild(el);
  }
  sidebarBody.scrollTop = sidebarBody.scrollHeight;
}
renderSidebar();

$("menuBtn").addEventListener("click", openSidebar);
$("sidebarClose").addEventListener("click", closeSidebar);
sidebarOverlay.addEventListener("click", closeSidebar);
$("clearHistoryBtn").addEventListener("click", async () => {
  history = [];
  saveHistory();
  renderSidebar();
  chatInner.textContent = "";
  conversationStarted = false;
  welcome.hidden = false;
  chatInner.appendChild(welcome);
  try {
    await fetch("/api/session/reset", { method: "POST" });
  } catch { /* server-side memory will expire on its own */ }
  toast("Conversation cleared", "success");
});

/* ----------------------------------------------------------------- modal */
function openModal(id) {
  const modal = $(id);
  modal.classList.add("open");
  modal.setAttribute("aria-hidden", "false");
  modal.querySelector("button, textarea, input")?.focus();
}
function closeModal(id) {
  const modal = $(id);
  modal.classList.remove("open");
  modal.setAttribute("aria-hidden", "true");
}
document.querySelectorAll("[data-close-modal]").forEach((btn) => {
  btn.addEventListener("click", () => closeModal(btn.dataset.closeModal));
});
document.querySelectorAll(".modal").forEach((modal) => {
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal(modal.id);
  });
});
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  document.querySelectorAll(".modal.open").forEach((m) => closeModal(m.id));
  if (sidebar.classList.contains("open")) closeSidebar();
});

/* ---------------------------------------------------------------- report */
let reportContext = { query: "", response: "", source: "chat" };
let reportRating = 0;
const stars = Array.from(document.querySelectorAll(".star"));

function setRating(value) {
  reportRating = value;
  stars.forEach((star, index) => {
    star.classList.toggle("on", index < value);
    star.setAttribute("aria-checked", String(index + 1 === value));
  });
}
stars.forEach((star) => {
  star.addEventListener("click", () => setRating(Number(star.dataset.value)));
  star.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setRating(Number(star.dataset.value));
    }
  });
});

function openReport(query, response, source) {
  reportContext = { query, response, source };
  setRating(0);
  $("reportText").value = "";
  openModal("reportModal");
}

$("submitReport").addEventListener("click", async () => {
  if (!reportRating) {
    toast("Please pick a star rating first", "error");
    return;
  }
  const button = $("submitReport");
  button.disabled = true;
  button.textContent = "Sending...";
  try {
    const response = await fetch("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: reportContext.query,
        response: reportContext.response,
        rating: reportRating,
        reason: $("reportText").value,
        source: reportContext.source,
      }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    closeModal("reportModal");
    toast("Thanks for the feedback!", "success");
  } catch (error) {
    console.error("[SINA] report failed", error);
    toast("Could not send feedback. Please try again.", "error");
  } finally {
    button.disabled = false;
    button.textContent = "Submit feedback";
  }
});

/* -------------------------------------------------------------- changelog */
const CHANGELOG_KEY = "sina.changelog.v4";
if (!localStorage.getItem(CHANGELOG_KEY)) {
  setTimeout(() => {
    openModal("changelogModal");
    try { localStorage.setItem(CHANGELOG_KEY, "1"); } catch { /* ignore */ }
  }, 1800);
}

/* --------------------------------------------------------------- streaming */
/**
 * Read an NDJSON stream, calling `onEvent` for each decoded object.
 * Tolerates partial lines and SSE-style `data:` prefixes.
 */
async function readNdjson(response, onEvent, signal) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      if (signal?.aborted) break;
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (let line of lines) {
        line = line.trim();
        if (!line) continue;
        if (line.startsWith("data:")) line = line.slice(5).trim();
        if (!line || line === "[DONE]") continue;
        try {
          onEvent(JSON.parse(line));
        } catch {
          // A truncated frame is not fatal - keep reading.
        }
      }
    }
  } finally {
    try { reader.releaseLock(); } catch { /* already released */ }
  }
}

/* -------------------------------------------------------------- chat tab */
const chatScroll = $("chatScroll");
const chatInner = $("chatInner");
const welcome = $("welcome");
const chatInput = $("chatInput");
const sendBtn = $("sendBtn");
const stopBtn = $("stopBtn");
const charCount = $("charCount");
const kbHint = $("kbHint");
const MAX_CHARS = Number(chatInput.getAttribute("maxlength")) || 800;

let conversationStarted = false;
let inFlight = null;
let pinnedToBottom = true;

chatScroll.addEventListener("scroll", () => {
  const distance = chatScroll.scrollHeight - chatScroll.scrollTop - chatScroll.clientHeight;
  pinnedToBottom = distance < 90;
});

function scrollToBottom(force = false) {
  if (!force && !pinnedToBottom) return;
  requestAnimationFrame(() => {
    chatScroll.scrollTo({
      top: chatScroll.scrollHeight,
      behavior: REDUCED_MOTION ? "auto" : "smooth",
    });
  });
}

function addMessage(role, text) {
  if (!conversationStarted) {
    welcome.hidden = true;
    conversationStarted = true;
  }
  const wrap = document.createElement("div");
  wrap.className = "msg";

  const avatarEl = document.createElement("div");
  avatarEl.className = `avatar ${role === "user" ? "user" : "bot"}`;
  avatarEl.innerHTML = role === "user" ? icons.user : icons.bot;
  avatarEl.setAttribute("aria-hidden", "true");

  const body = document.createElement("div");
  body.className = "msg-body";

  const roleEl = document.createElement("div");
  roleEl.className = "msg-role";
  roleEl.textContent = role === "user" ? "You" : "Sina";

  const textEl = document.createElement("div");
  textEl.className = "msg-text";
  if (role === "user") {
    textEl.textContent = text;
  } else if (text) {
    textEl.innerHTML = renderMarkdown(text);
    wrapTables(textEl);
  }

  body.append(roleEl, textEl);
  wrap.append(avatarEl, body);
  chatInner.appendChild(wrap);
  scrollToBottom(role === "user");
  return { wrap, body, textEl };
}

function addActions(body, textEl, query, answer) {
  const actions = document.createElement("div");
  actions.className = "msg-actions";

  const copy = document.createElement("button");
  copy.className = "msg-action";
  copy.type = "button";
  copy.innerHTML = `${icons.copy}<span>Copy</span>`;
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(answer);
      copy.querySelector("span").textContent = "Copied";
      setTimeout(() => { copy.querySelector("span").textContent = "Copy"; }, 1600);
    } catch {
      toast("Clipboard is blocked in this browser", "error");
    }
  });

  const speak = document.createElement("button");
  speak.className = "msg-action";
  speak.type = "button";
  speak.innerHTML = `${icons.speak}<span>Listen</span>`;
  speak.addEventListener("click", () => speakAnswer(answer, speak));

  const report = document.createElement("button");
  report.className = "msg-action";
  report.type = "button";
  report.innerHTML = `${icons.flag}<span>Report</span>`;
  report.addEventListener("click", () => openReport(query, answer, "chat"));

  actions.append(copy, speak, report);
  body.appendChild(actions);
}

async function speakAnswer(text, button) {
  const label = button.querySelector("span");
  label.textContent = "Loading";
  button.disabled = true;
  try {
    const url = await fetchSpeech(text);
    if (!url) throw new Error("no audio");
    switchTab("sina");
    await avatar.speak(url);
    URL.revokeObjectURL(url);
  } catch {
    toast("Could not play that answer aloud", "error");
  } finally {
    label.textContent = "Listen";
    button.disabled = false;
  }
}

function showThinking() {
  const { wrap, body, textEl } = addMessage("ai", "");
  const thinking = document.createElement("div");
  thinking.className = "thinking";
  thinking.innerHTML = '<span class="thinking-orb"></span><span>Thinking...</span>';
  textEl.appendChild(thinking);
  return { wrap, body, textEl, thinking };
}

function setBusy(busy) {
  sendBtn.hidden = busy;
  stopBtn.hidden = !busy;
  chatInput.setAttribute("aria-busy", String(busy));
}

async function sendChatMessage(text) {
  const message = text.trim();
  if (!message || inFlight) return;

  addMessage("user", message);
  pushHistory("user", message);
  chatInput.value = "";
  autoGrow();
  updateCounter();

  const slot = showThinking();
  const controller = new AbortController();
  inFlight = controller;
  setBusy(true);

  let answer = "";
  let firstToken = true;

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal: controller.signal,
    });

    if (response.status === 429) {
      throw new Error("You're sending messages very quickly. Please wait a moment.");
    }
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || `Server error (${response.status})`);
    }

    // Re-parsing the whole answer on every token is O(n^2) and janks long
    // replies, so repaints are coalesced onto animation frames.
    let repaintQueued = false;
    const repaint = () => {
      repaintQueued = false;
      slot.textEl.innerHTML = renderMarkdown(answer);
      wrapTables(slot.textEl);
      scrollToBottom();
    };

    await readNdjson(
      response,
      (event) => {
        if (event.token) {
          if (firstToken) {
            firstToken = false;
            slot.thinking.remove();
          }
          answer += event.token;
          if (!repaintQueued) {
            repaintQueued = true;
            requestAnimationFrame(repaint);
          }
        }
        if (event.error) throw new Error(event.error);
      },
      controller.signal
    );
    repaint(); // guarantee the final state is rendered

    if (!answer.trim()) {
      slot.thinking?.remove();
      slot.textEl.textContent = "I didn't get a response that time. Please try again.";
    } else {
      pushHistory("ai", answer);
      addActions(slot.body, slot.textEl, message, answer);
    }
  } catch (error) {
    slot.thinking?.remove();
    if (error.name === "AbortError") {
      const note = document.createElement("div");
      note.className = "msg-role";
      note.textContent = "Stopped";
      slot.body.appendChild(note);
      if (answer.trim()) pushHistory("ai", answer);
    } else {
      console.error("[SINA] chat failed", error);
      const note = document.createElement("div");
      note.className = "error-note";
      note.textContent = friendlyError(error);
      slot.body.appendChild(note);
    }
  } finally {
    inFlight = null;
    setBusy(false);
    scrollToBottom();
    chatInput.focus();
  }
}

function autoGrow() {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 168)}px`;
}

function updateCounter() {
  const length = chatInput.value.length;
  charCount.textContent = `${length} / ${MAX_CHARS}`;
  charCount.classList.toggle("visible", length > 0);
  charCount.classList.toggle("warn", length > MAX_CHARS * 0.75);
  charCount.classList.toggle("limit", length >= MAX_CHARS);
  sendBtn.classList.toggle("ready", length > 0);
}

chatInput.addEventListener("input", () => { autoGrow(); updateCounter(); });
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage(chatInput.value);
  }
});
let hintShown = false;
chatInput.addEventListener("focus", () => {
  if (hintShown) return;
  hintShown = true;
  kbHint.classList.add("visible");
  setTimeout(() => kbHint.classList.remove("visible"), 3200);
});
sendBtn.addEventListener("click", () => sendChatMessage(chatInput.value));
stopBtn.addEventListener("click", () => inFlight?.abort());

document.querySelectorAll(".prompt-card").forEach((card) => {
  card.addEventListener("click", () => {
    switchTab("chat");
    sendChatMessage(card.dataset.prompt);
  });
});

/* -------------------------------------------------------------- sina tab */
const sinaInput = $("sinaInput");
const sinaSend = $("sinaSend");
const sinaBubble = $("sinaBubble");
const sinaStatus = $("sinaStatus");
const micBtn = $("micBtn");

let sinaBusy = false;
let pendingSinaMessage = null;

function setStatus(message) {
  sinaStatus.textContent = message || "";
  sinaStatus.classList.toggle("visible", Boolean(message));
}

function showBubble(text, query) {
  sinaBubble.textContent = "";
  const body = document.createElement("div");
  body.textContent = text; // voice replies are plain text by design
  sinaBubble.appendChild(body);

  const actions = document.createElement("div");
  actions.className = "msg-actions";
  actions.style.opacity = "1";
  const report = document.createElement("button");
  report.className = "msg-action";
  report.type = "button";
  report.innerHTML = `${icons.flag}<span>Report</span>`;
  report.addEventListener("click", () => openReport(query, text, "sina"));
  actions.appendChild(report);
  sinaBubble.appendChild(actions);

  sinaBubble.classList.add("visible");
}
function hideBubble() { sinaBubble.classList.remove("visible"); }

async function fetchSpeech(text) {
  const response = await fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) return null;
  const blob = await response.blob();
  return blob.size ? URL.createObjectURL(blob) : null;
}

let hideBubbleTimer = null;

async function sendSinaMessage(text) {
  const message = text.trim();
  if (!message) return;

  if (sinaBusy) {
    // Barge-in: interrupt the current answer rather than silently ignoring the
    // new question. Stopping the audio settles the pending speak() promise, so
    // the in-flight turn unwinds and releases `sinaBusy`.
    avatar.stopSpeaking();
    pendingSinaMessage = message;
    setStatus("One moment...");
    return;
  }

  sinaBusy = true;
  sinaSend.disabled = true;
  avatar.stopSpeaking();
  clearTimeout(hideBubbleTimer);
  hideBubble();
  setStatus("Thinking...");
  avatar.setState("thinking");
  pushHistory("user", message);

  let answer = "";
  try {
    const response = await fetch("/api/sina-chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (response.status === 429) throw new Error("Slow down a moment, please.");
    if (!response.ok) throw new Error(`Server error (${response.status})`);

    await readNdjson(response, (event) => {
      if (event.token) answer += event.token;
      if (event.done && event.full_text) answer = event.full_text;
    });

    if (!answer.trim()) {
      setStatus("");
      avatar.setState("idle");
      toast("No answer came back. Please try again.", "error");
      return;
    }

    pushHistory("ai", answer);
    // Show the text immediately - the user should never wait on TTS to read it.
    showBubble(answer, message);
    setStatus("Speaking...");

    let audioUrl = null;
    try {
      audioUrl = await fetchSpeech(answer);
    } catch (error) {
      console.warn("[SINA] TTS unavailable", error);
    }

    if (audioUrl) {
      await avatar.speak(audioUrl);
      URL.revokeObjectURL(audioUrl);
      setStatus("");
      hideBubbleTimer = setTimeout(hideBubble, 8000);
    } else {
      // No audio: the answer can only be read, so say so and leave it up long
      // enough to actually read - roughly reading speed, floored at 12s.
      avatar.setState("idle");
      setStatus("");
      toast("Voice is unavailable right now - here's the answer in text.", "info", 4000);
      const readingMs = Math.min(45000, Math.max(12000, answer.split(/\s+/).length * 320));
      hideBubbleTimer = setTimeout(hideBubble, readingMs);
    }
  } catch (error) {
    console.error("[SINA] voice chat failed", error);
    avatar.setState("idle");
    setStatus("");
    toast(friendlyError(error), "error", 5000);
  } finally {
    sinaBusy = false;
    sinaSend.disabled = false;
    // Run whatever the user asked while the previous answer was still playing.
    if (pendingSinaMessage) {
      const next = pendingSinaMessage;
      pendingSinaMessage = null;
      sendSinaMessage(next);
    }
  }
}

sinaSend.addEventListener("click", () => {
  const text = sinaInput.value;
  sinaInput.value = "";
  sendSinaMessage(text);
});
sinaInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    const text = sinaInput.value;
    sinaInput.value = "";
    sendSinaMessage(text);
  }
});

/* ------------------------------------------------------------ microphone */
let recorder = null;
let chunks = [];
let recording = false;

function pickMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  return candidates.find((type) => window.MediaRecorder?.isTypeSupported?.(type)) || "";
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast("This browser cannot record audio", "error");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    const mimeType = pickMimeType();
    recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    chunks = [];
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
      // Always release the microphone, even if transcription then fails.
      stream.getTracks().forEach((track) => track.stop());
      handleRecordingStop(mimeType || "audio/webm");
    };
    recorder.start();
    recording = true;
    micBtn.classList.add("recording");
    micBtn.setAttribute("aria-pressed", "true");
    $("micIcon").hidden = true;
    $("micWave").classList.add("active");
    setStatus("Listening...");
    avatar.setState("listening");
  } catch (error) {
    console.error("[SINA] microphone denied", error);
    toast("Microphone access was blocked", "error");
    setStatus("");
  }
}

function stopRecording() {
  if (recorder?.state === "recording") recorder.stop();
  recording = false;
  micBtn.classList.remove("recording");
  micBtn.classList.add("processing");
  micBtn.setAttribute("aria-pressed", "false");
  $("micIcon").hidden = false;
  $("micWave").classList.remove("active");
  setStatus("Transcribing...");
}

async function handleRecordingStop(mimeType) {
  const blob = new Blob(chunks, { type: mimeType });
  chunks = [];
  if (!blob.size) {
    micBtn.classList.remove("processing");
    setStatus("");
    return;
  }

  const form = new FormData();
  form.append("audio", blob, "recording.webm");
  try {
    const response = await fetch("/api/transcribe", { method: "POST", body: form });
    const data = await response.json().catch(() => ({}));
    micBtn.classList.remove("processing");

    if (!response.ok || data.error) {
      setStatus("");
      toast(data.error || "Could not transcribe that", "error");
      avatar.setState("idle");
      return;
    }
    const text = (data.text || "").trim();
    if (!text) {
      setStatus("No speech detected");
      avatar.setState("idle");
      setTimeout(() => setStatus(""), 2400);
      return;
    }
    sinaInput.value = "";
    await sendSinaMessage(text);
  } catch (error) {
    console.error("[SINA] transcription failed", error);
    micBtn.classList.remove("processing");
    setStatus("");
    avatar.setState("idle");
    toast("Transcription failed", "error");
  }
}

micBtn.addEventListener("click", () => (recording ? stopRecording() : startRecording()));

/* ------------------------------------------------------------------ tabs */
function switchTab(tab) {
  if (activeTab === tab) return;
  activeTab = tab;
  const isSina = tab === "sina";

  $("sinaPanel").classList.toggle("active", isSina);
  $("chatPanel").classList.toggle("active", !isSina);
  $("tabSina").setAttribute("aria-selected", String(isSina));
  $("tabChat").setAttribute("aria-selected", String(!isSina));
  $("menuBtn").hidden = false;

  // Only the visible scene renders - this is where most of the old GPU cost went.
  avatar.setVisible(isSina);
  wave.setVisible(!isSina);
  if (isSina) {
    avatar.resize();
  } else {
    avatar.stopSpeaking();
    wave.resize();
    chatInput.focus();
  }
}
$("tabSina").addEventListener("click", () => switchTab("sina"));
$("tabChat").addEventListener("click", () => switchTab("chat"));
$("themeBtn").addEventListener("click", () => applyTheme(theme === "dark" ? "light" : "dark"));

/* ------------------------------------------------------------------ init */
applyTheme(theme);
window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", (e) => {
  // Only follow the OS while the user has not made an explicit choice.
  if (!localStorage.getItem(THEME_KEY)) applyTheme(e.matches ? "light" : "dark");
});

avatar.setVisible(true);
wave.setVisible(false);
updateCounter();
