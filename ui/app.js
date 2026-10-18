"use strict";

// ------------------------------------------------------------------ utilities
const $ = (sel) => document.querySelector(sel);
const t = (key) => window.I18N.t(key);

function apiKey() { return localStorage.getItem("apiKey") || ""; }

function apiFetch(url, options = {}) {
  const headers = Object.assign({}, options.headers);
  if (apiKey()) headers["X-API-Key"] = apiKey();
  return fetch(url, Object.assign({}, options, { headers }));
}

async function getJSON(url) {
  const r = await apiFetch(url);
  const text = await r.text();
  if (!r.ok) throw new Error(text || `${r.status}`);
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new Error(text ? text.slice(0, 160) : "invalid JSON response");
  }
}

function wsURL(path) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const key = apiKey();
  return `${proto}://${location.host}${path}${key ? `?api_key=${encodeURIComponent(key)}` : ""}`;
}

const fmtMB = (bytes) => ((bytes || 0) / (1024 * 1024)).toFixed(0);

// FastAPI errors are JSON `{"detail": "..."}`; fall back to the raw body.
function errorFromBody(text) {
  try {
    return JSON.parse(text).detail || text;
  } catch (e) {
    return text;
  }
}

// ----------------------------------------------------------------- player PCM
class PCMPlayer {
  constructor(sampleRate, prebufferMs = 200) {
    this.sr = sampleRate || 24000;
    this.prebuffer = Math.max(0, prebufferMs || 0) / 1000;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.time = 0;
    this.leftover = new Uint8Array(0);
    this.started = false;
    this.pending = [];
    this.pendingSeconds = 0;
    this.flushTimer = null;
    // Master gain: barge-in ducks the assistant so the user hears themselves and
    // the mic picks up less of our own audio (which helps the detector).
    this.gain = this.ctx.createGain();
    this.gain.gain.value = 1;
    this.gain.connect(this.ctx.destination);
    // Sources already scheduled on the output, so `reset()` can cancel them.
    this.sources = [];
  }
  pushBytes(bytes) {
    let raw;
    if (this.leftover.length) {
      raw = new Uint8Array(this.leftover.length + bytes.length);
      raw.set(this.leftover); raw.set(bytes, this.leftover.length);
    } else raw = bytes;
    const usable = raw.length - (raw.length % 2);
    this.leftover = usable < raw.length ? raw.slice(usable) : new Uint8Array(0);
    if (usable > 0) {
      const ab = new ArrayBuffer(usable);
      new Uint8Array(ab).set(raw.subarray(0, usable));
      this.enqueue(new Int16Array(ab));
    }
  }
  // Buffer the first chunks until `prebufferMs` of audio is available (or a
  // short timeout) so playback starts without an immediate underrun/gap.
  enqueue(int16) {
    if (this.started) { this.play(int16); return; }
    this.pending.push(int16);
    this.pendingSeconds += int16.length / this.sr;
    if (this.pendingSeconds >= this.prebuffer) { this.flush(); return; }
    if (!this.flushTimer) this.flushTimer = setTimeout(() => this.flush(), 250);
  }
  flush() {
    if (this.flushTimer) { clearTimeout(this.flushTimer); this.flushTimer = null; }
    if (this.started) return;
    this.started = true;
    const items = this.pending;
    this.pending = []; this.pendingSeconds = 0;
    items.forEach((it) => this.play(it));
  }
  play(int16) {
    // Some browsers gate audio until a user gesture: resume if the context is
    // suspended (e.g. a reply created from a WebSocket callback).
    if (this.ctx.state === "suspended") this.ctx.resume().catch(() => {});
    const f = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) f[i] = int16[i] / 32768;
    const buf = this.ctx.createBuffer(1, f.length, this.sr);
    buf.copyToChannel(f, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf; src.connect(this.gain);
    src.onended = () => {
      const i = this.sources.indexOf(src);
      if (i >= 0) this.sources.splice(i, 1);
    };
    this.sources.push(src);
    const clock = Math.max(this.ctx.currentTime, this.time);
    src.start(clock);
    this.time = clock + buf.duration;
  }
  // Audio still scheduled on the output: the server may have finished the turn
  // while the client keeps speaking for a moment.
  remainingSeconds() { return Math.max(0, this.time - this.ctx.currentTime); }
  setVolume(value) {
    const target = Math.max(0, Math.min(1, value));
    const now = this.ctx.currentTime;
    try {
      this.gain.gain.cancelScheduledValues(now);
      this.gain.gain.setTargetAtTime(target, now, 0.05);
    } catch (e) {
      this.gain.gain.value = target;
    }
  }
  reset() {
    if (this.flushTimer) { clearTimeout(this.flushTimer); this.flushTimer = null; }
    // Cancel anything already scheduled, otherwise a new generate would overlap.
    for (const src of this.sources) { try { src.stop(); } catch (e) { /* ended */ } }
    this.sources = [];
    this.leftover = new Uint8Array(0);
    this.started = false; this.pending = []; this.pendingSeconds = 0;
    this.time = this.ctx.currentTime;
  }
  stop() { try { this.ctx.close(); } catch (e) { /* already closed */ } }
}

function downsampleTo16k(input, inRate) {
  const ratio = inRate / 16000;
  const outLen = Math.round(input.length / ratio);
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const v = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)] || 0));
    out[i] = v * 32767;
  }
  return out;
}

// --------------------------------------------------------------- global state
let CONFIG = null;
const TTS_SESSION = { format: "pcm", sample_rate: 24000 };
let sttModels = [];
let sttActive = "";
let sttService = { enabled: true, ready: false, error: "" };
const sttPickers = [];
let sttDownload = null;
let sttDownloadAbort = null;
let avatars = [];
let voiceEntries = [];
let selectedVoiceId = localStorage.getItem("chatVoice") || "";
let micDeviceId = "";

// ---------------------------------------------------------------- tabs + i18n
window.I18N.init();
window.I18N.apply();

function switchTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((el) => el.classList.remove("active"));
  const section = $(`#tab-${name}`);
  if (section) section.classList.add("active");
  if (name === "chat") fitChatLayout();
}

// The conversation fills whatever height is left below the header/tabs: measure it
// instead of hard-coding the chrome height (fonts, zoom and language vary).
function fitChatLayout() {
  const layout = document.querySelector("#tab-chat .chat-layout");
  if (!layout) return;
  if (window.matchMedia("(max-width: 900px)").matches) { layout.style.height = ""; return; }
  const top = layout.getBoundingClientRect().top;
  layout.style.height = `${Math.max(320, window.innerHeight - top - 6)}px`;
}
window.addEventListener("resize", () => { fitChatLayout(); scrollChatToBottom(); });

document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

$("#lang-select").addEventListener("change", (e) => {
  window.I18N.setLang(e.target.value);
  if (CONFIG) {
    buildTtsFamilySelect();
    renderSttPickers();
    renderAvatars();
    renderVoiceList();
    loadMicDevices();
    applyCloneEngineFields(CONFIG);
    initCloneEngineSelects();
  }
});

// ------------------------------------------------------------------ init app
async function init() {
  try {
    CONFIG = await getJSON("/api/config");
  } catch (e) {
    // A stale/expired stored key: try to (re)establish an admin session.
    try {
      await ensureSession();
      CONFIG = await getJSON("/api/config");
    } catch (e2) {
      $("#health").className = "dot err";
      $("#active-model").textContent = t("status.configUnavailable");
      return;
    }
  }
  $("#title").textContent = CONFIG.ui.title || t("app.title");
  $("#health").className = "dot ok";

  buildTtsFamilySelect();
  $("#tts-family").addEventListener("change", onFamilyChange);

  const lang = $("#tts-language");
  lang.innerHTML = "";
  (CONFIG.languages || ["Auto"]).forEach((l) => {
    const o = document.createElement("option"); o.value = l; o.textContent = l; lang.appendChild(o);
  });

  ["#chat-language", "#ad-lang"].forEach((sel) => {
    const el = $(sel);
    if (!el) return;
    el.innerHTML = "";
    (CONFIG.languages || ["Auto"]).forEach((l) => {
      const o = document.createElement("option"); o.value = l; o.textContent = l; el.appendChild(o);
    });
  });
  if ($("#chat-language")) $("#chat-language").value = "Auto";

  applySettingsFields(CONFIG);
  initCloneEngineSelects();

  applyAccessState();
  if (CONFIG.auth_required && !apiKey()) {
    try { await ensureSession(); } catch (e) { /* not admin-authenticated yet */ }
  }
  loadApiKeys();

  mountSttPickers();
  await loadVoices();
  await loadSttModels();
  await loadAvatars();
  await loadConversationVoices();
  const preload = selectedVoice();
  if (preload) loadVoiceModel(preload);
  await loadMicDevices();
  fitChatLayout();
  refreshConfigModels();
  refreshStatus();
}

function applySettingsFields(cfg) {
  $("#set-llm-url").value = cfg.llm.base_url || "";
  $("#set-llm-model").value = cfg.llm.model || "";
  $("#set-llm-system").value = cfg.llm.system_prompt || "";
  if ($("#set-log-level")) $("#set-log-level").value = (cfg.server || {}).log_level || "info";
  applyVllmFields(cfg);
  applyGenerationFields(cfg);
  applyCloneEngineFields(cfg);
}

function applyVllmFields(cfg) {
  const v = (cfg && cfg.vllm) || {};
  setVllmEngine(!!v.enabled);
  if ($("#set-vllm-family")) $("#set-vllm-family").value = v.family === "base" ? "base" : "custom_voice";
  fillVllmModels();
  refreshVllmBudget();
  refreshVllmService();
  refreshEngineWarning();
  if ($("#set-vllm-url")) $("#set-vllm-url").value = v.base_url || "";
  if ($("#set-vllm-path")) $("#set-vllm-path").value = v.path || "/v1/audio/speech";
  if ($("#set-vllm-sleep")) $("#set-vllm-sleep").value = v.sleep_after || 0;
  if ($("#set-vllm-stage0")) $("#set-vllm-stage0").value = (v.stage_memory || {})["0"] || "";
  if ($("#set-vllm-stage1")) $("#set-vllm-stage1").value = (v.stage_memory || {})["1"] || "";
  if ($("#set-vllm-maxlen")) $("#set-vllm-maxlen").value = v.max_model_len || 2048;
  if ($("#set-vllm-eager")) $("#set-vllm-eager").checked = v.enforce_eager !== false;
  if ($("#set-vllm-sleep-level")) $("#set-vllm-sleep-level").value = String(v.sleep_level || 1);
}

function applyGenerationFields(cfg) {
  const g = cfg.generation || {};
  const set = (id, v) => { if ($(id) && v !== undefined && v !== null) $(id).value = v; };
  set("#gen-temp", g.temperature);
  set("#gen-topp", g.top_p);
  set("#gen-reppen", g.repetition_penalty);
  set("#gen-exag", g.exaggeration);
  set("#gen-cfg", g.cfg_weight);
  set("#gen-minp", g.min_p);
  set("#gen-batch", g.batch_size);
  set("#gen-maxchars", (cfg.chunking || {}).max_chars);
  const text = cfg.text || {};
  set("#gen-cleanup", text.cleanup);
  set("#gen-emoji", text.emoji);
  set("#gen-codeblocks", text.code_blocks);
  set("#gen-format", (cfg.audio || {}).format);
  // The Synthesis tab's per-request fields display the current defaults (what is
  // shown is what the request will carry); "Reset" re-applies them after an edit.
  fillTtsGenerationFields(g);
}

function applyCloneEngineFields(cfg) {
  const m = (cfg && cfg.models) || {};
  fillCloneEngineSelect($("#set-clone-backend"));
  if ($("#set-clone-backend")) $("#set-clone-backend").value = cloneEngineValue(m);
  if ($("#set-cb-watermark")) $("#set-cb-watermark").checked = !!(m.chatterbox && m.chatterbox.watermark);
  toggleCloneEngineFields();
}

// The clone engine is (1) an engine and, for Qwen3, (2) the size of the Base model:
// `models.clone_backend` picks the engine, `models.registry.base` the size.
// Chatterbox first, then every "base" variant offered by the service.
function cloneEngineChoices() {
  const base = tt("chat.backendQwen", "Qwen3-TTS (Base)");
  const choices = [{ value: "chatterbox", label: tt("chat.backendChatterbox", "Chatterbox") }];
  const variants = ((CONFIG && CONFIG.models && CONFIG.models.variants) || [])
    .filter((v) => v.family === "base")
    .slice()
    .sort((a, b) => String(a.label || a.id).localeCompare(String(b.label || b.id)));
  if (variants.length) {
    variants.forEach((v) => choices.push({
      value: `qwen3:${v.label || v.id}`, label: `${base} ${v.label || v.id}`, repo: v.repo,
    }));
  } else {
    choices.push({ value: "qwen3:0.6B", label: `${base} 0.6B`, repo: "Qwen/Qwen3-TTS-12Hz-0.6B-Base" });
    choices.push({ value: "qwen3:1.7B", label: `${base} 1.7B`, repo: "Qwen/Qwen3-TTS-12Hz-1.7B-Base" });
  }
  return choices;
}

function fillCloneEngineSelect(sel) {
  if (!sel) return;
  const current = sel.value;
  sel.innerHTML = "";
  cloneEngineChoices().forEach((c) => {
    const o = document.createElement("option");
    o.value = c.value;
    o.textContent = c.label;
    sel.appendChild(o);
  });
  if (current) sel.value = current;
}

function cloneEngineValue(models) {
  const m = models || {};
  if ((m.clone_backend || "qwen3") === "chatterbox") return "chatterbox";
  const choices = cloneEngineChoices();
  const baseChoices = choices.slice(1);
  const repo = (m.registry || {}).base || "";
  if (repo) {
    const match = choices.find((c) => c.repo === repo);
    if (match) return match.value;
  }
  // No registry entry for this repo (older config): fall back to the recorded size.
  const size = String(m.default_size || "").trim().toLowerCase();
  const bySize = size
    ? baseChoices.find((c) => c.value.toLowerCase().endsWith(`:${size}`))
    : null;
  return (bySize || baseChoices[0] || { value: "qwen3" }).value;
}

// Partial settings patch for a clone-engine choice (the API merges nested dicts).
function cloneEnginePayload(value) {
  const choice = cloneEngineChoices().find((c) => c.value === value);
  if (!choice || !choice.repo) return { models: { clone_backend: "chatterbox" } };
  const label = String(value).split(":")[1] || "";
  return { models: { clone_backend: "qwen3", registry: { base: choice.repo }, default_size: label } };
}

function toggleCloneEngineFields() {
  const cb = $("#set-clone-backend") && $("#set-clone-backend").value === "chatterbox";
  document.querySelectorAll(".cb-only").forEach((el) => { el.style.display = cb ? "" : "none"; });
  // The Chatterbox-only knobs of the *defaults* card follow the default engine.
  show($("#gen-cb-row"), cb);
}


// -------------------------------------------------------------- TTS model UI
function tt(key, fallback) {
  const v = window.I18N.t(key);
  return v === key ? fallback : v;
}
function ttsName(family) { return tt(`tts.model.${family}.name`, family); }
function ttsDesc(family) { return tt(`tts.model.${family}.desc`, ""); }

function selectedFamily() {
  const sel = $("#tts-family");
  if (sel && sel.value) return sel.value;
  const fams = CONFIG.models.families || [];
  return fams[0] ? fams[0].family : "custom_voice";
}

function buildTtsFamilySelect() {
  const sel = $("#tts-family");
  const families = CONFIG.models.families || [];
  const current = sel.value;
  sel.innerHTML = "";
  families.forEach((f) => {
    const o = document.createElement("option");
    o.value = f.family;
    const tag = f.loaded ? ` ✓ ${t("models.loaded")}` : "";
    o.textContent = `${ttsName(f.family)}${tag}`;
    sel.appendChild(o);
  });
  if (current && families.some((f) => f.family === current)) sel.value = current;
  else if (families.length) sel.value = families[0].family;
  buildVariantSelect();
  updateTtsModelDesc();
  updateTtsVoiceControls();
}

function buildVariantSelect() {
  const sel = $("#tts-variant");
  if (!sel) return;
  const family = selectedFamily();
  const variants = (CONFIG.models.variants || []).filter((v) => v.family === family);
  sel.innerHTML = "";
  variants.forEach((v) => {
    const o = document.createElement("option");
    o.value = v.id;
    o.textContent = v.label || v.id;
    sel.appendChild(o);
  });
  sel.disabled = variants.length === 0;
  // Reflect the repo currently registered for this family, if any variant matches.
  const fam = (CONFIG.models.families || []).find((f) => f.family === family);
  if (fam && fam.repo_id) {
    const match = variants.find((v) => v.repo === fam.repo_id);
    if (match) sel.value = match.id;
  }
}

async function selectVariant(id) {
  if (!id) return;
  const status = $("#tts-model-status");
  status.textContent = t("models.loading");
  try {
    const r = await apiFetch("/api/models/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant: id }),
    });
    const text = await r.text();
    if (!r.ok) {
      let detail = text;
      detail = errorFromBody(text);
      throw new Error(detail);
    }
    status.textContent = t("status.saved");
  } catch (e) {
    status.textContent = `${t("status.error")} ${e}`;
  } finally {
    await refreshConfigModels();
    await loadVoices();
  }
}

function updateTtsModelDesc() {
  $("#tts-model-desc").textContent = ttsDesc(selectedFamily());
}

async function onFamilyChange() {
  buildVariantSelect();
  updateTtsModelDesc();
  updateTtsVoiceControls();
  buildTtsAvatarSelect();
  await loadVoices();
  await loadConversationVoices();
}

async function refreshConfigModels() {
  try {
    const data = await getJSON("/api/models");
    CONFIG.models.families = data.families;
    if (data.variants) CONFIG.models.variants = data.variants;
    if (data.default_size) CONFIG.models.default_size = data.default_size;
    buildTtsFamilySelect();
    const loaded = (data.loaded || []).map(ttsName).join(", ");
    $("#tts-model-status").textContent = loaded ? `${t("models.loaded")}: ${loaded}` : t("models.notLoaded");
  } catch (e) { /* ignore */ }
}

$("#tts-load").addEventListener("click", async () => {
  const family = selectedFamily();
  const btn = $("#tts-load");
  btn.disabled = true;
  $("#tts-model-status").textContent = t("models.loading");
  try {
    const r = await apiFetch("/api/models/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ family }),
    });
    const text = await r.text();
    if (!r.ok) {
      let detail = text;
      detail = errorFromBody(text);
      $("#tts-model-status").textContent = `${t("status.error")} ${detail}`;
    } else {
      $("#tts-model-status").textContent = `${ttsName(family)} — ${t("models.loaded")}`;
    }
  } catch (e) {
    $("#tts-model-status").textContent = `${t("status.error")} ${e}`;
  } finally {
    btn.disabled = false;
    await refreshConfigModels();
    await loadVoices();
    await loadConversationVoices();
  }
});

$("#tts-refresh-models").addEventListener("click", refreshConfigModels);

async function loadVoices() {
  const family = selectedFamily();
  try {
    const data = await getJSON(`/api/voices?family=${encodeURIComponent(family)}`);
    const speakers = data.speakers || [];
    const sel = $("#tts-speaker");
    sel.innerHTML = "";
    speakers.forEach((s) => {
      const o = document.createElement("option"); o.value = s; o.textContent = s; sel.appendChild(o);
    });
    if (!speakers.length) {
      const o = document.createElement("option"); o.value = ""; o.textContent = t("tts.defaultSpeaker"); sel.appendChild(o);
    }
  } catch (e) { /* ignore */ }
}

async function refreshEngineWarning() {
  const box = $("#engine-warning");
  if (!box) return;
  // Only relevant when the vLLM-Omni engine is selected: a request routed there
  // now fails loudly instead of silently falling back to the local models.
  if (!(CONFIG && CONFIG.vllm && CONFIG.vllm.enabled)) { box.hidden = true; return; }
  try {
    const r = await apiFetch("/api/vllm/health");
    if (r.status === 403) { box.hidden = true; return; }  // not an admin: no probe
    const body = await r.json().catch(() => ({}));
    if (body.ok) { box.hidden = true; return; }
    box.hidden = false;
    box.textContent = `${t("engine.vllmDown")} ${body.error || `HTTP ${body.status || r.status}`}`;
  } catch (e) {
    box.hidden = true;
  }
}

async function refreshStatus() {
  if (!CONFIG) return;  // config never loaded: leave the error state alone
  refreshEngineWarning();
  try {
    const h = await getJSON("/health");
    $("#health").className = "dot ok";
    $("#active-model").textContent = h.active_family
      ? `${t("status.active")}: ${ttsName(h.active_family)}`
      : t("status.noModel");
  } catch (e) {
    $("#health").className = "dot err";
    $("#active-model").textContent = t("status.offline");
  }
}
setInterval(refreshStatus, 15000);

// ------------------------------------------------------------------ TTS tab
function familyInfo(family) {
  return (CONFIG.models.families || []).find((f) => f.family === family) || null;
}

function familyBackend(family) {
  const info = familyInfo(family || selectedFamily());
  return info && info.backend ? info.backend : "qwen3";
}

function isCloneFamily(family) {
  const f = family || selectedFamily();
  return f === "base" || familyBackend(f) === "chatterbox";
}

function show(el, visible) {
  if (el) el.style.display = visible ? "" : "none";
}

function buildTtsAvatarSelect() {
  const sel = $("#tts-avatar");
  if (!sel) return;
  const current = sel.value;
  sel.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = t("tts.avatarNone");
  sel.appendChild(none);
  avatars.forEach((a) => {
    const o = document.createElement("option");
    o.value = a.id;
    o.textContent = a.name;
    sel.appendChild(o);
  });
  if (current && avatars.some((a) => a.id === current)) sel.value = current;
}

function updateTtsVoiceControls() {
  const family = selectedFamily();
  const backend = familyBackend(family);
  const clone = isCloneFamily(family);
  const qwenClone = clone && backend === "qwen3";
  const englishOnly = family === "chatterbox_turbo" || family === "chatterbox_nano";

  show($("#tts-speaker-wrap"), family === "custom_voice");
  show($("#tts-instruct-wrap"), family === "custom_voice" || family === "voice_design");
  show($("#tts-avatar-wrap"), clone);
  show($("#clone-box"), clone);
  show($("#tts-reftext-wrap"), qwenClone);
  show($("#tts-language-wrap"), !englishOnly);
  // Chatterbox Multilingual exposes exaggeration/CFG/min-p (Turbo/Nano ignore them).
  show($("#tts-gen-cb-row"), family === "chatterbox");

  const hint = $("#tts-clone-hint");
  if (hint) hint.textContent = clone ? t(qwenClone ? "tts.cloneHintQwen" : "tts.cloneHintChatterbox") : "";
}

function ttsPayload(text) {
  const family = selectedFamily();
  const backend = familyBackend(family);
  const clone = isCloneFamily(family);
  const payload = {
    text,
    family,
    language: $("#tts-language").value,
    chunking: $("#tts-chunk").checked,
    format: "pcm",
    stream: $("#tts-stream").checked,
    generation: ttsGenerationParams(),
  };
  if (family === "custom_voice") payload.speaker = $("#tts-speaker").value;
  if (family === "custom_voice" || family === "voice_design") {
    payload.instruct = $("#tts-instruct").value || undefined;
  }
  if (clone) {
    const avatar = $("#tts-avatar").value;
    if (avatar) {
      // A pre-built avatar (already a clone) takes precedence.
      payload.avatar = avatar;
    } else if (backend === "qwen3") {
      payload.ref_text = $("#tts-ref-text").value || undefined;
      payload.x_vector_only_mode = $("#tts-xvec").checked;
    }
  }
  return payload;
}

async function uploadReference() {
  const file = $("#tts-ref-audio").files[0];
  if (!file) return null;
  // The transcript/useful for ICL cloning only (Qwen); skip STT for other backends.
  if (familyBackend() === "qwen3") {
    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await apiFetch("/v1/audio/transcriptions", { method: "POST", body: fd });
      const text = await r.text();
      if (r.ok) { const j = JSON.parse(text); if (j.text) $("#tts-ref-text").value = j.text; }
    } catch (e) { /* STT may be disabled */ }
  }
  return await fileToDataURL(file);
}

function genNum(id, parse, dflt) {
  const el = $(id);
  if (!el) return dflt;
  const v = parse(el.value);
  return Number.isFinite(v) ? v : dflt;
}

function ttsGenerationParams() {
  const g = {
    temperature: genNum("#tts-gen-temp", parseFloat, undefined),
    top_p: genNum("#tts-gen-topp", parseFloat, undefined),
    repetition_penalty: genNum("#tts-gen-reppen", parseFloat, undefined),
    exaggeration: genNum("#tts-gen-exag", parseFloat, undefined),
    cfg_weight: genNum("#tts-gen-cfg", parseFloat, undefined),
    min_p: genNum("#tts-gen-minp", parseFloat, undefined),
  };
  return Object.fromEntries(Object.entries(g).filter(([, v]) => v !== undefined));
}

// The Synthesis tab's generation fields start from the server defaults (Settings
// → Default TTS engine) and are sent with the request, so a one-off tweak does
// not change the defaults for every other client.
const TTS_GEN_FIELDS = {
  "tts-gen-temp": "temperature",
  "tts-gen-topp": "top_p",
  "tts-gen-reppen": "repetition_penalty",
  "tts-gen-exag": "exaggeration",
  "tts-gen-cfg": "cfg_weight",
  "tts-gen-minp": "min_p",
};

function fillTtsGenerationFields(gen) {
  const g = gen || {};
  Object.entries(TTS_GEN_FIELDS).forEach(([id, key]) => {
    const el = $(`#${id}`);
    if (el) el.value = g[key] === undefined || g[key] === null ? "" : g[key];
  });
}

$("#tts-gen-reset").addEventListener("click", () => {
  fillTtsGenerationFields((CONFIG && CONFIG.generation) || {});
});

function fileToDataURL(file) {
  return new Promise((res) => {
    const fr = new FileReader();
    fr.onload = () => res(fr.result);
    fr.readAsDataURL(file);
  });
}

let ttsPlayer = null;

$("#tts-go").addEventListener("click", async () => {
  const text = $("#tts-text").value.trim();
  if (!text) return;
  $("#tts-status").textContent = t("status.generating");
  $("#tts-log").textContent = "";
  const payload = ttsPayload(text);
  // Clone families use the reference audio unless an existing avatar was chosen.
  if (isCloneFamily() && !payload.avatar) {
    const ref = await uploadReference();
    if (ref) payload.ref_audio = ref;
  }
  if (payload.stream) {
    try {
      const r = await apiFetch("/api/tts", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error((await r.text()).slice(0, 200));
      // Honour the rate the server announces instead of the configured one.
      const rate = parseInt(r.headers.get("x-sample-rate") || "0", 10);
      if (rate && rate !== TTS_SESSION.sample_rate && ttsPlayer) {
        ttsPlayer.stop();
        ttsPlayer = null;
      }
      if (rate) TTS_SESSION.sample_rate = rate;
      ttsPlayer = ttsPlayer || new PCMPlayer(TTS_SESSION.sample_rate);
      ttsPlayer.reset();
      const reader = r.body.getReader();
      const t0 = performance.now(); let bytes = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        ttsPlayer.pushBytes(value);
        bytes += value.length;
        $("#tts-log").textContent = `${bytes} bytes in ${((performance.now() - t0) / 1000).toFixed(2)} s`;
      }
      $("#tts-status").textContent = t("status.done");
    } catch (e) {
      $("#tts-status").textContent = t("status.error");
      $("#tts-log").textContent = String(e);
    }
  } else {
    try {
      const r = await apiFetch("/api/tts", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(Object.assign({}, payload, { stream: false, format: "wav" })),
      });
      if (!r.ok) throw new Error((await r.text()).slice(0, 200));
      const player = $("#tts-player");
      if (player.dataset.url) URL.revokeObjectURL(player.dataset.url);
      const url = URL.createObjectURL(await r.blob());
      player.dataset.url = url;
      player.src = url;
      $("#tts-status").textContent = t("status.done");
    } catch (e) {
      $("#tts-status").textContent = t("status.error");
      $("#tts-log").textContent = String(e);
    }
  }
  refreshStatus();
});

// ------------------------------------------------------------------ avatars
function humanSize(bytes) {
  if (!bytes) return "0";
  const units = ["B", "kB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value.toFixed(value < 10 && unit > 0 ? 1 : 0)} ${units[unit]}`;
}

async function loadAvatarsHygiene() {
  const box = $("#avatars-hygiene");
  if (!box) return;
  try {
    const h = await getJSON("/api/avatars/hygiene");
    const bits = [
      `${h.avatars} ${t("avatars.hygVoices")}`,
      `${h.files} ${t("avatars.hygFiles")} (${humanSize(h.total_bytes)})`,
    ];
    if (h.orphans.length) bits.push(`${h.orphans.length} ${t("avatars.hygOrphans")} (${humanSize(h.orphan_bytes)})`);
    if (h.duplicates.length) bits.push(`${h.duplicates.length} ${t("avatars.hygDuplicates")}`);
    if (h.broken.length) bits.push(`${h.broken.length} ${t("avatars.hygBroken")}`);
    box.textContent = bits.join(" · ");
    if ($("#av-clean")) $("#av-clean").disabled = h.orphans.length === 0;
  } catch (e) {
    box.textContent = "";
  }
}

async function loadAvatars() {
  try {
    const data = await getJSON("/api/avatars");
    avatars = data.avatars || [];
  } catch (e) {
    avatars = [];
  }
  renderAvatars();
  buildTtsAvatarSelect();
  loadAvatarsHygiene();
}

$("#av-clean") && $("#av-clean").addEventListener("click", async () => {
  const box = $("#avatars-hygiene");
  box.textContent = "…";
  try {
    const r = await apiFetch("/api/avatars/hygiene/clean", { method: "POST" });
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || r.status);
    await loadAvatars();
    await loadAvatarsHygiene();
    if (body.removed.length) {
      box.textContent = `${body.removed.length} ${t("avatars.hygRemoved")} — ${box.textContent}`;
    }
  } catch (e) {
    box.textContent = `${t("status.error")} ${e}`;
  }
});

function renderAvatars() {
  const list = $("#avatars-list");
  list.innerHTML = "";
  if (!avatars.length) {
    const p = document.createElement("div");
    p.className = "muted";
    p.textContent = t("avatars.empty");
    list.appendChild(p);
    return;
  }
  avatars.forEach((a) => {
    const card = document.createElement("div");
    card.className = "avatar";
    const img = document.createElement("div");
    img.className = "avatar-photo";
    if (a.has_photo) img.style.backgroundImage = `url('/api/avatars/${encodeURIComponent(a.id)}/photo')`;
    card.appendChild(img);

    const info = document.createElement("div");
    info.className = "avatar-info";
    const nm = document.createElement("strong");
    nm.textContent = a.name;
    const meta = document.createElement("div");
    meta.className = "muted";
    const lang = a.language && a.language !== "Auto" ? ` · ${a.language}` : "";
    const xvec = a.x_vector_only ? ` · ${t("tts.xvec")}` : "";
    meta.textContent = `${t("avatars.kindClone")}${lang}${xvec}`;
    info.appendChild(nm); info.appendChild(meta);
    card.appendChild(info);

    const actions = document.createElement("div");
    actions.className = "avatar-actions";
    const mk = (label, cls, fn) => {
      const b = document.createElement("button");
      b.textContent = label; if (cls) b.className = cls;
      b.addEventListener("click", fn);
      return b;
    };
    actions.appendChild(mk(t("avatars.use"), "primary", () => {
      const entry = voiceEntries.find((v) => v.avatarId === a.id);
      if (entry) selectConversationVoice(entry);
      else loadConversationVoices().then(() => {
        const e2 = voiceEntries.find((v) => v.avatarId === a.id);
        if (e2) selectConversationVoice(e2);
      });
      switchTab("chat");
    }));
    actions.appendChild(mk(t("avatars.preview"), "", () => window.open(`/api/avatars/${encodeURIComponent(a.id)}/audio`, "_blank")));
    actions.appendChild(mk(t("avatars.edit"), "", () => openAvatarDialog(a.id)));
    card.appendChild(actions);
    list.appendChild(card);
  });
}

// x-vector only is implied when there is no reference transcript (ICL needs one).
function syncXvec(transcriptEl, xvecEl) {
  $(xvecEl).checked = !$(transcriptEl).value.trim();
}

function avatarPhotoPreview(id, name) {
  const el = $("#ad-photo-preview");
  const a = id ? avatars.find((x) => x.id === id) : null;
  el.textContent = "";
  if (a && a.has_photo) {
    el.style.background = "";
    el.style.backgroundImage = `url('/api/avatars/${encodeURIComponent(id)}/photo')`;
  } else {
    el.style.backgroundImage = "";
    el.style.background = name ? colorFor(name) : "var(--panel-2)";
    el.textContent = name ? initialOf(name) : "";
  }
}

function openAvatarDialog(id) {
  const dlg = $("#avatar-dialog");
  const editing = !!id;
  dlg.dataset.id = id || "";
  dlg.dataset.fingerprint = "";
  const a = editing ? avatars.find((x) => x.id === id) : null;
  $("#ad-title").textContent = editing ? t("avatars.editTitle") : t("avatars.createTitle");
  $("#ad-name").value = a ? (a.name || "") : "";
  $("#ad-lang").value = a ? (a.language || "Auto") : "English";
  $("#ad-personality").value = a ? (a.personality || "") : "";
  $("#ad-prompt").value = a ? (a.prompt || "") : "";
  $("#ad-reftext").value = a ? (a.ref_text || "") : "";
  $("#ad-xvec").checked = a ? !!a.x_vector_only : true;
  $("#ad-ref").value = "";
  $("#ad-photo").value = "";
  const fpAudio = $("#ad-fingerprint-audio");
  if (fpAudio) { fpAudio.hidden = true; fpAudio.removeAttribute("src"); }
  $("#ad-regenerate").hidden = !editing || !(a && (a.prompt || "").trim());
  $("#ad-delete").hidden = !editing;
  $("#ad-save").textContent = editing ? t("settings.save") : t("avatars.createBtn");
  $("#ad-status").textContent = "";
  avatarPhotoPreview(id, a ? a.name : "");
  if (dlg.showModal) dlg.showModal(); else dlg.setAttribute("open", "");
}

$("#av-new").addEventListener("click", () => openAvatarDialog(null));
$("#ad-reftext").addEventListener("input", () => syncXvec("#ad-reftext", "#ad-xvec"));
$("#ad-name").addEventListener("input", () => {
  if (!$("#ad-photo").files[0]) avatarPhotoPreview($("#avatar-dialog").dataset.id || null, $("#ad-name").value);
});
$("#ad-photo").addEventListener("change", () => {
  const f = $("#ad-photo").files[0];
  if (!f) return;
  const el = $("#ad-photo-preview");
  el.textContent = "";
  el.style.background = "";
  if (el.dataset.url) URL.revokeObjectURL(el.dataset.url);
  const url = URL.createObjectURL(f);
  el.dataset.url = url;
  el.style.backgroundImage = `url('${url}')`;
});
$("#ad-close").addEventListener("click", () => $("#avatar-dialog").close());

$("#ad-fingerprint").addEventListener("click", async () => {
  const prompt = $("#ad-prompt").value.trim();
  const status = $("#ad-status");
  if (!prompt) { status.textContent = t("avatars.needPrompt"); return; }
  status.textContent = t("avatars.generating");
  try {
    const fd = new FormData();
    fd.append("prompt", prompt);
    fd.append("language", $("#ad-lang").value);
    const r = await apiFetch("/api/avatars/fingerprint", { method: "POST", body: fd });
    if (!r.ok) {
      let detail = await r.text();
      detail = errorFromBody(detail);
      throw new Error(detail);
    }
    const blob = await r.blob();
    const dialog = $("#avatar-dialog");
    if (dialog.dataset.fingerprint) URL.revokeObjectURL(dialog.dataset.fingerprint);
    const url = URL.createObjectURL(blob);
    const el = $("#ad-fingerprint-audio");
    el.src = url; el.hidden = false;
    dialog.dataset.fingerprint = url;
    status.textContent = t("avatars.fingerprintReady");
  } catch (e) {
    status.textContent = `${t("status.error")} ${e}`;
  }
});

function avatarDialogForm() {
  const fd = new FormData();
  fd.append("name", $("#ad-name").value.trim());
  fd.append("language", $("#ad-lang").value);
  fd.append("personality", $("#ad-personality").value);
  fd.append("prompt", $("#ad-prompt").value);
  fd.append("ref_text", $("#ad-reftext").value);
  fd.append("x_vector_only", String($("#ad-xvec").checked));
  if ($("#ad-ref").files[0]) fd.append("ref_audio", $("#ad-ref").files[0]);
  if ($("#ad-photo").files[0]) fd.append("image", $("#ad-photo").files[0]);
  return fd;
}

async function reloadAvatarViews() {
  await loadAvatars();
  await loadConversationVoices();
  // An edited/regenerated voice invalidates the warm-up marks and preload cache.
  voiceCache.clear();
  voiceLoad.loaded = "";
  renderVoiceList();
}

$("#ad-save").addEventListener("click", async () => {
  const id = $("#avatar-dialog").dataset.id;
  const status = $("#ad-status");
  if (!$("#ad-name").value.trim()) { status.textContent = t("avatars.name"); return; }
  if (!id && !$("#ad-ref").files[0] && !$("#ad-prompt").value.trim()) {
    status.textContent = t("avatars.needAudioOrPrompt"); return;
  }
  status.textContent = "…";
  try {
    const r = await apiFetch(id ? `/api/avatars/${encodeURIComponent(id)}` : "/api/avatars", {
      method: id ? "PUT" : "POST", body: avatarDialogForm(),
    });
    if (!r.ok) {
      let detail = await r.text();
      detail = errorFromBody(detail);
      throw new Error(detail);
    }
    await reloadAvatarViews();
    $("#avatar-dialog").close();
  } catch (e) {
    status.textContent = `${t("status.error")} ${e}`;
  }
});

$("#ad-regenerate").addEventListener("click", async () => {
  const id = $("#avatar-dialog").dataset.id;
  if (!id) return;
  $("#ad-status").textContent = "…";
  try {
    const r = await apiFetch(`/api/avatars/${encodeURIComponent(id)}`, { method: "PUT", body: avatarDialogForm() });
    if (!r.ok) throw new Error((await r.text()).slice(0, 200));
    const reg = await apiFetch(`/api/avatars/${encodeURIComponent(id)}/regenerate`, { method: "POST" });
    if (!reg.ok) throw new Error((await reg.text()).slice(0, 200));
    await reloadAvatarViews();
    $("#ad-status").textContent = t("status.done");
  } catch (e) {
    $("#ad-status").textContent = `${t("status.error")} ${e}`;
  }
});

$("#ad-delete").addEventListener("click", async () => {
  const id = $("#avatar-dialog").dataset.id;
  if (!id || !window.confirm(`${t("avatars.delete")}?`)) return;
  try {
    await apiFetch(`/api/avatars/${encodeURIComponent(id)}`, { method: "DELETE" });
    await reloadAvatarViews();
    $("#avatar-dialog").close();
  } catch (e) {
    $("#ad-status").textContent = `${t("status.error")} ${e}`;
  }
});

// ------------------------------------------------ conversation voice selector
function colorFor(name) {
  let h = 0;
  const s = name || "?";
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
  return `hsl(${h}, 55%, 45%)`;
}

function initialOf(name) { return (name || "?").trim().charAt(0).toUpperCase() || "?"; }

async function loadConversationVoices() {
  try { avatars = (await getJSON("/api/avatars")).avatars || []; } catch (e) { /* ignore */ }
  let presets = [];
  try { presets = (await getJSON("/api/voices?family=custom_voice")).speakers || []; } catch (e) { /* ignore */ }
  voiceEntries = [
    ...avatars.map((a) => ({
      id: `av:${a.id}`, kind: "avatar", avatarId: a.id,
      name: a.name, has_photo: a.has_photo, personality: a.personality || "",
    })),
    ...presets.map((s) => ({ id: `cv:${s}`, kind: "preset", speaker: s, name: s, personality: "" })),
  ];
  if (!voiceEntries.some((v) => v.id === selectedVoiceId)) {
    selectedVoiceId = voiceEntries.length ? voiceEntries[0].id : "";
  }
  renderVoiceList();
}

function selectedVoice() { return voiceEntries.find((v) => v.id === selectedVoiceId) || null; }

const voiceLoad = { inFlight: "", loaded: "" };
const voiceCache = new Set();

function renderVoiceList() {
  const list = $("#chat-voice-list");
  if (!list) return;
  list.innerHTML = "";
  if (!voiceEntries.length) {
    const p = document.createElement("div");
    p.className = "muted"; p.textContent = t("avatars.empty");
    list.appendChild(p);
    return;
  }
  voiceEntries.forEach((v) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "voice-item" + (v.id === selectedVoiceId ? " selected" : "");
    const icon = document.createElement("span");
    icon.className = "voice-icon";
    if (v.has_photo) icon.style.backgroundImage = `url('/api/avatars/${encodeURIComponent(v.avatarId)}/photo')`;
    else { icon.style.background = colorFor(v.name); icon.textContent = initialOf(v.name); }
    item.appendChild(icon);
    const meta = document.createElement("span");
    meta.className = "voice-meta";
    const nm = document.createElement("strong"); nm.textContent = v.name; meta.appendChild(nm);
    const sub = document.createElement("small"); sub.className = "muted";
    sub.textContent = v.kind === "avatar" ? t("chat.voiceAvatar") : t("chat.voicePresetName");
    meta.appendChild(sub);
    item.appendChild(meta);
    if (voiceLoad.inFlight === v.id) {
      const sp = document.createElement("span"); sp.className = "spinner"; item.appendChild(sp);
    } else if (voiceLoad.loaded === v.id) {
      const ok = document.createElement("span"); ok.className = "voice-ok"; ok.textContent = "✓"; item.appendChild(ok);
    }
    item.addEventListener("click", () => selectConversationVoice(v));
    list.appendChild(item);
  });
  const cur = selectedVoice();
  const wrap = $("#chat-lang-wrap");
  if (wrap) wrap.hidden = !cur || cur.kind === "avatar";
  updateChatEngineControl();
}

function voiceLoadBody(v) {
  return v.kind === "avatar" ? { avatar: v.avatarId } : { family: "custom_voice" };
}

async function selectConversationVoice(v) {
  selectedVoiceId = v.id;
  localStorage.setItem("chatVoice", selectedVoiceId);
  renderVoiceList();
  sendChatConfig();
  await loadVoiceModel(v);
}

async function loadVoiceModel(v) {
  if (voiceCache.has(v.id)) { voiceLoad.loaded = v.id; renderVoiceList(); return; }
  voiceLoad.inFlight = v.id;
  voiceLoad.loaded = "";
  renderVoiceList();
  if ($("#chat-status")) $("#chat-status").textContent = t("chat.loadingVoice");
  try {
    const r = await apiFetch("/api/voices/load", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(voiceLoadBody(v)),
    });
    if (!r.ok) throw new Error((await r.text()).slice(0, 200));
    voiceCache.add(v.id);
    voiceLoad.loaded = v.id;
    if ($("#chat-status")) $("#chat-status").textContent = "";
  } catch (e) {
    if ($("#chat-status")) $("#chat-status").textContent = `${t("status.error")} ${e}`;
  } finally {
    voiceLoad.inFlight = "";
    renderVoiceList();
  }
}

function chatLang() {
  const s = $("#chat-language");
  return s && s.value ? s.value : "Auto";
}

// The Conversation tab reuses its own language selector for the STT input too
// (the microphone/VAD): "Auto" lets the model decide, anything else becomes the
// STT language hint — the gateway normalizes the name to a code.
function chatSttLanguage() {
  const lang = chatLang();
  return lang && lang !== "Auto" ? lang : "";
}

function chatCloneEngine() {
  const sel = $("#chat-clone-engine");
  return (sel && sel.value) || (CONFIG && CONFIG.models && CONFIG.models.clone_backend) || "qwen3";
}

// Cloned voices let the user pick the engine per session; built-in voices have a
// forced engine (Qwen3-TTS) so the selector is hidden.
function updateChatEngineControl() {
  const v = selectedVoice();
  const isClone = !!(v && v.kind === "avatar");
  if ($("#chat-engine-select-wrap")) $("#chat-engine-select-wrap").hidden = !isClone;
  if ($("#chat-engine-fixed")) $("#chat-engine-fixed").hidden = isClone;
}

function initCloneEngineSelects() {
  const backend = (CONFIG && CONFIG.models && CONFIG.models.clone_backend) || "qwen3";
  if ($("#chat-clone-engine")) $("#chat-clone-engine").value = backend;
  fillCloneEngineSelect($("#set-clone-backend"));
  if ($("#set-clone-backend")) {
    $("#set-clone-backend").value = cloneEngineValue((CONFIG && CONFIG.models) || {});
  }
}

function voiceConfig() {
  const v = selectedVoice();
  if (!v) return { family: "custom_voice", language: chatLang() };
  if (v.kind === "avatar") return { avatar: v.avatarId, clone_backend: chatCloneEngine() };
  return { family: "custom_voice", speaker: v.speaker, language: chatLang() };
}

function chatPersonality() {
  const v = selectedVoice();
  return v && v.kind === "avatar" ? (v.personality || "") : "";
}

// ------------------------------------------------ reusable STT model picker
function sttModelDesc(m) {
  const id = `${m.repo} ${(m.aliases || []).join(" ")}`.toLowerCase();
  if (id.includes("nemotron-3.5")) return t("stt.desc.nemotron3_5");
  if (id.includes("parakeet-tdt")) return t("stt.desc.parakeet_v3");
  if (id.includes("nemotron-speech-streaming-en")) return t("stt.desc.nemotron_en");
  return t("stt.desc.generic");
}

function sttModelState(m) {
  if (!m) return "absent";
  if (m.active && sttService.ready) return "loaded";
  if (m.active) return "starting";
  if (m.cached) return "cached";
  return "absent";
}

function sttStateLabel(state) { return t(`stt.state.${state}`); }

class SttPicker {
  constructor(root, opts = {}) {
    this.root = root;
    this.opts = Object.assign({
      load: true, stop: true, refresh: true, desc: true, status: true, initial: null,
      empty: false, label: null,
    }, opts);
    this.value = "";
    this.touched = false;
    this.build();
    sttPickers.push(this);
    this.populate();
  }

  build() {
    this.root.classList.add("stt-picker");
    this.root.innerHTML = "";
    const row = document.createElement("div"); row.className = "row";
    const label = document.createElement("label"); label.className = "grow";
    this.labelSpan = document.createElement("span"); label.appendChild(this.labelSpan);
    this.select = document.createElement("select"); label.appendChild(this.select);
    row.appendChild(label);
    this.root.appendChild(row);
    if (this.opts.load) this.loadBtn = this.addButton(t("stt.load"), "primary", () => this.download());
    if (this.opts.stop) {
      this.stopBtn = this.addButton(t("stt.stop"), "", () => { if (sttDownloadAbort) sttDownloadAbort.abort(); });
      this.stopBtn.hidden = true;
    }
    if (this.opts.refresh) this.addButton(t("stt.refresh"), "", () => loadSttModels());
    if (this.opts.desc) { this.desc = document.createElement("div"); this.desc.className = "muted"; this.root.appendChild(this.desc); }
    if (this.opts.load) { this.progress = document.createElement("progress"); this.progress.max = 100; this.progress.value = 0; this.progress.hidden = true; this.root.appendChild(this.progress); }
    if (this.opts.status) { this.status = document.createElement("div"); this.status.className = "muted"; this.root.appendChild(this.status); }
    this.select.addEventListener("change", () => { this.value = this.select.value; this.touched = true; this.update(); updateVadAvailability(); });
  }

  addButton(text, cls, fn) {
    const b = document.createElement("button");
    b.textContent = text; if (cls) b.className = cls;
    b.addEventListener("click", fn);
    this.root.querySelector(".row").appendChild(b);
    return b;
  }

  resolveInitial() {
    const want = this.opts.initial && this.opts.initial();
    if (!want) return "";
    const m = sttModels.find((x) => x.repo === want || x.filename === want);
    return m ? m.repo : "";
  }

  populate() {
    const prev = this.touched ? this.value : (this.resolveInitial() || this.value);
    this.labelSpan.textContent = t(this.opts.label || "stt.model");
    this.select.innerHTML = "";
    if (this.opts.empty) {
      // "Do not preload anything": an explicit choice, not a missing value.
      const none = document.createElement("option");
      none.value = "";
      none.textContent = t("stt.preloadNone");
      this.select.appendChild(none);
    }
    if (!sttModels.length) {
      const o = document.createElement("option"); o.value = ""; o.textContent = t("stt.indexUnavailable"); this.select.appendChild(o);
    }
    sttModels.forEach((m) => {
      const o = document.createElement("option");
      o.value = m.repo;
      o.textContent = `${m.filename || m.repo} · ${sttStateLabel(sttModelState(m))} · ${fmtMB(m.size)} ${t("stt.sizeMb")}`;
      this.select.appendChild(o);
    });
    if (prev && sttModels.some((m) => m.repo === prev)) {
      this.select.value = prev;
    } else if (prev === "" && this.opts.empty) {
      this.select.value = "";
    } else if (sttModels.length) {
      // sttActive is the active model *filename*; options are keyed by repo.
      const active = sttModels.find((m) => m.filename === sttActive || m.repo === sttActive);
      this.select.value = (active || sttModels[0]).repo;
    }
    this.value = this.select.value;
    this.update();
  }

  current() { return sttModels.find((x) => x.repo === this.select.value) || null; }

  update() {
    const m = this.current();
    const downloading = !!sttDownload && sttDownload.repo === this.select.value;
    if (this.desc) {
      if (!m && this.opts.empty) this.desc.textContent = t("stt.preloadNoneHint");
      else this.desc.textContent = m ? sttModelDesc(m) : "";
    }
    if (this.loadBtn) {
      this.loadBtn.textContent = m && m.cached ? t("stt.load") : t("stt.download");
      this.loadBtn.disabled = !m || !!sttDownload || (m.active && sttService.ready);
    }
    if (this.stopBtn) this.stopBtn.hidden = !downloading;
    if (this.progress) {
      this.progress.hidden = !downloading;
      if (downloading && sttDownload.total) this.progress.value = Math.round((sttDownload.downloaded / sttDownload.total) * 100);
    }
    if (this.status) this.status.textContent = this.statusText(m);
  }

  statusText(m) {
    if (sttError && sttError.repo === this.select.value) return sttError.message;
    if (sttDownload && sttDownload.repo === this.select.value) {
      return `${t("stt.downloading")} ${fmtMB(sttDownload.downloaded)} / ${fmtMB(sttDownload.total)} ${t("stt.sizeMb")}`;
    }
    if (!m) return t("stt.indexUnavailable");
    const state = sttModelState(m);
    if (state === "loaded") return t("stt.state.loadedHint");
    if (state === "starting") return t("stt.state.startingHint");
    if (state === "cached") return t("stt.state.cachedHint");
    return t("stt.state.absentHint");
  }

  download() { const repo = this.select.value; if (repo) downloadSttModel(repo); }
}

let sttError = null;
let settingsSttPicker = null;
let chatSttPicker = null;

function mountSttPickers() {
  const sttRoot = $("#stt-picker");
  if (sttRoot && !sttRoot.dataset.mounted) { sttRoot.dataset.mounted = "1"; new SttPicker(sttRoot); }
  const chatRoot = $("#chat-stt-picker");
  if (chatRoot && !chatRoot.dataset.mounted) {
    chatRoot.dataset.mounted = "1";
    chatSttPicker = new SttPicker(chatRoot, { load: true, stop: false, refresh: false, desc: false, status: false });
  }
  const setRoot = $("#set-startup-stt-picker");
  if (setRoot && !setRoot.dataset.mounted) {
    setRoot.dataset.mounted = "1";
    // Same picker as the Transcription tab (model list, state, description), plus
    // an explicit "do not preload" entry and an explicit label: this card talks
    // about STT, not about the TTS models.
    settingsSttPicker = new SttPicker(setRoot, {
      load: false, stop: false, refresh: false, desc: true, status: false,
      empty: true, label: "settings.startupStt",
      initial: () => (CONFIG ? CONFIG.startup.stt_model : ""),
    });
  }
}

function renderSttPickers() { sttPickers.forEach((p) => p.populate()); updateVadAvailability(); renderSttLanguage(); }

// Input language for the STT requests: a single select over the configured
// `stt.languages`, or the service's language list when none is configured. The
// empty value is "auto" (the model's own default: detection for the multilingual
// models). The service takes a single code, so only one is ever sent.
function sttLanguageOptions() {
  const configured = (CONFIG && CONFIG.stt && CONFIG.stt.languages) || [];
  if (configured.length) return configured.slice();
  return ((CONFIG && CONFIG.languages) || []).filter((l) => l && l !== "Auto");
}
function sttLanguage() {
  const select = $("#stt-lang");
  return (select && select.value) || localStorage.getItem("sttLanguage") || "";
}
function setSttLanguage(value) {
  if (value) localStorage.setItem("sttLanguage", value);
  else localStorage.removeItem("sttLanguage");
  localStorage.removeItem("sttLanguages");  // superseded by the single value
}
function renderSttLanguage() {
  const select = $("#stt-lang");
  if (!select) return;
  const options = sttLanguageOptions();
  const configured = (CONFIG && CONFIG.stt && CONFIG.stt.language) || "";
  const wanted = localStorage.getItem("sttLanguage") || configured || "";
  select.innerHTML = "";
  const auto = document.createElement("option");
  auto.value = "";
  auto.textContent = t("stt.langAuto");
  select.appendChild(auto);
  options.forEach((code) => {
    const option = document.createElement("option");
    option.value = code;
    option.textContent = code;
    select.appendChild(option);
  });
  select.value = options.includes(wanted) ? wanted : "";
}
$("#stt-lang") && $("#stt-lang").addEventListener("change", (e) => setSttLanguage(e.target.value));

// The chat VAD needs a streaming-capable STT model (and a reachable service).
function chatSttModel() {
  if (!chatSttPicker) return null;
  if (typeof chatSttPicker.current === "function") {
    const cur = chatSttPicker.current();
    if (cur) return cur;
  }
  const repo = chatSttPicker.value;
  return sttModels.find((m) => m.repo === repo) || null;
}

function vadAvailable() {
  const model = chatSttModel();
  return !!(model && model.streaming && sttService.ready);
}

function updateVadAvailability() {
  const btn = $("#chat-vad");
  if (!btn) return;
  const ok = vadAvailable();
  // Only shown when a streaming STT model is active (and the service is up).
  btn.hidden = !ok;
  if (!ok && dictateActive && dictateVadActive) stopDictation();
  updateVadButton();
}

function updateVadButton() {
  const btn = $("#chat-vad");
  if (!btn) return;
  btn.classList.toggle("primary", !!(dictateActive && dictateVadActive));
}

async function loadSttModels() {
  try {
    const data = await getJSON("/api/stt/models");
    if (!data.available) { sttModels = []; sttActive = ""; }
    else { sttModels = data.models || []; sttActive = data.active || ""; }
  } catch (e) { sttModels = []; sttActive = ""; }
  await refreshSttServiceStatus();
  renderSttPickers();
}

async function refreshSttServiceStatus() {
  try {
    const h = await getJSON("/api/stt/health");
    sttService = { enabled: h.enabled !== false, ready: h.enabled !== false && !h.error, error: h.error || "" };
  } catch (e) {
    sttService = { enabled: true, ready: false, error: String(e) };
  }
}

async function downloadSttModel(repo) {
  if (!repo || sttDownload) return;
  sttError = null;
  sttDownload = { repo, downloaded: 0, total: 0 };
  sttDownloadAbort = new AbortController();
  renderSttPickers();
  try {
    const r = await apiFetch("/api/stt/download", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: repo }), signal: sttDownloadAbort.signal,
    });
    if (!r.ok) throw new Error((await r.text()).slice(0, 200));
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const part of parts) {
        const line = part.replace(/^data:\s*/, "").trim();
        if (!line) continue;
        let ev; try { ev = JSON.parse(line); } catch (e) { continue; }
        if (ev.status === "downloading" || ev.status === "cached") {
          sttDownload.downloaded = ev.downloaded || 0;
          sttDownload.total = ev.total || sttDownload.total;
        } else if (ev.status === "done") {
          sttDownload.downloaded = sttDownload.total || sttDownload.downloaded;
        } else if (ev.status === "error") {
          sttError = { repo, message: `${t("status.error")} ${ev.message}` };
        }
        renderSttPickers();
      }
    }
    await loadSttModels();
  } catch (e) {
    sttError = { repo, message: e.name === "AbortError" ? t("stt.cancelled") : `${t("status.error")} ${e}` };
  } finally {
    sttDownload = null; sttDownloadAbort = null;
    renderSttPickers();
  }
}

// mic device
async function loadMicDevices() {
  const sel = $("#stt-mic-device");
  if (!sel) return;
  const keep = sel.value || micDeviceId;
  sel.innerHTML = "";
  const def = document.createElement("option");
  def.value = ""; def.textContent = t("stt.defaultMic"); sel.appendChild(def);
  if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      devices.filter((d) => d.kind === "audioinput").forEach((d, i) => {
        const o = document.createElement("option");
        o.value = d.deviceId; o.textContent = d.label || `Microphone ${i + 1}`; sel.appendChild(o);
      });
    } catch (e) { /* permission not granted */ }
  }
  sel.value = keep || "";
  micDeviceId = sel.value;
}
$("#stt-mic-device").addEventListener("change", (e) => { micDeviceId = e.target.value; });

// Re-enumerate when a device is plugged/unplugged, and fall back to the default
// device when the selected one disappeared (e.g. after swapping microphones).
if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => { loadMicDevices(); });
}

async function openMicStream() {
  // Enable browser echo cancellation / noise suppression: this is what makes
  // hands-free barge-in usable without headphones.
  const audio = { echoCancellation: true, noiseSuppression: true, autoGainControl: true };
  if (!micDeviceId) return navigator.mediaDevices.getUserMedia({ audio });
  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: Object.assign({ deviceId: { exact: micDeviceId } }, audio),
    });
  } catch (e) {
    if (e && ["OverconstrainedError", "NotFoundError", "DevicesNotFoundError"].includes(e.name)) {
      micDeviceId = "";
      const sel = $("#stt-mic-device"); if (sel) sel.value = "";
      return navigator.mediaDevices.getUserMedia({ audio });
    }
    throw e;
  }
}

// Microphone capture with an AudioWorklet (preferred) and a ScriptProcessor
// fallback. Calls onSamples(Float32Array, sampleRate) for every captured frame.
async function startMicCapture(onSamples) {
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(t("stt.micInsecure"));
  }
  const stream = await openMicStream();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === "suspended") { try { await ctx.resume(); } catch (e) { /* ignore */ } }
  const source = ctx.createMediaStreamSource(stream);
  let node = null, sink = null, worklet = false;
  if (ctx.audioWorklet) {
    try {
      await ctx.audioWorklet.addModule("/ui/pcm-worklet.js");
      node = new AudioWorkletNode(ctx, "pcm-capture");
      node.port.onmessage = (e) => onSamples(e.data, ctx.sampleRate);
      sink = ctx.createGain(); sink.gain.value = 0;  // keep the graph alive, silent
      node.connect(sink); sink.connect(ctx.destination);
      worklet = true;
    } catch (e) { node = null; sink = null; }
  }
  if (!node) {
    node = ctx.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = (e) => onSamples(e.inputBuffer.getChannelData(0), ctx.sampleRate);
    node.connect(ctx.destination);
  }
  source.connect(node);
  return { ctx, stream, source, node, sink, worklet };
}

// STT microphone
let micCtx = null, micStream = null, micSource = null, micNode = null, micSink = null, sttSocket = null, sttFinal = "", sttPartial = "";

$("#stt-record").addEventListener("click", async () => {
  const btn = $("#stt-record");
  if (sttSocket) { stopMic(); return; }
  try {
    const cap = await startMicCapture((samples, sr) => {
      if (!sttSocket || sttSocket.readyState !== 1) return;
      sttSocket.send(downsampleTo16k(samples, sr).buffer);
    });
    micCtx = cap.ctx; micStream = cap.stream; micSource = cap.source; micNode = cap.node; micSink = cap.sink;
    loadMicDevices();
    const sock = new WebSocket(wsURL("/ws/stt"));
    sock.binaryType = "arraybuffer";
    sttSocket = sock;
    sttFinal = $("#stt-text").textContent;
    sttPartial = "";

    sock.onopen = () => {
      if (sttSocket !== sock) { try { sock.close(); } catch (e) { /* aborted */ } return; }
      sock.send(JSON.stringify({
        type: "session.update",
        session: { sample_rate: 16000, language: sttLanguage() || undefined, automatic_punctuation: true },
      }));
      $("#stt-status").textContent = t("stt.recording");
      btn.textContent = t("stt.stop");
      btn.classList.remove("primary");
    };
    sock.onmessage = (ev) => {
      if (sttSocket !== sock) return;
      if (typeof ev.data !== "string") return;
      let msg; try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.type && msg.type.includes("delta")) {
        sttPartial += (msg.delta || "");
        $("#stt-text").textContent = sttFinal + sttPartial;
      } else if (msg.type && msg.type.includes("completed")) {
        sttFinal += (msg.transcript || msg.text || "") + " ";
        sttPartial = "";
        $("#stt-text").textContent = sttFinal;
      } else if (msg.type === "error") {
        $("#stt-status").textContent = msg.message || t("stt.serviceNotReady");
      }
    };
    sock.onerror = () => {
      if (sttSocket === sock) $("#stt-status").textContent = t("stt.serviceNotReady");
    };
    sock.onclose = (ev) => {
      if (sttSocket !== sock) return;  // already stopped/replaced: not our business
      stopMic();
      if (ev && ev.code !== 1000 && ev.code !== 4401) $("#stt-status").textContent = t("stt.serviceNotReady");
    };
  } catch (e) {
    $("#stt-status").textContent = `${t("status.micUnavailable")}: ${e}`;
  }
});

function stopMic() {
  if (micSource) micSource.disconnect();
  if (micNode) micNode.disconnect();
  if (micSink) micSink.disconnect();
  if (micCtx) micCtx.close();
  if (micStream) micStream.getTracks().forEach((track) => track.stop());
  const sock = sttSocket;
  sttSocket = null;  // null *before* closing, so onclose() knows this is not its socket
  if (sock) {
    try {
      if (sock.readyState === 1) sock.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
      sock.close();  // also aborts a socket still CONNECTING
    } catch (e) { /* ignore */ }
  }
  micCtx = null; micStream = null; micSource = null; micNode = null; micSink = null;
  $("#stt-status").textContent = t("stt.stopped");
  const btn = $("#stt-record"); btn.textContent = t("stt.start"); btn.classList.add("primary");
}

$("#stt-upload").addEventListener("click", async () => {
  const file = $("#stt-file").files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  fd.append("response_format", $("#stt-diar").checked ? "verbose_json" : "json");
  if ($("#stt-diar").checked) fd.append("diarization", "true");
  { const lang = sttLanguage(); if (lang) fd.append("language", lang); }
  try {
    const r = await apiFetch("/v1/audio/transcriptions", { method: "POST", body: fd });
    const text = await r.text();
    let data = null;
    try { data = JSON.parse(text); } catch (e) { data = null; }
    if (!r.ok || data === null) {
      $("#stt-file-out").textContent = `${t("status.error")} ${(data && data.detail) || text || r.status}`;
      return;
    }
    $("#stt-file-out").textContent = JSON.stringify(data.words ? { text: data.text, words: data.words } : data, null, 2);
  } catch (e) {
    $("#stt-file-out").textContent = `${t("status.error")} ${e}`;
  }
});

// ----------------------------------------------------------------- chat tab
let chatSocket = null, chatPlayer = null, currentAssistant = null, chatBusy = false;
let chatConnectPromise = null;  // in-flight connect, so two sends share one socket
let chatTurnSeq = 0;            // client turn ids, echoed by the server
let activeTurn = 0;
let chatSentAt = 0;  // performance.now() of the last send, for the audio-start log

// Replay cache: keep the last N assistant messages' PCM in memory; older ones
// are dropped and regenerated on demand via /api/tts.
const CHAT_AUDIO_KEEP = 8;
const chatAudioCache = new Map();
let chatAudioSeq = 0;
let currentAudio = null;
let replayPlayer = null;

function newAudioRecord() {
  const id = ++chatAudioSeq;
  const record = { id, text: "", chunks: [], sr: TTS_SESSION.sample_rate };
  chatAudioCache.set(id, record);
  while (chatAudioCache.size > CHAT_AUDIO_KEEP) {
    const oldest = chatAudioCache.keys().next().value;
    chatAudioCache.delete(oldest);
  }
  return record;
}

function playPcmChunks(chunks, sr) {
  if (replayPlayer) { replayPlayer.stop(); replayPlayer = null; }
  replayPlayer = new PCMPlayer(sr || TTS_SESSION.sample_rate);
  chunks.forEach((chunk) => replayPlayer.pushBytes(chunk));
}

async function replayMessage(record, btn) {
  if (!record) return;
  if (record.chunks && record.chunks.length) {
    playPcmChunks(record.chunks, record.sr);
    return;
  }
  const text = (record.text || "").trim();
  if (!text) return;
  btn.disabled = true;
  btn.textContent = "…";
  try {
    const payload = Object.assign(
      { text, format: "wav", stream: false, chunking: true },
      voiceConfig(),
    );
    const r = await apiFetch("/api/tts", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error((await r.text()).slice(0, 160));
    const url = URL.createObjectURL(await r.blob());
    const audio = new Audio(url);
    audio.addEventListener("ended", () => URL.revokeObjectURL(url));
    await audio.play();
  } catch (e) {
    $("#chat-status").textContent = `${t("status.error")} ${e}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "🔊";
  }
}

function addReplayButton(stack, record) {
  const actions = document.createElement("div");
  actions.className = "bubble-actions";
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "bubble-replay";
  btn.title = t("chat.replay");
  btn.textContent = "🔊";
  btn.addEventListener("click", () => replayMessage(record, btn));
  actions.appendChild(btn);
  stack.appendChild(actions);
}

// Follow the newest message, unless the user scrolled up to read something: then
// their position is left alone until they come back to the bottom.
function chatLogEl() { return $("#chat-log"); }
function chatFollowsBottom(slack = 90) {
  const el = chatLogEl();
  if (!el) return true;
  return el.scrollHeight - el.scrollTop - el.clientHeight < slack;
}
function scrollChatToBottom(force = false) {
  const el = chatLogEl();
  if (!el) return;
  if (!force && !chatFollowsBottom()) return;
  el.scrollTop = el.scrollHeight;
}

function appendTurn(role, text, name) {
  const row = document.createElement("div");
  row.className = `bubble-row ${role}`;
  let bubble;
  if (role === "assistant") {
    const icon = document.createElement("span");
    icon.className = "bubble-icon";
    const v = selectedVoice();
    if (v && v.has_photo) icon.style.backgroundImage = `url('/api/avatars/${encodeURIComponent(v.avatarId)}/photo')`;
    else { icon.style.background = colorFor(name || (v ? v.name : "AI")); icon.textContent = initialOf(name || (v ? v.name : "AI")); }
    row.appendChild(icon);

    const stack = document.createElement("div");
    stack.className = "bubble-stack";
    bubble = document.createElement("div");
    bubble.className = "bubble assistant";
    bubble.textContent = text;
    stack.appendChild(bubble);

    const record = newAudioRecord();
    record.bubble = bubble;
    currentAudio = record;
    addReplayButton(stack, record);
    row.appendChild(stack);
  } else {
    bubble = document.createElement("div");
    bubble.className = `bubble ${role}`;
    bubble.textContent = text;
    row.appendChild(bubble);
  }
  $("#chat-log").appendChild(row);
  scrollChatToBottom(true);
  return bubble;
}

function chatConfig() {
  return {
    type: "session.config",
    speak: $("#chat-speak").checked,
    personality: chatPersonality(),
    tts: Object.assign({
      format: "pcm",
      chunking: true,
    }, voiceConfig()),
  };
}

function sendChatConfig() {
  if (chatSocket && chatSocket.readyState === 1) chatSocket.send(JSON.stringify(chatConfig()));
}

function ensureChatSocket() {
  if (chatSocket && chatSocket.readyState === 1) return Promise.resolve(chatSocket);
  if (chatConnectPromise) return chatConnectPromise;  // dedupe concurrent sends
  chatConnectPromise = new Promise((resolve, reject) => {
    const sock = new WebSocket(wsURL("/ws/chat"));
    sock.binaryType = "arraybuffer";
    sock.onopen = () => {
      chatConnectPromise = null;
      chatSocket = sock;
      sock.onmessage = onChatMessage;
      sock.send(JSON.stringify(chatConfig()));
      resolve(sock);
    };
    sock.onerror = () => { chatConnectPromise = null; reject(new Error("connection failed")); };
    sock.onclose = () => { if (chatSocket === sock) chatSocket = null; };
  });
  return chatConnectPromise;
}

function resetChatAudio() {
  if (chatPlayer) { chatPlayer.stop(); chatPlayer = null; }
  if (replayPlayer) { replayPlayer.stop(); replayPlayer = null; }
}

function onChatMessage(ev) {
  if (typeof ev.data !== "string") {
    const bytes = new Uint8Array(ev.data);
    if (chatPlayer && $("#chat-speak").checked) chatPlayer.pushBytes(bytes);
    if (currentAudio && currentAudio.chunks) currentAudio.chunks.push(bytes);
    return;
  }
  let msg; try { msg = JSON.parse(ev.data); } catch (e) { return; }
  // Events of a turn the client has already moved past must not touch the current
  // one: after a barge-in / new message the server can still emit the previous
  // turn's `turn.cancelled`, `error`, remaining `llm.delta` or audio. The optional
  // `turn` id (echoed by the server) is what tells them apart.
  if (msg.turn != null && msg.turn !== activeTurn) return;
  if (msg.type === "llm.delta") {
    if (currentAssistant) currentAssistant.textContent += msg.text;
    // The bubble grows as tokens arrive: keep it in view while the user is at the
    // bottom of the log (their position is preserved otherwise).
    scrollChatToBottom();
  } else if (msg.type === "audio.start") {
    // Debug aid: the time between sending the message and the first audio packet,
    // directly comparable with the server's "chat tts: first audio" log line.
    if (chatSentAt) {
      console.info(`[chat] first audio ${Math.round(performance.now() - chatSentAt)} ms after send`);
      chatSentAt = 0;
    }
    // Ignore audio for a cancelled turn (chatBusy is false after interrupt).
    if (msg.sample_rate && msg.sample_rate !== TTS_SESSION.sample_rate) {
      if (chatPlayer) { chatPlayer.stop(); chatPlayer = null; }
      TTS_SESSION.sample_rate = msg.sample_rate;
    }
    if (chatBusy && !chatPlayer) chatPlayer = new PCMPlayer(TTS_SESSION.sample_rate);
  } else if (msg.type === "llm.done") {
    if (currentAssistant && !currentAssistant.textContent) currentAssistant.textContent = msg.text || "";
    if (currentAudio) { currentAudio.text = (currentAssistant ? currentAssistant.textContent : msg.text) || ""; currentAudio = null; }
    currentAssistant = null;
    chatBusy = false;
    scrollChatToBottom();
  } else if (msg.type === "turn.cancelled") {
    if (currentAssistant) currentAssistant.classList.add("cancelled");
    if (currentAudio) { currentAudio.text = currentAssistant ? currentAssistant.textContent : ""; currentAudio = null; }
    currentAssistant = null;
    chatBusy = false;
    resetChatAudio();
    scrollChatToBottom();
  } else if (msg.type === "error") {
    if (currentAssistant) currentAssistant.textContent += ` [error: ${msg.message}]`;
    if (currentAudio) { currentAudio.text = currentAssistant ? currentAssistant.textContent : ""; currentAudio = null; }
    currentAssistant = null;
    chatBusy = false;
    $("#chat-status").textContent = msg.message;
    scrollChatToBottom();
  }
}

async function chatSend() {
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  // A new turn replaces whatever is still playing: never let two replies overlap.
  if (chatBusy || assistantSpeaking()) interruptChat("");
  const turn = ++chatTurnSeq;
  activeTurn = turn;
  appendTurn("user", text);
  currentAssistant = appendTurn("assistant", "");
  $("#chat-status").textContent = "";
  chatBusy = true;
  chatSentAt = performance.now();
  try {
    const sock = await ensureChatSocket();
    sock.send(JSON.stringify(chatConfig()));
    sock.send(JSON.stringify({ type: "input.text", text, turn }));
  } catch (e) {
    if (currentAssistant) currentAssistant.textContent = `${t("status.error")} ${e}`;
    currentAssistant = null;
    chatBusy = false;
    $("#chat-status").textContent = `${t("status.error")} ${e}`;
  }
}

$("#chat-send").addEventListener("click", chatSend);
$("#chat-input").addEventListener("keydown", (e) => {
  // Enter confirms an IME candidate (CJK input): only a plain Enter sends.
  if (e.key === "Enter" && !e.isComposing && e.keyCode !== 229) chatSend();
});
$("#chat-speak").addEventListener("change", sendChatConfig);
$("#chat-language").addEventListener("change", sendChatConfig);
$("#chat-clone-engine").addEventListener("change", sendChatConfig);
$("#chat-stop").addEventListener("click", () => interruptChat(t("chat.stopped")));

// Chat dictation: push-to-talk, or continuous with voice-activity detection (VAD).
let dictateActive = false, dictateCtx = null, dictateStream = null, dictateSource = null, dictateNode = null, dictateSink = null;
let dictateSocket = null, dictateText = "", dictatePartial = "", dictateTimer = null;
let dictateVadActive = false, dictateAnalyser = null, dictateVadTimer = null;
let vadSpeaking = false, vadLastVoice = 0;
// Absolute floor for a quiet room (AGC makes levels vary a lot), plus the margin
// the voice must show above the *measured* ambient floor: a fixed RMS cannot work
// across microphones, gains and echo-cancellation residuals.
const VAD_MIN_RMS = 0.008;
const VAD_SNR = 3.0;
const VAD_SILENCE_MS = 900;         // silence after speech => end of utterance
const VAD_BARGE_MS = 260;           // voiced time (without a long dip) that barges in
const VAD_BARGE_HOLD_MS = 260;      // dip tolerated between two syllables
const VAD_DUCK_FROM_MS = 100;       // duck the assistant as soon as the user starts
const VAD_DUCK_VOLUME = 0.35;

// Is the assistant still audible? `chatBusy` covers the server side, but the client
// plays from a jitter buffer: without this the tail of a reply counts as silence and
// barge-in would not interrupt anything.
function assistantSpeaking() {
  if (chatBusy) return true;
  return !!(chatPlayer && chatPlayer.remainingSeconds() > 0.05);
}

// Mic samples held back while the assistant speaks (~0.6 s), sent as soon as it stops.
let dictatePending = [];
const DICTATE_PENDING_MAX = 12;

function flushDictatePending() {
  if (!dictatePending.length) return;
  const chunks = dictatePending;
  dictatePending = [];
  if (!dictateSocket || dictateSocket.readyState !== 1) return;
  for (const chunk of chunks) {
    try { dictateSocket.send(chunk.buffer); } catch (e) { /* socket gone */ }
  }
}

// Endpoint sooner when the streaming transcript already looks like a complete
// sentence (lower time-to-response without cutting the user off mid-phrase).
function vadSilenceMs() {
  const partial = (dictatePartial || "").trim();
  if (partial && /[.!?…。！？]$/.test(partial)) return Math.round(VAD_SILENCE_MS * 0.5);
  return VAD_SILENCE_MS;
}

function takeUtteranceText() {
  const text = (dictateText + dictatePartial).trim();
  dictateText = ""; dictatePartial = "";
  return text;
}

function sendUtterance(text) {
  if (!text) return false;
  if (assistantSpeaking()) {
    // The assistant is still talking (or its audio is still buffered): speaking over
    // it means "stop", not "wait".
    interruptChat(t("chat.vadBargeIn"));
  }
  $("#chat-input").value = text;
  chatSend();
  return true;
}

function endUtterance() {
  if (dictateSocket && dictateSocket.readyState === 1) {
    try { dictateSocket.send(JSON.stringify({ type: "input_audio_buffer.commit" })); } catch (e) { /* ignore */ }
  }
  // Let the server emit the final transcript, then send it to the chat.
  clearTimeout(dictateTimer);
  dictateTimer = setTimeout(() => {
    if (dictateVadActive) sendUtterance(takeUtteranceText());
  }, 700);
}

// Interrupt the current assistant turn (Stop button, or barge-in).
function interruptChat(message) {
  resetChatAudio();
  flushDictatePending();  // the words spoken over the assistant belong to the next turn
  if (chatSocket && chatSocket.readyState === 1) {
    try { chatSocket.send(JSON.stringify({ type: "input.cancel" })); } catch (e) { /* ignore */ }
  }
  if (currentAssistant) currentAssistant.classList.add("cancelled");
  if (currentAudio) { currentAudio.text = currentAssistant ? currentAssistant.textContent : ""; currentAudio = null; }
  currentAssistant = null;
  chatBusy = false;
  if (message && $("#chat-status")) $("#chat-status").textContent = message;
}

function startVadLoop() {
  const analyser = dictateAnalyser;
  if (!analyser) return;
  const buf = new Float32Array(analyser.fftSize);
  const rmsOf = () => {
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    return Math.sqrt(sum / buf.length);
  };
  // Envelope state: the ambient floor we measure, the voiced time accumulated and
  // the dip we are currently tolerating.
  let floor = 0.01, voiceMs = 0, silenceMs = 0, ducked = false;
  const tick = () => {
    if (!dictateActive || !dictateVadActive) return;
    const now = performance.now();
    const rms = rmsOf();
    // Follow the quiet parts quickly, the loud ones slowly: that is the room's noise
    // floor (fan, mic self-noise, echo-cancellation residual…), whatever the gain.
    if (rms < floor) floor = floor * 0.7 + rms * 0.3;
    else floor = Math.min(0.06, floor * 0.995 + rms * 0.005);
    const speaking = rms >= Math.max(VAD_MIN_RMS, floor * VAD_SNR);
    if (speaking) { voiceMs += 60; silenceMs = 0; } else {
      silenceMs += 60;
      if (silenceMs > VAD_BARGE_HOLD_MS) voiceMs = 0;
    }

    if (assistantSpeaking()) {
      // Barge-in. The count accumulates across syllables (a single sub-threshold
      // tick used to reset it, which is why it only worked when shouting) and the
      // assistant is ducked as soon as the user starts, for immediate feedback.
      if (voiceMs >= VAD_DUCK_FROM_MS && chatPlayer && !ducked) {
        ducked = true;
        chatPlayer.setVolume(VAD_DUCK_VOLUME);
      }
      if (voiceMs >= VAD_BARGE_MS) {
        voiceMs = 0; silenceMs = 0;
        vadSpeaking = false;
        interruptChat(t("chat.vadBargeIn"));
      }
    } else {
      if (ducked) { ducked = false; if (chatPlayer) chatPlayer.setVolume(1); }
      // The assistant ended on its own: what was held is mostly our own tail (and
      // room noise), so drop it. Only a real barge-in hands it over (see interruptChat).
      if (dictatePending.length) dictatePending = [];
      if (speaking) {
        if (!vadSpeaking) {
          vadSpeaking = true;
          if ($("#chat-status")) $("#chat-status").textContent = t("chat.vadListening");
        }
        vadLastVoice = now;
      } else if (vadSpeaking && silenceMs > vadSilenceMs()) {
        vadSpeaking = false;
        endUtterance();
      }
    }
    dictateVadTimer = setTimeout(tick, 60);
  };
  dictateVadTimer = setTimeout(tick, 60);
}

async function startDictation(useVad = false) {
  try {
    // Never stack sessions: a previous one must be released upstream first.
    if (dictateSocket) {
      try { dictateSocket.close(); } catch (e) { /* ignore */ }
      dictateSocket = null;
    }
    const cap = await startMicCapture((samples, sr) => {
      if (!dictateSocket || dictateSocket.readyState !== 1) return;
      const pcm = downsampleTo16k(samples, sr);
      if (dictateVadActive && assistantSpeaking()) {
        // Never feed our own TTS back into the STT, but keep what the user says in
        // the meantime: the mic is held for the ~0.3 s the barge-in takes to fire,
        // and those samples are sent right after, so "stop, do this instead" is not
        // clipped to "this instead".
        dictatePending.push(pcm);
        while (dictatePending.length > DICTATE_PENDING_MAX) dictatePending.shift();
        return;
      }
      flushDictatePending();
      dictateSocket.send(pcm.buffer);
    });
    dictateCtx = cap.ctx; dictateStream = cap.stream; dictateSource = cap.source; dictateNode = cap.node; dictateSink = cap.sink;
    dictateSocket = new WebSocket(wsURL("/ws/stt"));
    dictateSocket.binaryType = "arraybuffer";
    dictateText = ""; dictatePartial = ""; dictatePending = [];
    dictateSocket.onopen = () => {
      dictateSocket.send(JSON.stringify({
        type: "session.update",
        session: { sample_rate: 16000, language: chatSttLanguage() || undefined, automatic_punctuation: true },
      }));
    };
    dictateSocket.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      let msg; try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.type && msg.type.includes("delta")) dictatePartial += (msg.delta || "");
      else if (msg.type && msg.type.includes("completed")) { dictateText += (msg.transcript || "") + " "; dictatePartial = ""; }
      else if (msg.type === "error") $("#chat-status").textContent = msg.message || t("stt.serviceNotReady");
    };
    dictateSocket.onerror = () => { $("#chat-status").textContent = t("stt.serviceNotReady"); };
    dictateSocket.onclose = () => { if (dictateActive) finishDictation(); };
    dictateActive = true;
    $("#chat-mic").textContent = "⏹";

    dictateVadActive = useVad && vadAvailable();
    updateVadButton();
    if (dictateVadActive) {
      dictateAnalyser = dictateCtx.createAnalyser();
      dictateAnalyser.fftSize = 1024;
      dictateSource.connect(dictateAnalyser);
      vadSpeaking = false;
      startVadLoop();
      if ($("#chat-status")) $("#chat-status").textContent = t("chat.vadListening");
    }
  } catch (e) {
    $("#chat-status").textContent = `${t("status.error")} ${e}`;
  }
}

function finishDictation() {
  clearTimeout(dictateTimer);
  clearTimeout(dictateVadTimer);
  dictateVadTimer = null;
  if (dictateAnalyser && dictateSource) { try { dictateSource.disconnect(dictateAnalyser); } catch (e) { /* ignore */ } }
  if (dictateSource) dictateSource.disconnect();
  if (dictateNode) dictateNode.disconnect();
  if (dictateSink) dictateSink.disconnect();
  if (dictateCtx) dictateCtx.close();
  if (dictateStream) dictateStream.getTracks().forEach((track) => track.stop());
  // Always close the STT socket: leaving it open keeps the upstream realtime
  // session busy, and the STT service stops accepting new ones after a few.
  const sock = dictateSocket;
  if (sock) {
    try { if (sock.readyState === 1) sock.send(JSON.stringify({ type: "input_audio_buffer.commit" })); } catch (e) { /* ignore */ }
    try { sock.close(); } catch (e) { /* ignore */ }
  }
  dictateSocket = null; dictateCtx = null; dictateStream = null;
  dictateSource = null; dictateNode = null; dictateSink = null; dictateAnalyser = null;
  dictateActive = false;
  dictateVadActive = false;
  vadSpeaking = false;
  dictatePending = [];
  $("#chat-mic").textContent = "🎤";
  updateVadButton();
  const text = takeUtteranceText();
  if (text) sendUtterance(text);
  else if ($("#chat-status")) $("#chat-status").textContent = t("stt.noSpeech");
}

function stopDictation() {
  if (dictateSocket && dictateSocket.readyState === 1) {
    try { dictateSocket.send(JSON.stringify({ type: "input_audio_buffer.commit" })); } catch (e) { /* ignore */ }
  }
  clearTimeout(dictateTimer);
  dictateTimer = setTimeout(finishDictation, 900);
}

$("#chat-mic").addEventListener("click", () => { if (dictateActive) stopDictation(); else startDictation(false); });

$("#chat-vad").addEventListener("click", () => {
  if (dictateActive && dictateVadActive) stopDictation();
  else if (!dictateActive) startDictation(true);
});

// ------------------------------------------------------------- settings tab
$("#set-startup-save").addEventListener("click", async () => {
  const status = $("#set-startup-status");
  status.textContent = "…";
  const payload = {
    startup: { stt_model: settingsSttPicker ? settingsSttPicker.value : "" },
  };
  try {
    const r = await apiFetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const text = await r.text();
    if (!r.ok) {
      const detail = errorFromBody(text);
      throw new Error(detail);
    }
    const cfg = JSON.parse(text);
    CONFIG = Object.assign({}, CONFIG, { startup: cfg.startup, models: cfg.models });
    applyCloneEngineFields(cfg);
    initCloneEngineSelects();
    status.textContent = t("status.saved");
  } catch (e) {
    status.textContent = `${t("status.error")} ${e}`;
  }
});

$("#tts-variant").addEventListener("change", () => selectVariant($("#tts-variant").value));

async function saveSettings() {
  const llm = {
    base_url: $("#set-llm-url").value.trim(),
    model: $("#set-llm-model").value.trim(),
    system_prompt: $("#set-llm-system").value,
  };
  const key = $("#set-llm-key").value;
  if (key) llm.api_key = key;
  try {
    const r = await apiFetch("/api/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ llm }),
    });
    const text = await r.text();
    if (!r.ok) {
      const detail = errorFromBody(text);
      return { ok: false, detail };
    }
    let cfg = null; try { cfg = JSON.parse(text); } catch (e) { /* ignore */ }
    if (cfg && cfg.llm) {
      CONFIG = Object.assign({}, CONFIG, { llm: cfg.llm });
      $("#set-llm-url").value = cfg.llm.base_url || "";
      $("#set-llm-model").value = cfg.llm.model || "";
      $("#set-llm-system").value = cfg.llm.system_prompt || "";
    }
    $("#set-llm-key").value = "";
    return { ok: true };
  } catch (e) {
    return { ok: false, detail: String(e) };
  }
}

// The "Default TTS engine" section saves both halves in one request. The clone
// engine only joins the payload when it actually changed: sending `models` makes
// the gateway drop every loaded model (the next request pays a full reload).
$("#tts-save").addEventListener("click", async () => {
  const status = $("#tts-status");
  status.textContent = "…";
  const payload = {
    generation: {},
    chunking: {},
    audio: {},
  };
  const num = (id, parse) => {
    const v = parse($(`#${id}`).value);
    return Number.isFinite(v) ? v : undefined;
  };
  const setIf = (obj, key, v) => { if (v !== undefined) obj[key] = v; };
  setIf(payload.generation, "temperature", num("gen-temp", parseFloat));
  setIf(payload.generation, "top_p", num("gen-topp", parseFloat));
  setIf(payload.generation, "repetition_penalty", num("gen-reppen", parseFloat));
  setIf(payload.generation, "exaggeration", num("gen-exag", parseFloat));
  setIf(payload.generation, "cfg_weight", num("gen-cfg", parseFloat));
  setIf(payload.generation, "min_p", num("gen-minp", parseFloat));
  setIf(payload.generation, "batch_size", num("gen-batch", (v) => parseInt(v, 10)));
  const maxChars = num("gen-maxchars", (v) => parseInt(v, 10));
  // One chunk must fit in a single generation: the server clamps this too (40..600).
  setIf(payload.chunking, "max_chars",
    maxChars === undefined ? undefined : Math.min(600, Math.max(40, maxChars)));
  if ($("#gen-format").value) payload.audio.format = $("#gen-format").value;

  const backend = $("#set-clone-backend").value;
  const watermark = $("#set-cb-watermark").checked;
  const current = (CONFIG && CONFIG.models) || {};
  if (backend !== (current.clone_backend || "") ||
      watermark !== !!((current.chatterbox || {}).watermark)) {
    const clone = cloneEnginePayload(backend);
    clone.models.chatterbox = { watermark };
    Object.assign(payload, clone);
  }
  try {
    const r = await apiFetch("/api/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const text = await r.text();
    if (!r.ok) throw new Error(errorFromBody(text));
    const cfg = JSON.parse(text);
    CONFIG = Object.assign({}, CONFIG, {
      generation: cfg.generation, chunking: cfg.chunking, text: cfg.text || {},
      audio: cfg.audio, models: cfg.models || CONFIG.models,
    });
    applyGenerationFields(cfg);
    applyCloneEngineFields(cfg);
    initCloneEngineSelects();
    status.textContent = t("status.saved");
  } catch (e) {
    status.textContent = `${t("status.error")} ${e}`;
  }
});

$("#set-clone-backend").addEventListener("change", toggleCloneEngineFields);

// Text cleanup: its own section (it is a server-wide policy, and an API key can
// override it per client).
$("#text-save").addEventListener("click", async () => {
  const status = $("#text-status");
  status.textContent = "…";
  const payload = {
    text: {
      cleanup: $("#gen-cleanup").value,
      emoji: $("#gen-emoji").value,
      code_blocks: $("#gen-codeblocks").value,
    },
  };
  try {
    const r = await apiFetch("/api/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const raw = await r.text();
    if (!r.ok) throw new Error(errorFromBody(raw));
    const cfg = JSON.parse(raw);
    CONFIG = Object.assign({}, CONFIG, { text: cfg.text || {} });
    applyGenerationFields(cfg);
    status.textContent = t("status.saved");
  } catch (e) {
    status.textContent = `${t("status.error")} ${e}`;
  }
});

$("#set-log-save").addEventListener("click", async () => {
  const status = $("#set-log-status");
  status.textContent = "…";
  try {
    const r = await apiFetch("/api/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ server: { log_level: $("#set-log-level").value } }),
    });
    const text = await r.text();
    if (!r.ok) {
      const detail = errorFromBody(text);
      throw new Error(detail);
    }
    const cfg = JSON.parse(text);
    CONFIG = Object.assign({}, CONFIG, { server: cfg.server });
    status.textContent = t("status.saved");
  } catch (e) {
    status.textContent = `${t("status.error")} ${e}`;
  }
});

// ------------------------------------------------------------ vLLM-Omni card

function vllmEngine() {
  const el = $("#set-vllm-enabled");
  return !!(el && el.checked);
}

function setVllmEngine(enabled) {
  if ($("#set-vllm-enabled")) $("#set-vllm-enabled").checked = !!enabled;
}

function vllmStageMemory() {
  const out = {};
  for (const stage of ["0", "1"]) {
    const value = Number(($(`#set-vllm-stage${stage}`) || {}).value || 0);
    if (value > 0) out[stage] = value;
  }
  return out;
}

function vllmFamilyForRepo(repo) {
  return /-Base$/i.test(repo || "") ? "base" : "custom_voice";
}

function fillVllmModels() {
  const select = $("#set-vllm-model");
  if (!select) return;
  const variants = (CONFIG.models && CONFIG.models.variants) || [];
  const current = select.value;
  select.innerHTML = "";
  for (const variant of variants) {
    if (variant.family !== "custom_voice" && variant.family !== "base") continue;
    const option = document.createElement("option");
    option.value = variant.repo;
    option.textContent = `${variant.label || ""} ${variant.repo.split("/").pop()}`.trim();
    select.appendChild(option);
  }
  // Fall back to the checkpoint the settings ask for, then the local registry.
  const wanted = current
    || (CONFIG.vllm && CONFIG.vllm.model)
    || ((CONFIG.models && CONFIG.models.registry) || {})[
      (CONFIG.vllm && CONFIG.vllm.family) === "base" ? "base" : "custom_voice"
    ];
  if (wanted) select.value = wanted;
}

async function refreshVllmBudget() {
  const box = $("#set-vllm-budget");
  const select = $("#set-vllm-model");
  if (!box || !select || !select.value) return;
  try {
    const r = await getJSON(`/api/vllm/budget?model=${encodeURIComponent(select.value)}`);
    if (r.error && !Object.keys(r.stages || {}).length) { box.textContent = r.error; return; }
    const stages = r.stages || {};
    const mine = Object.keys(vllmStageMemory() || {}).length > 0;
    const budget = Object.entries(stages).map(([k, v]) => `${k}=${v}`).join(" + ");
    const parts = [];
    if (r.total_gib) parts.push(`${t("vllm.card")} ${r.total_gib} GiB (${r.free_gib} GiB ${t("vllm.free")})`);
    if (r.weights_gib) parts.push(`${t("vllm.weights")} ${r.weights_gib.talker} + ${r.weights_gib.codec} GiB`);
    parts.push(`${t(mine ? "vllm.manual" : "vllm.suggested")} ${budget}`);
    box.textContent = parts.join(" · ");
  } catch (e) {
    box.textContent = "";
  }
}

async function refreshVllmService() {
  const status = $("#set-vllm-status");
  const control = !!(CONFIG.docker && CONFIG.docker.control);
  for (const id of ["#set-vllm-start", "#set-vllm-stop", "#set-vllm-restart"]) {
    if ($(id)) $(id).disabled = !control;
  }
  if (!status || !control) return;
  try {
    const r = await getJSON("/api/vllm/service");
    if (r.state) {
      const state = r.state.running ? t("vllm.running") : t("vllm.stopped");
      const asleep = r.sleeping ? ` (${t("vllm.asleep")})` : "";
      const actual = (r.profile && r.profile.actual) || {};
      const bits = [];
      const budget = Object.entries(actual.stage_memory || {})
        .map(([stage, value]) => `${stage}=${value}`).join(" + ");
      if (budget) bits.push(budget);
      if (actual.max_model_len) bits.push(`ctx ${actual.max_model_len}`);
      if (actual.enforce_eager) bits.push("eager");
      const suffix = bits.length ? ` · ${bits.join(" · ")}` : "";
      const diverges = r.profile && !r.profile.matches
        ? ` ⚠ ${t("vllm.diverges")}` : "";
      status.textContent =
        `${r.container}: ${state}${asleep}${actual.model ? ` — ${actual.model}` : ""}${suffix}${diverges}`;
      if (r.state.model && $("#set-vllm-model")) {
        $("#set-vllm-model").value = r.state.model;
        if ($("#set-vllm-family")) $("#set-vllm-family").value = vllmFamilyForRepo(r.state.model);
      }
    } else if (r.error) {
      status.textContent = r.error;
    }
  } catch (e) {
    status.textContent = `${t("status.error")} ${e}`;
  }
}

$("#set-vllm-save").addEventListener("click", async () => {
  const status = $("#set-vllm-status");
  status.textContent = "…";
  const vllm = {
    enabled: vllmEngine(),
    family: $("#set-vllm-family") ? $("#set-vllm-family").value : "custom_voice",
    base_url: $("#set-vllm-url").value.trim() || "http://tts-vllm:8091",
    path: $("#set-vllm-path").value.trim() || "/v1/audio/speech",
    sleep_after: Number($("#set-vllm-sleep") ? $("#set-vllm-sleep").value : 0) || 0,
    sleep_level: Number($("#set-vllm-sleep-level") ? $("#set-vllm-sleep-level").value : 1) || 1,
    max_model_len: Number($("#set-vllm-maxlen") ? $("#set-vllm-maxlen").value : 2048) || 2048,
    enforce_eager: $("#set-vllm-eager") ? $("#set-vllm-eager").checked : true,
    // 0/empty = let the gateway compute the stage's share from the card.
    stage_memory: {
      "0": Number($("#set-vllm-stage0") ? $("#set-vllm-stage0").value : 0) || 0,
      "1": Number($("#set-vllm-stage1") ? $("#set-vllm-stage1").value : 0) || 0,
    },
    model: $("#set-vllm-model") ? $("#set-vllm-model").value : "",
  };
  try {
    const r = await apiFetch("/api/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ vllm }),
    });
    const text = await r.text();
    if (!r.ok) {
      const detail = errorFromBody(text);
      throw new Error(detail);
    }
    const cfg = JSON.parse(text);
    CONFIG = Object.assign({}, CONFIG, { vllm: cfg.vllm });
    applyVllmFields(cfg);
    status.textContent = t("status.saved");
    // The gateway applies the profile itself (checkpoint, flags, start/stop): the
    // container recreation takes a moment, so refresh the state afterwards.
    if (CONFIG.docker && CONFIG.docker.control) {
      status.textContent = `${t("status.saved")} — ${t("vllm.applying")}`;
      setTimeout(refreshVllmService, 5000);
      setTimeout(refreshVllmService, 20000);
    }
  } catch (e) {
    status.textContent = `${t("status.error")} ${e}`;
  }
});

for (const id of ["#set-vllm-stage0", "#set-vllm-stage1"]) {
  const input = $(id);
  if (input) input.addEventListener("change", refreshVllmBudget);
}

$("#set-vllm-model") && $("#set-vllm-model").addEventListener("change", () => {
  // The checkpoint implies the family the gateway has to route to it.
  if ($("#set-vllm-family")) $("#set-vllm-family").value = vllmFamilyForRepo($("#set-vllm-model").value);
  refreshVllmBudget();
});

for (const [id, action] of [["#set-vllm-start", "start"], ["#set-vllm-stop", "stop"], ["#set-vllm-restart", "restart"]]) {
  const button = $(id);
  if (!button) continue;
  button.addEventListener("click", async () => {
    const status = $("#set-vllm-status");
    status.textContent = t("status.testing");
    try {
      const r = await apiFetch(`/api/vllm/service/${action}`, { method: "POST" });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.detail || r.status);
      await refreshVllmService();
      if (action !== "stop") status.textContent = `${status.textContent} — ${t("vllm.loading")}`;
    } catch (e) {
      status.textContent = `${t("status.error")} ${e}`;
    }
  });
}


$("#set-vllm-test").addEventListener("click", async () => {
  const status = $("#set-vllm-status");
  status.textContent = t("status.testing");
  try {
    const r = await getJSON("/api/vllm/health");
    const models = (r.models || []).join(", ");
    const served = r.warning ? ` ⚠ ${r.warning}` : (r.served_family ? ` (${r.served_family})` : "");
    status.textContent = r.ok
      ? `${t("status.online")} — ${models || r.base_url}${served}`
      : `${t("status.offline")} ${r.error || r.status || ""}`;
  } catch (e) {
    status.textContent = `${t("status.error")} ${e}`;
  }
});

$("#set-llm-ping").addEventListener("click", async () => {
  $("#set-llm-status").textContent = t("status.testing");
  const res = await saveSettings();
  if (!res.ok) { $("#set-llm-status").textContent = `${t("status.error")} ${res.detail}`; return; }
  try {
    const r = await getJSON("/api/llm/ping");
    $("#set-llm-status").textContent = r.ok ? `OK (${r.url})` : `${t("status.error")} ${r.error || r.status}`;
  } catch (e) {
    $("#set-llm-status").textContent = `${t("status.error")} ${e}`;
  }
});

// ------------------------------------------------- admin session & API keys
function sessionState(text, cls) {
  const chip = $("#session-chip");
  const el = $("#session-label");
  if (el) { el.textContent = text || ""; el.className = cls || "muted"; }
  const enabled = !!(CONFIG && CONFIG.auth_required);
  const logged = !!apiKey();
  if ($("#session-login")) $("#session-login").hidden = !enabled || logged;
  if ($("#session-logout")) $("#session-logout").hidden = !enabled || !logged;
  if (chip) chip.hidden = !enabled;
}

function fmtDate(v) {
  return v ? new Date(v).toLocaleString() : "—";
}

async function ensureSession() {
  const r = await apiFetch("/api/session", { method: "POST" });
  const text = await r.text();
  if (!r.ok) {
    let detail = text;
    detail = errorFromBody(text);
    throw new Error(detail || `HTTP ${r.status}`);
  }
  const data = JSON.parse(text);
  if (data.key) localStorage.setItem("apiKey", data.key);
  return data;
}

async function logoutSession() {
  try { await apiFetch("/api/session/logout", { method: "POST" }); } catch (e) { /* ignore */ }
  localStorage.removeItem("apiKey");
}

let lastKeys = [];

function renderApiKeys(keys) {
  const body = $("#set-keys-body");
  if (!body) return;
  body.innerHTML = "";
  if (!keys.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4; td.className = "muted"; td.textContent = t("settings.noKeys");
    tr.appendChild(td); body.appendChild(tr);
    return;
  }
  keys.forEach((k) => {
    const tr = document.createElement("tr");
    const nameCell = document.createElement("td");
    const name = document.createElement("div");
    name.textContent = k.name || "—";
    if (k.expired) {
      const badge = document.createElement("span");
      badge.className = "muted";
      badge.textContent = ` ${t("settings.expired")}`;
      name.appendChild(badge);
    }
    nameCell.appendChild(name);
    const policy = policySummary(k.text);
    if (policy) {
      const line = document.createElement("div");
      line.className = "key-policy";
      line.textContent = policy;
      nameCell.appendChild(line);
    }
    tr.appendChild(nameCell);
    [fmtDate(k.created_at), k.expires_at ? fmtDate(k.expires_at) : t("settings.never")]
      .forEach((c) => { const td = document.createElement("td"); td.textContent = c; tr.appendChild(td); });
    const actions = document.createElement("td");
    const edit = document.createElement("button");
    edit.textContent = t("settings.edit");
    edit.addEventListener("click", () => openKeyDialog(k.id));
    actions.appendChild(edit);
    tr.appendChild(actions);
    body.appendChild(tr);
  });
}

// Per-key text policy: a field left on "" inherits the global setting.
const KEY_POLICY_FIELDS = {
  cleanup: [["off", "settings.cleanupOff"], ["light", "settings.cleanupLight"],
            ["llm", "settings.cleanupLlm"]],
  emoji: [["keep", "settings.emojiKeep"], ["strip", "settings.emojiStrip"],
          ["name", "settings.emojiName"]],
  code_blocks: [["drop", "settings.codeDrop"], ["announce", "settings.codeAnnounce"]],
};

function policyLabel(field, value) {
  const entry = KEY_POLICY_FIELDS[field].find(([v]) => v === value);
  return entry ? t(entry[1]) : value;
}

function policySummary(text) {
  if (!text) return "";
  return Object.keys(KEY_POLICY_FIELDS)
    .filter((f) => text[f])
    .map((f) => `${t("settings.textCleanup")}: ${policyLabel(f, text[f])}`)
    .join(" · ");
}

function policyFields() {
  return {
    cleanup: $("#kd-cleanup").value,
    emoji: $("#kd-emoji").value,
    code_blocks: $("#kd-codeblocks").value,
  };
}

function applyPolicyToDialog(text) {
  const policy = text || {};
  $("#kd-cleanup").value = policy.cleanup || "";
  $("#kd-emoji").value = policy.emoji || "";
  $("#kd-codeblocks").value = policy.code_blocks || "";
}

function openKeyDialog(id) {
  const dlg = $("#key-dialog");
  dlg.dataset.id = id || "";
  dlg.dataset.created = "";
  $("#kd-status").textContent = "";
  $("#kd-name").value = "";
  $("#kd-days").value = "";
  applyPolicyToDialog(null);
  $("#kd-delete").hidden = !id;
  $("#kd-title").textContent = id ? t("settings.keyEdit") : t("settings.keyNew");
  if (!id) {
    if (dlg.showModal) dlg.showModal(); else dlg.setAttribute("open", "");
    return;
  }
  const key = (lastKeys || []).find((k) => k.id === id) || {};
  $("#kd-name").value = key.name || "";
  applyPolicyToDialog(key.text);
  if (key.expires_at) {
    const days = Math.max(0, Math.round((new Date(key.expires_at) - Date.now()) / 86400000));
    $("#kd-days").value = String(days);
  }
  if (dlg.showModal) dlg.showModal(); else dlg.setAttribute("open", "");
}

async function saveKeyDialog() {
  const dlg = $("#key-dialog");
  const id = dlg.dataset.id;
  const status = $("#kd-status");
  const name = $("#kd-name").value.trim();
  if (!name) { status.textContent = t("settings.keyNameRequired"); return; }
  const days = parseInt($("#kd-days").value, 10);
  const expiry = Number.isFinite(days) && days > 0 ? { expires_in_days: days } : { expires_at: null };
  const text = policyFields();
  const hasPolicy = Object.keys(text).some((f) => text[f]);
  status.textContent = "…";
  try {
    let r;
    if (id) {
      const body = Object.assign({ name, text: hasPolicy ? text : null }, expiry);
      r = await apiFetch(`/api/api-keys/${encodeURIComponent(id)}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } else {
      const body = Object.assign({ name }, expiry);
      if (hasPolicy) body.text = text;
      r = await apiFetch("/api/api-keys", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    }
    const raw = await r.text();
    if (!r.ok) throw new Error(errorFromBody(raw));
    const data = JSON.parse(raw);
    if (data.key) showNewKey(data.key);
    dlg.close();
    await loadApiKeys();
  } catch (e) {
    status.textContent = `${t("status.error")} ${e}`;
  }
}

async function deleteKeyFromDialog() {
  const id = $("#key-dialog").dataset.id;
  if (!id) return;
  const status = $("#kd-status");
  status.textContent = "…";
  try {
    const r = await apiFetch(`/api/api-keys/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (!r.ok) throw new Error(errorFromBody(await r.text()));
    $("#key-dialog").close();
    await loadApiKeys();
    $("#set-keys-status").textContent = t("status.saved");
  } catch (e) {
    status.textContent = `${t("status.error")} ${e}`;
  }
}

function showNewKey(secret) {
  const box = $("#set-key-new");
  box.textContent = "";
  const label = document.createElement("span");
  label.textContent = `${t("settings.newKey")} `;
  const code = document.createElement("code");
  code.textContent = secret;
  const copy = document.createElement("button");
  copy.textContent = t("settings.copy");
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(secret);
      copy.textContent = t("settings.copied");
    } catch (e) {
      copy.textContent = t("status.error");
    }
  });
  box.appendChild(label); box.appendChild(code); box.appendChild(copy);
}

async function loadApiKeys() {
  const wrap = $("#set-keys-wrap");
  applyAccessState();
  if (!wrap) return;
  if (!(CONFIG && CONFIG.auth_required)) {
    wrap.hidden = true;
    if ($("#set-keys-disabled")) $("#set-keys-disabled").hidden = false;
    return;
  }
  try {
    const data = await getJSON("/api/api-keys");
    if (!data.enabled) { wrap.hidden = true; return; }
    if ($("#set-keys-disabled")) $("#set-keys-disabled").hidden = true;
    wrap.hidden = false;
    lastKeys = data.keys || [];
    renderApiKeys(lastKeys);
    sessionState(t("settings.sessionActive"));
  } catch (e) {
    wrap.hidden = true;
    sessionState(t("settings.notAdmin"));
  }
}

function applyAccessState() {
  const enabled = !!(CONFIG && CONFIG.auth_required);
  sessionState(enabled ? "" : t("settings.accessDisabled"));
  if (!enabled) {
    if ($("#set-keys-wrap")) $("#set-keys-wrap").hidden = true;
    if ($("#set-keys-disabled")) $("#set-keys-disabled").hidden = false;
  }
}

// The API-key dialog (create and edit share it; delete lives in it too).
$("#set-key-create").addEventListener("click", () => openKeyDialog(""));
$("#kd-close").addEventListener("click", () => $("#key-dialog").close());
$("#kd-save").addEventListener("click", saveKeyDialog);
$("#kd-delete").addEventListener("click", deleteKeyFromDialog);

$("#session-login").addEventListener("click", async () => {
  try {
    await ensureSession();
    await loadApiKeys();
  } catch (e) {
    sessionState(`${t("settings.loginFailed")}: ${e}`);
  }
});

$("#session-logout").addEventListener("click", async () => {
  await logoutSession();
  await loadApiKeys();
});

// Settings sections are `<details>` accordions: remember which ones are open.
function initSettingsSections() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem("settingsSections") || "{}"); } catch (e) { saved = {}; }
  const sections = document.querySelectorAll("#tab-settings details.card");
  sections.forEach((sec, idx) => {
    if (sec.id in saved) sec.open = !!saved[sec.id];
    else sec.open = idx < 2;      // the two most used ones start open
    sec.addEventListener("toggle", () => {
      saved[sec.id] = sec.open;
      try { localStorage.setItem("settingsSections", JSON.stringify(saved)); } catch (e) { /* ignore */ }
    });
  });
}
initSettingsSections();

init();
