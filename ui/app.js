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

// ----------------------------------------------------------------- player PCM
class PCMPlayer {
  constructor(sampleRate) {
    this.sr = sampleRate || 24000;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.time = 0;
    this.leftover = new Uint8Array(0);
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
      this.pushInt16(new Int16Array(ab));
    }
  }
  pushInt16(int16) {
    const f = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) f[i] = int16[i] / 32768;
    const buf = this.ctx.createBuffer(1, f.length, this.sr);
    buf.copyToChannel(f, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf; src.connect(this.ctx.destination);
    const clock = Math.max(this.ctx.currentTime, this.time);
    src.start(clock);
    this.time = clock + buf.duration;
  }
  reset() { this.leftover = new Uint8Array(0); this.time = this.ctx.currentTime; }
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
let speakers = [];
let micDeviceId = "";

// ---------------------------------------------------------------- tabs + i18n
window.I18N.init();
window.I18N.apply();

function switchTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((el) => el.classList.remove("active"));
  const section = $(`#tab-${name}`);
  if (section) section.classList.add("active");
}

document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

$("#lang-select").addEventListener("change", (e) => {
  window.I18N.setLang(e.target.value);
  if (CONFIG) {
    buildTtsFamilySelect();
    buildStartupVoiceSelect();
    renderSttPickers();
    renderAvatars();
    renderVoiceList();
    loadMicDevices();
  }
});

// ------------------------------------------------------------------ init app
async function init() {
  try {
    CONFIG = await getJSON("/api/config");
  } catch (e) {
    $("#health").className = "dot err";
    $("#active-model").textContent = t("status.configUnavailable");
    return;
  }
  $("#title").textContent = CONFIG.ui.title || t("app.title");
  TTS_SESSION.sample_rate = CONFIG.audio.sample_rate;
  $("#health").className = "dot ok";

  buildTtsFamilySelect();
  $("#tts-family").addEventListener("change", () => { loadVoices(); updateTtsModelDesc(); });

  const lang = $("#tts-language");
  lang.innerHTML = "";
  (CONFIG.languages || ["Auto"]).forEach((l) => {
    const o = document.createElement("option"); o.value = l; o.textContent = l; lang.appendChild(o);
  });

  const avLang = $("#av-lang");
  avLang.innerHTML = "";
  (CONFIG.languages || ["Auto"]).forEach((l) => {
    const o = document.createElement("option"); o.value = l; o.textContent = l; avLang.appendChild(o);
  });
  avLang.value = "English";

  applySettingsFields(CONFIG);

  mountSttPickers();
  await loadVoices();
  await loadSttModels();
  await loadAvatars();
  await loadConversationVoices();
  const preload = selectedVoice();
  if (preload) loadVoiceModel(preload);
  buildStartupVoiceSelect();
  await loadMicDevices();
  refreshConfigModels();
  refreshStatus();
}

function applySettingsFields(cfg) {
  $("#set-llm-url").value = cfg.llm.base_url || "";
  $("#set-llm-model").value = cfg.llm.model || "";
  $("#set-llm-system").value = cfg.llm.system_prompt || "";
  $("#set-temp").value = cfg.generation.temperature;
  $("#set-batch").value = cfg.generation.batch_size;
  $("#set-maxchars").value = cfg.chunking.max_chars;
  $("#set-apikey").value = apiKey();
}

// -------------------------------------------------------------- TTS model UI
function tt(key, fallback) {
  const v = window.I18N.t(key);
  return v === key ? fallback : v;
}
function ttsName(family) { return tt(`tts.model.${family}.name`, family); }
function ttsDesc(family) { return tt(`tts.model.${family}.desc`, ""); }

function buildTtsFamilySelect() {
  const fam = $("#tts-family");
  const current = fam.value || CONFIG.models.default_family;
  fam.innerHTML = "";
  (CONFIG.models.families || []).forEach((m) => {
    const o = document.createElement("option");
    o.value = m.family;
    const tag = m.loaded ? t("models.loaded") : t("models.notLoaded");
    o.textContent = `${ttsName(m.family)} [${tag}]`;
    fam.appendChild(o);
  });
  fam.value = current;
  updateTtsModelDesc();
}

function updateTtsModelDesc() {
  $("#tts-model-desc").textContent = ttsDesc($("#tts-family").value);
}

async function refreshConfigModels() {
  try {
    const data = await getJSON("/api/models");
    CONFIG.models.families = data.families;
    buildTtsFamilySelect();
    const loaded = (data.loaded || []).map(ttsName).join(", ");
    $("#tts-model-status").textContent = loaded ? `${t("models.loaded")}: ${loaded}` : t("models.notLoaded");
  } catch (e) { /* ignore */ }
}

$("#tts-load").addEventListener("click", async () => {
  const family = $("#tts-family").value;
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
      try { detail = JSON.parse(text).detail || text; } catch (e) { /* keep */ }
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
  const family = $("#tts-family").value;
  try {
    const data = await getJSON(`/api/voices?family=${encodeURIComponent(family)}`);
    speakers = data.speakers || [];
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

async function refreshStatus() {
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
function ttsPayload(text) {
  const family = $("#tts-family").value;
  const speaker = family === "custom_voice" ? $("#tts-speaker").value : undefined;
  return {
    text,
    family,
    language: $("#tts-language").value,
    speaker,
    instruct: $("#tts-instruct").value || undefined,
    ref_text: $("#tts-ref-text").value || undefined,
    x_vector_only_mode: $("#tts-xvec").checked,
    chunking: $("#tts-chunk").checked,
    format: "pcm",
    stream: $("#tts-stream").checked,
  };
}

async function uploadReference() {
  const file = $("#tts-ref-audio").files[0];
  if (!file) return null;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await apiFetch("/v1/audio/transcriptions", { method: "POST", body: fd });
    const text = await r.text();
    if (r.ok) { const j = JSON.parse(text); if (j.text) $("#tts-ref-text").value = j.text; }
  } catch (e) { /* STT may be disabled */ }
  return await fileToDataURL(file);
}

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
  if ($("#tts-family").value === "base") {
    const ref = await uploadReference();
    if (ref) payload.ref_audio = ref;
  }
  if (payload.stream) {
    ttsPlayer = ttsPlayer || new PCMPlayer(TTS_SESSION.sample_rate);
    ttsPlayer.reset();
    try {
      const r = await apiFetch("/api/tts", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error((await r.text()).slice(0, 200));
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
      $("#tts-player").src = URL.createObjectURL(await r.blob());
      $("#tts-status").textContent = t("status.done");
    } catch (e) {
      $("#tts-status").textContent = t("status.error");
      $("#tts-log").textContent = String(e);
    }
  }
  refreshStatus();
});

// ------------------------------------------------------------------ avatars
async function loadAvatars() {
  try {
    const data = await getJSON("/api/avatars");
    avatars = data.avatars || [];
  } catch (e) {
    avatars = [];
  }
  renderAvatars();
}

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
    meta.textContent = a.kind === "clone" ? t("avatars.kindClone") : t("avatars.kindDesign");
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
    if (a.kind === "design") {
      actions.appendChild(mk(t("avatars.editPrompt"), "", async () => {
        const prompt = window.prompt(t("avatars.prompt"), a.prompt || "");
        if (prompt === null) return;
        const fd = new FormData(); fd.append("prompt", prompt),
        fd.append("language", a.language || "English");
        await apiFetch(`/api/avatars/${encodeURIComponent(a.id)}`, { method: "PUT", body: fd });
        await regenerateAvatar(a.id);
      }));
      actions.appendChild(mk(t("avatars.regenerate"), "", () => regenerateAvatar(a.id)));
    }
    actions.appendChild(mk(t("avatars.rename"), "", async () => {
      const name = window.prompt(t("avatars.name"), a.name);
      if (!name) return;
      const fd = new FormData(); fd.append("name", name);
      await apiFetch(`/api/avatars/${encodeURIComponent(a.id)}`, { method: "PUT", body: fd });
      await loadAvatars();
    }));
    actions.appendChild(mk(t("avatars.personality"), "", async () => {
      const personality = window.prompt(t("avatars.personality"), a.personality || "");
      if (personality === null) return;
      const fd = new FormData(); fd.append("personality", personality);
      await apiFetch(`/api/avatars/${encodeURIComponent(a.id)}`, { method: "PUT", body: fd });
      await loadAvatars(); await loadConversationVoices();
    }));
    actions.appendChild(mk(t("avatars.delete"), "", async () => {
      if (!window.confirm(`${t("avatars.delete")} "${a.name}"?`)) return;
      await apiFetch(`/api/avatars/${encodeURIComponent(a.id)}`, { method: "DELETE" });
      await loadAvatars(); await loadConversationVoices(); buildStartupVoiceSelect();
    }));
    card.appendChild(actions);
    list.appendChild(card);
  });
}

async function regenerateAvatar(id) {
  $("#av-status").textContent = "…";
  try {
    const r = await apiFetch(`/api/avatars/${encodeURIComponent(id)}/regenerate`, { method: "POST" });
    if (!r.ok) throw new Error((await r.text()).slice(0, 200));
    $("#av-status").textContent = t("status.done");
    await loadAvatars();
  } catch (e) {
    $("#av-status").textContent = `${t("status.error")} ${e}`;
  }
}

$("#av-kind").addEventListener("change", () => {
  const clone = $("#av-kind").value === "clone";
  $("#av-design-box").hidden = clone;
  $("#av-clone-box").hidden = !clone;
});

$("#av-create").addEventListener("click", async () => {
  const name = $("#av-name").value.trim();
  if (!name) { $("#av-status").textContent = t("avatars.name"); return; }
  const fd = new FormData();
  fd.append("name", name);
  fd.append("kind", $("#av-kind").value);
  fd.append("language", $("#av-lang").value);
  fd.append("prompt", $("#av-prompt").value);
  fd.append("ref_text", $("#av-reftext").value);
  fd.append("personality", $("#av-personality").value);
  if ($("#av-ref").files[0]) fd.append("ref_audio", $("#av-ref").files[0]);
  if ($("#av-photo").files[0]) fd.append("image", $("#av-photo").files[0]);

  const btn = $("#av-create");
  btn.disabled = true;
  $("#av-status").textContent = t("models.loading");
  try {
    const r = await apiFetch("/api/avatars", { method: "POST", body: fd });
    const text = await r.text();
    if (!r.ok) {
      let detail = text;
      try { detail = JSON.parse(text).detail || text; } catch (e) { /* keep */ }
      throw new Error(detail);
    }
    $("#av-status").textContent = t("status.saved");
    $("#av-name").value = ""; $("#av-prompt").value = ""; $("#av-reftext").value = "";
    $("#av-personality").value = "";
    $("#av-ref").value = ""; $("#av-photo").value = "";
    await loadAvatars();
    await loadConversationVoices();
    buildStartupVoiceSelect();
  } catch (e) {
    $("#av-status").textContent = `${t("status.error")} ${e}`;
  } finally {
    btn.disabled = false;
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

function voiceConfig() {
  const v = selectedVoice();
  if (!v) return { family: "custom_voice" };
  if (v.kind === "avatar") return { avatar: v.avatarId };
  return { family: "custom_voice", speaker: v.speaker };
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
    this.opts = Object.assign({ load: true, stop: true, refresh: true, desc: true, status: true, initial: null }, opts);
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
    this.select.addEventListener("change", () => { this.value = this.select.value; this.touched = true; this.update(); });
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
    this.labelSpan.textContent = t("stt.model");
    this.select.innerHTML = "";
    if (!sttModels.length) {
      const o = document.createElement("option"); o.value = ""; o.textContent = t("stt.indexUnavailable"); this.select.appendChild(o);
    }
    sttModels.forEach((m) => {
      const o = document.createElement("option");
      o.value = m.repo;
      o.textContent = `${m.filename || m.repo} · ${sttStateLabel(sttModelState(m))} · ${fmtMB(m.size)} ${t("stt.sizeMb")}`;
      this.select.appendChild(o);
    });
    if (prev && sttModels.some((m) => m.repo === prev)) this.select.value = prev;
    else if (sttModels.length) this.select.value = sttActive || sttModels[0].repo;
    this.value = this.select.value;
    this.update();
  }

  current() { return sttModels.find((x) => x.repo === this.select.value) || null; }

  update() {
    const m = this.current();
    const downloading = !!sttDownload && sttDownload.repo === this.select.value;
    if (this.desc) this.desc.textContent = m ? sttModelDesc(m) : "";
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

function mountSttPickers() {
  const sttRoot = $("#stt-picker");
  if (sttRoot && !sttRoot.dataset.mounted) { sttRoot.dataset.mounted = "1"; new SttPicker(sttRoot); }
  const chatRoot = $("#chat-stt-picker");
  if (chatRoot && !chatRoot.dataset.mounted) {
    chatRoot.dataset.mounted = "1";
    new SttPicker(chatRoot, { load: true, stop: false, refresh: false, desc: false, status: false });
  }
  const setRoot = $("#set-startup-stt-picker");
  if (setRoot && !setRoot.dataset.mounted) {
    setRoot.dataset.mounted = "1";
    settingsSttPicker = new SttPicker(setRoot, {
      load: false, stop: false, refresh: false, desc: true, status: true,
      initial: () => (CONFIG ? CONFIG.startup.stt_model : ""),
    });
  }
}

function renderSttPickers() { sttPickers.forEach((p) => p.populate()); }

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
  if (!micDeviceId) return navigator.mediaDevices.getUserMedia({ audio: true });
  try {
    return await navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: micDeviceId } } });
  } catch (e) {
    if (e && ["OverconstrainedError", "NotFoundError", "DevicesNotFoundError"].includes(e.name)) {
      micDeviceId = "";
      const sel = $("#stt-mic-device"); if (sel) sel.value = "";
      return navigator.mediaDevices.getUserMedia({ audio: true });
    }
    throw e;
  }
}

// STT microphone
let micCtx = null, micStream = null, micNode = null, sttSocket = null, sttFinal = "", sttPartial = "";

$("#stt-record").addEventListener("click", async () => {
  const btn = $("#stt-record");
  if (sttSocket) { stopMic(); return; }
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("#stt-status").textContent = t("stt.micInsecure");
    return;
  }
  try {
    micStream = await openMicStream();
    loadMicDevices();
    micCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = micCtx.createMediaStreamSource(micStream);
    micNode = micCtx.createScriptProcessor(4096, 1, 1);
    sttSocket = new WebSocket(wsURL("/ws/stt"));
    sttSocket.binaryType = "arraybuffer";
    sttFinal = $("#stt-text").textContent;
    sttPartial = "";

    sttSocket.onopen = () => {
      sttSocket.send(JSON.stringify({
        type: "session.update",
        session: { sample_rate: 16000, language: $("#stt-lang").value || undefined, automatic_punctuation: true },
      }));
      $("#stt-status").textContent = t("stt.recording");
      btn.textContent = t("stt.stop");
      btn.classList.remove("primary");
    };
    sttSocket.onmessage = (ev) => {
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
    sttSocket.onerror = () => { $("#stt-status").textContent = t("stt.serviceNotReady"); };
    sttSocket.onclose = (ev) => {
      stopMic();
      if (ev && ev.code !== 1000 && ev.code !== 4401) $("#stt-status").textContent = t("stt.serviceNotReady");
    };
    micNode.onaudioprocess = (e) => {
      if (!sttSocket || sttSocket.readyState !== 1) return;
      sttSocket.send(downsampleTo16k(e.inputBuffer.getChannelData(0), micCtx.sampleRate).buffer);
    };
    source.connect(micNode);
    micNode.connect(micCtx.destination);
  } catch (e) {
    $("#stt-status").textContent = `${t("status.micUnavailable")}: ${e}`;
  }
});

function stopMic() {
  if (micNode) micNode.disconnect();
  if (micCtx) micCtx.close();
  if (micStream) micStream.getTracks().forEach((track) => track.stop());
  if (sttSocket && sttSocket.readyState === 1) {
    try { sttSocket.send(JSON.stringify({ type: "input_audio_buffer.commit" })); } catch (e) { /* ignore */ }
    sttSocket.close();
  }
  sttSocket = null; micCtx = null; micStream = null; micNode = null;
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

function appendTurn(role, text, name) {
  const row = document.createElement("div");
  row.className = `bubble-row ${role}`;
  if (role === "assistant") {
    const icon = document.createElement("span");
    icon.className = "bubble-icon";
    const v = selectedVoice();
    if (v && v.has_photo) icon.style.backgroundImage = `url('/api/avatars/${encodeURIComponent(v.avatarId)}/photo')`;
    else { icon.style.background = colorFor(name || (v ? v.name : "AI")); icon.textContent = initialOf(name || (v ? v.name : "AI")); }
    row.appendChild(icon);
  }
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  row.appendChild(bubble);
  $("#chat-log").appendChild(row);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
  return bubble;
}

function chatConfig() {
  return {
    type: "session.config",
    speak: $("#chat-speak").checked,
    personality: chatPersonality(),
    tts: Object.assign({
      language: $("#tts-language").value,
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
  return new Promise((resolve, reject) => {
    const sock = new WebSocket(wsURL("/ws/chat"));
    sock.binaryType = "arraybuffer";
    sock.onopen = () => { chatSocket = sock; sock.onmessage = onChatMessage; sock.send(JSON.stringify(chatConfig())); resolve(sock); };
    sock.onerror = () => reject(new Error("connection failed"));
    sock.onclose = () => { chatSocket = null; };
  });
}

function resetChatAudio() {
  if (chatPlayer) { chatPlayer.stop(); chatPlayer = null; }
}

function onChatMessage(ev) {
  if (typeof ev.data !== "string") {
    if (chatPlayer && $("#chat-speak").checked) chatPlayer.pushBytes(new Uint8Array(ev.data));
    return;
  }
  let msg; try { msg = JSON.parse(ev.data); } catch (e) { return; }
  if (msg.type === "llm.delta") {
    if (currentAssistant) currentAssistant.textContent += msg.text;
  } else if (msg.type === "audio.start") {
    if (!chatPlayer) chatPlayer = new PCMPlayer(TTS_SESSION.sample_rate);
  } else if (msg.type === "llm.done") {
    if (currentAssistant && !currentAssistant.textContent) currentAssistant.textContent = msg.text || "";
    currentAssistant = null;
    chatBusy = false;
  } else if (msg.type === "turn.cancelled") {
    if (currentAssistant) currentAssistant.classList.add("cancelled");
    currentAssistant = null;
    chatBusy = false;
    resetChatAudio();
  } else if (msg.type === "error") {
    if (currentAssistant) currentAssistant.textContent += ` [error: ${msg.message}]`;
    currentAssistant = null;
    chatBusy = false;
    $("#chat-status").textContent = msg.message;
  }
}

async function chatSend() {
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  appendTurn("user", text);
  currentAssistant = appendTurn("assistant", "");
  $("#chat-status").textContent = "";
  chatBusy = true;
  try {
    const sock = await ensureChatSocket();
    sock.send(JSON.stringify(chatConfig()));
    sock.send(JSON.stringify({ type: "input.text", text }));
  } catch (e) {
    if (currentAssistant) currentAssistant.textContent = `${t("status.error")} ${e}`;
    currentAssistant = null;
    chatBusy = false;
    $("#chat-status").textContent = `${t("status.error")} ${e}`;
  }
}

$("#chat-send").addEventListener("click", chatSend);
$("#chat-input").addEventListener("keydown", (e) => { if (e.key === "Enter") chatSend(); });
$("#chat-speak").addEventListener("change", sendChatConfig);
$("#chat-stop").addEventListener("click", () => {
  resetChatAudio();
  if (chatSocket && chatSocket.readyState === 1) chatSocket.send(JSON.stringify({ type: "input.cancel" }));
  if (currentAssistant) { currentAssistant.classList.add("cancelled"); currentAssistant = null; }
  chatBusy = false;
});

// Chat dictation
let dictateActive = false, dictateCtx = null, dictateStream = null, dictateNode = null;
let dictateSocket = null, dictateText = "", dictatePartial = "", dictateTimer = null;

async function startDictation() {
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("#chat-status").textContent = t("stt.micInsecure");
    return;
  }
  try {
    dictateStream = await openMicStream();
    dictateCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = dictateCtx.createMediaStreamSource(dictateStream);
    dictateNode = dictateCtx.createScriptProcessor(4096, 1, 1);
    dictateSocket = new WebSocket(wsURL("/ws/stt"));
    dictateSocket.binaryType = "arraybuffer";
    dictateText = ""; dictatePartial = "";
    dictateSocket.onopen = () => {
      dictateSocket.send(JSON.stringify({
        type: "session.update",
        session: { sample_rate: 16000, language: $("#stt-lang").value || undefined, automatic_punctuation: true },
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
    dictateNode.onaudioprocess = (e) => {
      if (!dictateSocket || dictateSocket.readyState !== 1) return;
      dictateSocket.send(downsampleTo16k(e.inputBuffer.getChannelData(0), dictateCtx.sampleRate).buffer);
    };
    source.connect(dictateNode);
    dictateNode.connect(dictateCtx.destination);
    dictateActive = true;
    $("#chat-mic").textContent = "⏹";
  } catch (e) {
    $("#chat-status").textContent = `${t("status.error")} ${e}`;
  }
}

function finishDictation() {
  if (dictateNode) dictateNode.disconnect();
  if (dictateCtx) dictateCtx.close();
  if (dictateStream) dictateStream.getTracks().forEach((track) => track.stop());
  dictateSocket = null; dictateCtx = null; dictateStream = null; dictateNode = null;
  dictateActive = false;
  $("#chat-mic").textContent = "🎤";
  const text = (dictateText + dictatePartial).trim();
  if (text) { $("#chat-input").value = text; chatSend(); }
  else if ($("#chat-status")) $("#chat-status").textContent = t("stt.noSpeech");
}

function stopDictation() {
  if (dictateSocket && dictateSocket.readyState === 1) {
    try { dictateSocket.send(JSON.stringify({ type: "input_audio_buffer.commit" })); } catch (e) { /* ignore */ }
  }
  clearTimeout(dictateTimer);
  dictateTimer = setTimeout(finishDictation, 900);
}

$("#chat-mic").addEventListener("click", () => { if (dictateActive) stopDictation(); else startDictation(); });

// ------------------------------------------------------------- settings tab
function buildStartupVoiceSelect() {
  const sel = $("#set-startup-voice");
  if (!sel) return;
  const current = sel.value || (CONFIG ? CONFIG.startup.voice : "");
  sel.innerHTML = "";
  const add = (value, label) => {
    const o = document.createElement("option"); o.value = value; o.textContent = label; sel.appendChild(o);
  };
  add("", "—");
  add("custom_voice", ttsName("custom_voice"));
  add("voice_design", ttsName("voice_design"));
  add("base", ttsName("base"));
  avatars.forEach((a) => add(`avatar:${a.id}`, `${t("avatars.voicePrefix")}: ${a.name}`));
  sel.value = current || "";
}

$("#set-startup-save").addEventListener("click", async () => {
  $("#set-startup-status").textContent = "…";
  try {
    const r = await apiFetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ startup: { stt_model: settingsSttPicker ? settingsSttPicker.value : "", voice: $("#set-startup-voice").value } }),
    });
    const text = await r.text();
    if (!r.ok) {
      let detail = text; try { detail = JSON.parse(text).detail || text; } catch (e) { /* keep */ }
      throw new Error(detail);
    }
    $("#set-startup-status").textContent = t("status.saved");
  } catch (e) {
    $("#set-startup-status").textContent = `${t("status.error")} ${e}`;
  }
});

async function saveSettings() {
  const num = (sel, parse) => {
    const v = parse($(sel).value);
    return Number.isFinite(v) ? v : undefined;
  };
  const llm = {
    base_url: $("#set-llm-url").value.trim(),
    model: $("#set-llm-model").value.trim(),
    system_prompt: $("#set-llm-system").value,
  };
  const key = $("#set-llm-key").value;
  if (key) llm.api_key = key;
  const payload = { llm, generation: {}, chunking: {} };
  const temp = num("#set-temp", parseFloat);
  if (temp !== undefined) payload.generation.temperature = temp;
  const batch = num("#set-batch", (v) => parseInt(v, 10));
  if (batch !== undefined) payload.generation.batch_size = batch;
  const maxc = num("#set-maxchars", (v) => parseInt(v, 10));
  if (maxc !== undefined) payload.chunking.max_chars = maxc;
  try {
    const r = await apiFetch("/api/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const text = await r.text();
    if (!r.ok) {
      let detail = text; try { detail = JSON.parse(text).detail || text; } catch (e) { /* keep */ }
      return { ok: false, detail };
    }
    let cfg = null; try { cfg = JSON.parse(text); } catch (e) { /* ignore */ }
    if (cfg && cfg.llm) {
      CONFIG = Object.assign({}, CONFIG, { llm: cfg.llm, generation: cfg.generation, chunking: cfg.chunking });
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

$("#set-save").addEventListener("click", async () => {
  $("#set-save-status").textContent = "…";
  const res = await saveSettings();
  $("#set-save-status").textContent = res.ok ? t("status.saved") : `${t("status.error")} ${res.detail}`;
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

$("#set-apikey-save").addEventListener("click", () => {
  localStorage.setItem("apiKey", $("#set-apikey").value);
  $("#set-save-status").textContent = t("status.remembered");
});

init();
