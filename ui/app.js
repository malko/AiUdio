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
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function wsURL(path) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const key = apiKey();
  return `${proto}://${location.host}${path}${key ? `?api_key=${encodeURIComponent(key)}` : ""}`;
}

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

// --------------------------------------------------------------- global state
let CONFIG = null;
const TTS_SESSION = { format: "pcm", sample_rate: 24000 };

// i18n: apply static labels + language switcher.
window.I18N.init();
window.I18N.apply();
$("#lang-select").addEventListener("change", (e) => window.I18N.setLang(e.target.value));

// --------------------------------------------------------------------- tabs
document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((el) => el.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");
  });
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

  const fam = $("#tts-family");
  fam.innerHTML = "";
  (CONFIG.models.families || []).forEach((m) => {
    const o = document.createElement("option");
    o.value = m.family;
    o.textContent = `${m.family}${m.loaded ? " *" : ""}`;
    fam.appendChild(o);
  });
  fam.value = CONFIG.models.default_family;
  fam.addEventListener("change", loadVoices);

  const lang = $("#tts-language");
  lang.innerHTML = "";
  (CONFIG.languages || ["Auto"]).forEach((l) => {
    const o = document.createElement("option");
    o.value = l; o.textContent = l; lang.appendChild(o);
  });

  $("#set-llm-url").value = CONFIG.llm.base_url || "";
  $("#set-llm-model").value = CONFIG.llm.model || "";
  $("#set-llm-system").value = CONFIG.llm.system_prompt || "";
  $("#set-temp").value = CONFIG.generation.temperature;
  $("#set-batch").value = CONFIG.generation.batch_size;
  $("#set-maxchars").value = CONFIG.chunking.max_chars;
  $("#set-apikey").value = apiKey();

  await loadVoices();
  refreshStatus();
}

async function loadVoices() {
  const family = $("#tts-family").value;
  try {
    const data = await getJSON(`/api/voices?family=${encodeURIComponent(family)}`);
    const sel = $("#tts-speaker");
    sel.innerHTML = "";
    (data.speakers || []).forEach((s) => {
      const o = document.createElement("option"); o.value = s; o.textContent = s; sel.appendChild(o);
    });
    if (!data.speakers || !data.speakers.length) {
      const o = document.createElement("option"); o.value = ""; o.textContent = "(n/a)"; sel.appendChild(o);
    }
  } catch (e) { /* ignore */ }
}

async function refreshStatus() {
  try {
    const h = await getJSON("/health");
    $("#health").className = "dot ok";
    $("#active-model").textContent = h.active_family
      ? `${t("status.active")}: ${h.active_family}`
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
  // Reuse the transcription proxy to prefill the reference transcript.
  try {
    const r = await apiFetch("/v1/audio/transcriptions", { method: "POST", body: fd });
    if (r.ok) { const j = await r.json(); if (j.text) $("#tts-ref-text").value = j.text; }
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
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error(await r.text());
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
      if (!r.ok) throw new Error(await r.text());
      const blob = await r.blob();
      $("#tts-player").src = URL.createObjectURL(blob);
      $("#tts-status").textContent = t("status.done");
    } catch (e) {
      $("#tts-status").textContent = t("status.error");
      $("#tts-log").textContent = String(e);
    }
  }
  refreshStatus();
});

// ------------------------------------------------------------------ STT tab
let micCtx = null, micStream = null, micNode = null, sttSocket = null;
let sttFinal = "";

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

$("#stt-record").addEventListener("click", async () => {
  const btn = $("#stt-record");
  if (sttSocket) { stopMic(); return; }
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = micCtx.createMediaStreamSource(micStream);
    micNode = micCtx.createScriptProcessor(4096, 1, 1);
    sttSocket = new WebSocket(wsURL("/ws/stt"));
    sttSocket.binaryType = "arraybuffer";
    sttFinal = $("#stt-text").textContent;

    sttSocket.onopen = () => {
      sttSocket.send(JSON.stringify({
        type: "session.update",
        session: { sample_rate: 16000, language: $("#stt-lang").value || undefined,
                   automatic_punctuation: true },
      }));
      $("#stt-status").textContent = t("stt.recording");
      btn.textContent = t("stt.stop");
      btn.classList.remove("primary");
    };
    sttSocket.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      let msg; try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.type && msg.type.includes("delta")) {
        $("#stt-text").textContent = sttFinal + msg.delta;
      } else if (msg.type && msg.type.includes("completed")) {
        sttFinal += (msg.text || msg.transcript || "") + " ";
        $("#stt-text").textContent = sttFinal;
      }
    };
    sttSocket.onclose = () => stopMic();

    micNode.onaudioprocess = (e) => {
      if (!sttSocket || sttSocket.readyState !== 1) return;
      const pcm = downsampleTo16k(e.inputBuffer.getChannelData(0), micCtx.sampleRate);
      sttSocket.send(pcm.buffer);
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
    const data = await r.json();
    $("#stt-file-out").textContent = JSON.stringify(
      data.words ? { text: data.text, words: data.words } : data, null, 2);
  } catch (e) {
    $("#stt-file-out").textContent = `${t("status.error")} ${e}`;
  }
});

// ----------------------------------------------------------------- chat tab
let chatSocket = null, chatPlayer = null, currentAssistant = null;

function appendTurn(role, text) {
  const div = document.createElement("div");
  div.className = "turn";
  div.innerHTML = `<span class="speaker">${role}</span>`;
  const span = document.createElement("span");
  span.className = role === "user" ? "user" : "assistant";
  span.textContent = text;
  div.appendChild(span);
  $("#chat-log").appendChild(div);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
  return span;
}

function ensureChatSocket() {
  if (chatSocket && chatSocket.readyState === 1) return Promise.resolve(chatSocket);
  return new Promise((resolve, reject) => {
    chatSocket = new WebSocket(wsURL("/ws/chat"));
    chatSocket.binaryType = "arraybuffer";
    chatSocket.onopen = () => {
      chatSocket.send(JSON.stringify({
        type: "session.config",
        tts: { family: $("#tts-family").value, language: $("#tts-language").value,
               speaker: $("#tts-speaker").value, format: "pcm", chunking: true },
      }));
      resolve(chatSocket);
    };
    chatSocket.onerror = reject;
  });
}

async function chatSend() {
  const text = $("#chat-input").value.trim();
  if (!text) return;
  $("#chat-input").value = "";
  appendTurn("user", text);
  currentAssistant = appendTurn("assistant", "");
  const sock = await ensureChatSocket();
  sock.onmessage = (ev) => {
    if (typeof ev.data !== "string") {
      if (chatPlayer && $("#chat-speak").checked) chatPlayer.pushBytes(new Uint8Array(ev.data));
      return;
    }
    const msg = JSON.parse(ev.data);
    if (msg.type === "llm.delta") currentAssistant.textContent += msg.text;
    else if (msg.type === "audio.start") {
      if (!chatPlayer) chatPlayer = new PCMPlayer(TTS_SESSION.sample_rate);
      chatPlayer.reset();
    } else if (msg.type === "llm.done") {
      currentAssistant = null;
    } else if (msg.type === "error") {
      currentAssistant.textContent += ` [error: ${msg.message}]`;
    }
  };
  sock.send(JSON.stringify({ type: "input.text", text }));
}

$("#chat-send").addEventListener("click", chatSend);
$("#chat-input").addEventListener("keydown", (e) => { if (e.key === "Enter") chatSend(); });
$("#chat-stop").addEventListener("click", () => {
  if (chatPlayer) { chatPlayer.stop(); chatPlayer = null; }
});

// ------------------------------------------------------------- settings tab
$("#set-llm-ping").addEventListener("click", async () => {
  $("#set-llm-status").textContent = t("status.testing");
  try {
    const r = await getJSON("/api/llm/ping");
    $("#set-llm-status").textContent = r.ok ? `OK (${r.url})` : `${t("status.error")} ${r.error || r.status}`;
  } catch (e) { $("#set-llm-status").textContent = `${t("status.error")} ${e}`; }
});

$("#set-save").addEventListener("click", async () => {
  const payload = {
    llm: {
      base_url: $("#set-llm-url").value,
      model: $("#set-llm-model").value,
      api_key: $("#set-llm-key").value || undefined,
      system_prompt: $("#set-llm-system").value,
    },
    generation: { temperature: parseFloat($("#set-temp").value), batch_size: parseInt($("#set-batch").value, 10) },
    chunking: { max_chars: parseInt($("#set-maxchars").value, 10) },
  };
  try {
    await apiFetch("/api/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    $("#set-save-status").textContent = t("status.saved");
  } catch (e) { $("#set-save-status").textContent = `${t("status.error")} ${e}`; }
});

$("#set-apikey-save").addEventListener("click", () => {
  localStorage.setItem("apiKey", $("#set-apikey").value);
  $("#set-save-status").textContent = t("status.remembered");
});

init();
