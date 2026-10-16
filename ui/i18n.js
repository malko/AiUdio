"use strict";

// Lightweight i18n for the web UI. Add a locale by extending MESSAGES.
// The repository/docs/code stay in English; the product itself is multilingual.
const MESSAGES = {
  en: {
    "app.title": "AiUdio Server",
    "nav.synthesis": "Synthesis",
    "nav.transcription": "Transcription",
    "nav.conversation": "Conversation",
    "nav.settings": "Settings",

    "tts.textLabel": "Text",
    "tts.textPlaceholder": "Hello, this is a test.",
    "tts.family": "Mode / family",
    "tts.language": "Language",
    "tts.speaker": "Voice (speaker)",
    "tts.instruct": "Instruction (style / emotion)",
    "tts.instructPlaceholder": "e.g. Speak with excitement",
    "tts.cloneTitle": "Voice cloning (family \"base\")",
    "tts.refAudio": "Reference audio",
    "tts.refText": "Reference transcript",
    "tts.xvec": "x-vector only",
    "tts.stream": "Streaming (PCM)",
    "tts.chunk": "Split by sentence",
    "tts.generate": "Generate",
    "tts.playback": "Playback",
    "tts.ready": "Ready.",

    "stt.micTitle": "Microphone (real time)",
    "stt.start": "Start",
    "stt.stop": "Stop",
    "stt.language": "Language",
    "stt.stopped": "Stopped.",
    "stt.recording": "Recording...",
    "stt.transcription": "Transcription",
    "stt.fileTitle": "Audio file",
    "stt.file": "File",
    "stt.diar": "Diarization (who speaks)",
    "stt.transcribe": "Transcribe",
    "stt.modelsTitle": "STT models (on demand)",
    "stt.model": "Model",
    "stt.download": "Download",
    "stt.refresh": "Refresh",
    "stt.cached": "cached",
    "stt.notCached": "not cached",
    "stt.downloading": "Downloading",
    "stt.downloadedMsg": "Download complete. The STT service starts automatically.",
    "stt.indexUnavailable": "Model index unavailable — start the STT service once.",
    "stt.sizeMb": "MB",

    "chat.title": "Conversation (external LLM)",
    "chat.placeholder": "Your message...",
    "chat.dictate": "Dictate via STT",
    "chat.send": "Send",
    "chat.stop": "Stop",
    "chat.speak": "Speak the reply (TTS)",

    "settings.llmTitle": "External LLM",
    "settings.baseUrl": "Base URL",
    "settings.model": "Model",
    "settings.apiKey": "API key",
    "settings.systemPrompt": "System prompt",
    "settings.test": "Test connection",
    "settings.genTitle": "Generation / audio",
    "settings.temperature": "Temperature",
    "settings.batch": "Batch size",
    "settings.chunkMax": "Chunk max (chars)",
    "settings.save": "Save",
    "settings.clientKeyTitle": "Client API key",
    "settings.clientKeyPlaceholder": "if auth is enabled",
    "settings.remember": "Remember (localStorage)",
    "settings.clientKeyHint": "Used by the UI for protected endpoints.",

    "status.online": "online",
    "status.offline": "offline",
    "status.active": "active",
    "status.noModel": "no model loaded",
    "status.configUnavailable": "config unavailable",
    "status.generating": "Generating...",
    "status.done": "Done.",
    "status.error": "Error.",
    "status.saved": "Saved.",
    "status.testing": "Testing...",
    "status.remembered": "Key stored.",
    "status.micUnavailable": "Microphone unavailable",
  },
  fr: {
    "app.title": "AiUdio Server",
    "nav.synthesis": "Synthèse",
    "nav.transcription": "Transcription",
    "nav.conversation": "Conversation",
    "nav.settings": "Réglages",

    "tts.textLabel": "Texte",
    "tts.textPlaceholder": "Bonjour, ceci est un test.",
    "tts.family": "Mode / famille",
    "tts.language": "Langue",
    "tts.speaker": "Voix (speaker)",
    "tts.instruct": "Instruction (style / émotion)",
    "tts.instructPlaceholder": "ex. Speak with excitement",
    "tts.cloneTitle": "Clonage de voix (famille « base »)",
    "tts.refAudio": "Audio de référence",
    "tts.refText": "Transcription de référence",
    "tts.xvec": "x-vector uniquement",
    "tts.stream": "Streaming (PCM)",
    "tts.chunk": "Découper par phrase",
    "tts.generate": "Générer",
    "tts.playback": "Lecture",
    "tts.ready": "Prêt.",

    "stt.micTitle": "Micro (temps réel)",
    "stt.start": "Démarrer",
    "stt.stop": "Arrêter",
    "stt.language": "Langue",
    "stt.stopped": "Arrêté.",
    "stt.recording": "Enregistrement…",
    "stt.transcription": "Transcription",
    "stt.fileTitle": "Fichier audio",
    "stt.file": "Fichier",
    "stt.diar": "Diarisation (qui parle)",
    "stt.transcribe": "Transcrire",
    "stt.modelsTitle": "Modèles STT (à la demande)",
    "stt.model": "Modèle",
    "stt.download": "Télécharger",
    "stt.refresh": "Rafraîchir",
    "stt.cached": "en cache",
    "stt.notCached": "absent",
    "stt.downloading": "Téléchargement",
    "stt.downloadedMsg": "Téléchargement terminé. Le service STT démarre automatiquement.",
    "stt.indexUnavailable": "Index des modèles indisponible — démarrez le service STT une fois.",
    "stt.sizeMb": "Mo",

    "chat.title": "Conversation (LLM externe)",
    "chat.placeholder": "Votre message…",
    "chat.dictate": "Dicter via STT",
    "chat.send": "Envoyer",
    "chat.stop": "Stop",
    "chat.speak": "Lire la réponse (TTS)",

    "settings.llmTitle": "LLM externe",
    "settings.baseUrl": "URL de base",
    "settings.model": "Modèle",
    "settings.apiKey": "Clé API",
    "settings.systemPrompt": "System prompt",
    "settings.test": "Tester la connexion",
    "settings.genTitle": "Génération / audio",
    "settings.temperature": "Température",
    "settings.batch": "Taille de lot (batch)",
    "settings.chunkMax": "Chunk max (caractères)",
    "settings.save": "Enregistrer",
    "settings.clientKeyTitle": "Clé d'accès API (client)",
    "settings.clientKeyPlaceholder": "si auth activée",
    "settings.remember": "Mémoriser (localStorage)",
    "settings.clientKeyHint": "Utilisée par l'interface pour les endpoints protégés.",

    "status.online": "en ligne",
    "status.offline": "hors ligne",
    "status.active": "actif",
    "status.noModel": "aucun modèle chargé",
    "status.configUnavailable": "config indisponible",
    "status.generating": "Génération…",
    "status.done": "Terminé.",
    "status.error": "Erreur.",
    "status.saved": "Enregistré.",
    "status.testing": "Test…",
    "status.remembered": "Clé mémorisée.",
    "status.micUnavailable": "Micro indisponible",
  },
};

const SUPPORTED = Object.keys(MESSAGES);

const I18N = {
  lang: "en",
  t(key) {
    return (MESSAGES[this.lang] && MESSAGES[this.lang][key]) || MESSAGES.en[key] || key;
  },
  setLang(lang) {
    this.lang = SUPPORTED.includes(lang) ? lang : "en";
    localStorage.setItem("lang", this.lang);
    document.documentElement.lang = this.lang;
    this.apply();
  },
  init() {
    const stored = localStorage.getItem("lang");
    const browser = (navigator.language || "en").slice(0, 2);
    this.lang = SUPPORTED.includes(stored) ? stored : (SUPPORTED.includes(browser) ? browser : "en");
    document.documentElement.lang = this.lang;
  },
  apply() {
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      el.textContent = this.t(el.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-ph]").forEach((el) => {
      el.placeholder = this.t(el.dataset.i18nPh);
    });
    document.querySelectorAll("[data-i18n-title]").forEach((el) => {
      el.title = this.t(el.dataset.i18nTitle);
    });
    const select = document.getElementById("lang-select");
    if (select) select.value = this.lang;
  },
};

window.I18N = I18N;
