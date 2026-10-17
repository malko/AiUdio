// Conversation over /ws/chat (Node >= 22, global WebSocket).
//   node chat.mjs "Hello, how are you?"
// Writes the assistant audio to reply.pcm (mono s16le PCM).
import { createWriteStream } from "node:fs";

const BASE = process.env.AIUDIO_BASE_URL || "http://localhost:8004";
const KEY = process.env.AIUDIO_API_KEY || "";
const text = process.argv.slice(2).join(" ") || "Hello, how are you?";

const url = BASE.replace(/^https/, "wss").replace(/^http/, "ws").replace(/\/$/, "")
  + "/ws/chat" + (KEY ? `?api_key=${encodeURIComponent(KEY)}` : "");

const ws = new WebSocket(url);
ws.binaryType = "arraybuffer";
const pcm = createWriteStream("reply.pcm");

ws.addEventListener("open", () => {
  ws.send(JSON.stringify({ type: "session.config", speak: true, tts: { speaker: "Vivian" } }));
  ws.send(JSON.stringify({ type: "input.text", text }));
});

ws.addEventListener("message", (event) => {
  if (typeof event.data === "string") {
    const ev = JSON.parse(event.data);
    if (ev.type === "llm.delta") process.stdout.write(ev.text || "");
    else console.log(`[${ev.type}]`, ev.text || ev.message || ev.sentence_text || "");
    if (ev.type === "llm.done" || ev.type === "turn.cancelled") ws.close();
  } else {
    pcm.write(Buffer.from(event.data));
  }
});

ws.addEventListener("close", () => {
  pcm.end();
  console.log("\nwrote reply.pcm");
});
