// One-shot TTS (Node >= 18, built-in fetch).
//   node tts.mjs "Hello world"
import { writeFile } from "node:fs/promises";

const BASE = process.env.AIUDIO_BASE_URL || "http://localhost:8004";
const KEY = process.env.AIUDIO_API_KEY || "";
const text = process.argv.slice(2).join(" ") || "Hello from AiUdio.";

const r = await fetch(`${BASE}/api/tts`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    ...(KEY ? { "X-API-Key": KEY } : {}),
  },
  body: JSON.stringify({
    text, family: "custom_voice", speaker: "Vivian", language: "English",
  }),
});
if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
await writeFile("out.wav", Buffer.from(await r.arrayBuffer()));
console.log("wrote out.wav");
