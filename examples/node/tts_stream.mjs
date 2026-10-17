// Streaming TTS -> raw mono s16le PCM (Node >= 18).
//   node tts_stream.mjs "Sentence one. Sentence two."
// Play: ffplay -f s16le -ar 24000 -ch_layout mono out.pcm
import { createWriteStream } from "node:fs";

const BASE = process.env.AIUDIO_BASE_URL || "http://localhost:8004";
const KEY = process.env.AIUDIO_API_KEY || "";
const text = process.argv.slice(2).join(" ") || "Sentence one. Sentence two. Sentence three.";

const r = await fetch(`${BASE}/api/tts`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    ...(KEY ? { "X-API-Key": KEY } : {}),
  },
  body: JSON.stringify({ text, stream: true, format: "pcm" }),
});
if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);

const out = createWriteStream("out.pcm");
for await (const chunk of r.body) {
  // A real app would feed `chunk` into a jitter buffer here.
  out.write(Buffer.from(chunk));
}
out.end();
console.log("wrote out.pcm", r.headers.get("x-sample-rate"), "Hz");
