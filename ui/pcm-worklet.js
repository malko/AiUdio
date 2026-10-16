// Minimal capture worklet: buffers mono input and forwards fixed-size frames to
// the main thread (replaces the deprecated ScriptProcessorNode).
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = 4096;
    this.buf = new Float32Array(this.size);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const data = input[0];
    let i = 0;
    while (i < data.length) {
      const take = Math.min(this.size - this.filled, data.length - i);
      this.buf.set(data.subarray(i, i + take), this.filled);
      this.filled += take;
      i += take;
      if (this.filled === this.size) {
        this.port.postMessage(this.buf);
        this.buf = new Float32Array(this.size);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
