"""Chatterbox backend client.

Chatterbox (Resemble AI) cannot share the gateway's Python environment: the
``chatterbox-tts`` package pins different ``torch``/``transformers`` versions than
``qwen-tts``. We therefore run it in an **isolated virtualenv** (``/opt/cb/venv``
by default) as a child process, driven over a line-delimited JSON protocol.

This module is the gateway-side client. The child process lives in
``app/chatterbox_worker.py``.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_WORKER = Path(__file__).resolve().parent / "chatterbox_worker.py"
_DEFAULT_PYTHON = "/opt/cb/venv/bin/python"
_TURBO_FAMILIES = ("chatterbox_turbo", "chatterbox_nano")


def resolve_python(configured: str = "") -> Optional[str]:
    """Return a usable interpreter for the Chatterbox venv, or None."""
    candidates = [configured] if configured else []
    candidates.append(_DEFAULT_PYTHON)
    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


class ChatterboxClient:
    """Manages a single Chatterbox worker process and one loaded model."""

    def __init__(self, python: str, family: str, repo: str, *, t3_model: str = "v3",
                 device: str = "cpu", watermark: bool = False, timeout: float = 600.0):
        self.family = family
        self.repo = repo
        self.timeout = timeout
        self._queue: queue.Queue[dict] = queue.Queue()
        self._lock = threading.Lock()
        env = dict(os.environ)
        # An empty HF_TOKEN (docker-compose sets it to "" when unset) makes
        # huggingface_hub send an invalid "Authorization: Bearer " header.
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if var in env and not env[var].strip():
                del env[var]
        self._proc = subprocess.Popen(
            [python, "-u", str(_WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._request({
            "cmd": "load", "family": family, "repo": repo,
            "t3_model": t3_model,
            "nano": family == "chatterbox_nano",
            "device": device, "watermark": watermark,
        })

    # -------------------------------------------------------------- transport
    def _read_stdout(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._queue.put(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Chatterbox worker sent non-JSON: %s", line[:200])

    def _read_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            log.debug("[chatterbox] %s", line.rstrip())

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError("Chatterbox worker is not running")
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
            try:
                reply = self._queue.get(timeout=self.timeout)
            except queue.Empty:
                raise RuntimeError("Chatterbox worker timed out") from None
        if not reply.get("ok"):
            msg = reply.get("error", "unknown error")
            log.error("Chatterbox worker error: %s\n%s", msg, reply.get("trace", ""))
            raise RuntimeError(f"Chatterbox: {msg}")
        return reply

    # ------------------------------------------------------------------- api
    def languages(self) -> List[str]:
        return list(self._request({"cmd": "languages"}).get("languages") or [])

    def synthesize(self, texts: List[str], *, language: str, audio_prompt: Optional[str],
                   params: Dict[str, Any]) -> List[Tuple[np.ndarray, int]]:
        import soundfile as sf

        out_dir = tempfile.mkdtemp(prefix="cb-out-")
        try:
            reply = self._request({
                "cmd": "synth",
                "texts": [str(t) for t in texts],
                "language": language,
                "audio_prompt": audio_prompt,
                "params": params,
                "out_dir": out_dir,
            })
        except Exception:
            self._cleanup(out_dir)
            raise
        results: List[Tuple[np.ndarray, int]] = []
        for item in reply.get("items", []):
            audio, sr = sf.read(item["path"], dtype="float32", always_2d=False)
            results.append((np.asarray(audio, dtype=np.float32), int(item.get("sr") or sr)))
        self._cleanup(out_dir)
        return results

    @staticmethod
    def _cleanup(out_dir: str) -> None:
        try:
            for f in Path(out_dir).glob("*.wav"):
                f.unlink(missing_ok=True)
            Path(out_dir).rmdir()
        except OSError:
            pass

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                self._request({"cmd": "shutdown"})
        except Exception:  # best effort
            pass
        finally:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()


def is_available(settings) -> bool:
    return resolve_python(settings.models.chatterbox.python) is not None
