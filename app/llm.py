"""External LLM client (OpenAI-compatible API: llama.cpp, Ollama, vLLM, OpenAI...).

The LLM is never served by this project: we only connect to it in streaming mode.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Dict, List

import httpx

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, cfg):
        self.cfg = cfg

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.cfg.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _payload(self, messages: List[Dict[str, str]], stream: bool) -> Dict:
        payload: Dict = {
            "messages": messages,
            "temperature": self.cfg.temperature,
            "stream": stream,
        }
        if self.cfg.model:
            payload["model"] = self.cfg.model
        if self.cfg.max_tokens and self.cfg.max_tokens > 0:
            payload["max_tokens"] = self.cfg.max_tokens
        return payload

    async def stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        payload = self._payload(messages, stream=True)
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            async with client.stream(
                "POST", self._url("chat/completions"), json=payload, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content

    async def complete(self, messages: List[Dict[str, str]]) -> str:
        payload = self._payload(messages, stream=False)
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            resp = await client.post(
                self._url("chat/completions"), json=payload, headers=self._headers()
            )
            resp.raise_for_status()
            data = resp.json()
            return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")

    async def ping(self) -> Dict:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(self._url("models"), headers=self._headers())
            return {"ok": resp.status_code < 400, "status": resp.status_code, "url": self.cfg.base_url}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "url": self.cfg.base_url}


def build_messages(system_prompt: str, history: List[Dict[str, str]], user_text: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def trim_history(history: List[Dict[str, str]], max_turns: int) -> List[Dict[str, str]]:
    if max_turns <= 0:
        return history
    return history[-2 * max_turns:]
