"""Piloting the vLLM-Omni container through the Docker API (opt-in)."""

import asyncio
import json

import httpx

from app import docker_ctl


def _daemon(monkeypatch, containers, calls):
    """A fake Docker daemon: answers the handful of routes we use."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        path = request.url.path
        if path == "/containers/json":
            return httpx.Response(200, json=containers)
        if path.endswith("/json"):
            return httpx.Response(200, json=containers[0]["_inspect"])
        if path == "/containers/create":
            body = json.loads(request.content)
            assert body["Cmd"] and body["Image"]
            calls.append(("CREATE", tuple(body["Cmd"])))
            # The daemon now reports the recreated container, as Docker does.
            containers[0]["_inspect"]["Config"]["Cmd"] = body["Cmd"]
            containers[0]["_inspect"]["State"] = {"Running": False, "Status": "created"}
            return httpx.Response(201, json={"Id": "new"})
        if path == "/containers/new/start":
            containers[0]["_inspect"]["State"] = {"Running": True, "Status": "running"}
            return httpx.Response(204)
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        docker_ctl, "client",
        lambda socket, timeout=30.0: httpx.AsyncClient(transport=transport, base_url="http://docker"),
    )


def _inspect(cmd, running=True):
    return {
        "Id": "abcdef1234567890",
        "Name": "/tts-vllm",
        "State": {"Running": running, "Status": "running" if running else "exited"},
        "Config": {"Image": "vllm/vllm-omni:latest", "Cmd": cmd,
                   "Env": ["HF_TOKEN="], "Labels": {}},
        "HostConfig": {"ShmSize": 8589934592, "Binds": ["hf_cache:/root/.cache/huggingface"]},
        "NetworkSettings": {"Networks": {"aiudio-net": {}}},
    }


CMD = ["vllm", "serve", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "--omni", "--port", "8091"]


def test_served_model_and_rewrite_keep_the_flags():
    assert docker_ctl.served_model(CMD) == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert docker_ctl.served_model(["python", "-m", "x"]) is None
    assert docker_ctl.with_model(CMD, "Qwen/Qwen3-TTS-12Hz-0.6B-Base") == [
        "vllm", "serve", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "--omni", "--port", "8091",
    ]


def test_with_model_rejects_a_non_vllm_command():
    import pytest

    with pytest.raises(docker_ctl.DockerError):
        docker_ctl.with_model(["sleep", "infinity"], "x")


def test_status_reports_the_served_checkpoint(monkeypatch):
    calls = []
    _daemon(monkeypatch, [{"Id": "abcdef1234567890", "Names": ["/tts-vllm"],
                           "_inspect": _inspect(CMD)}], calls)
    state = asyncio.run(docker_ctl.status("/var/run/docker.sock", "tts-vllm"))
    assert state["exists"] and state["running"]
    assert state["model"] == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def test_start_does_nothing_when_already_running(monkeypatch):
    calls = []
    _daemon(monkeypatch, [{"Id": "abcdef1234567890", "Names": ["/tts-vllm"],
                           "_inspect": _inspect(CMD)}], calls)
    asyncio.run(docker_ctl.start("/var/run/docker.sock", "tts-vllm"))
    assert ("POST", "/containers/abcdef1234567890/start") not in calls


def test_set_model_recreates_with_the_new_checkpoint(monkeypatch):
    calls = []
    _daemon(monkeypatch, [{"Id": "abcdef1234567890", "Names": ["/tts-vllm"],
                           "_inspect": _inspect(CMD)}], calls)
    asyncio.run(
        docker_ctl.set_model(
            "/var/run/docker.sock", "tts-vllm", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
    )
    paths = [p for _, p in calls]
    # Stop → rename out of the way → create → start → remove the previous one.
    assert "/containers/abcdef1234567890/stop" in paths
    assert "/containers/abcdef1234567890/rename" in paths
    assert "/containers/create" in paths
    assert "/containers/new/start" in paths
    assert ("DELETE", "/containers/abcdef1234567890") in calls
    created = [c for c in calls if c[0] == "CREATE"][0][1]
    assert created[2] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def test_set_model_is_a_noop_when_it_already_serves_it(monkeypatch):
    calls = []
    _daemon(monkeypatch, [{"Id": "abcdef1234567890", "Names": ["/tts-vllm"],
                           "_inspect": _inspect(CMD)}], calls)
    state = asyncio.run(
        docker_ctl.set_model(
            "/var/run/docker.sock", "tts-vllm", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        )
    )
    assert state["model"] == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert "/containers/create" not in [p for _, p in calls]


def test_missing_container_is_reported(monkeypatch):
    calls = []
    _daemon(monkeypatch, [], calls)

    import pytest

    with pytest.raises(docker_ctl.DockerError):
        asyncio.run(docker_ctl.start("/var/run/docker.sock", "tts-vllm"))


def test_set_model_can_prepare_a_checkpoint_without_starting_it(monkeypatch):
    calls = []
    _daemon(monkeypatch, [{"Id": "abcdef1234567890", "Names": ["/tts-vllm"],
                           "_inspect": _inspect(CMD, running=False)}], calls)
    state = asyncio.run(
        docker_ctl.set_model("/var/run/docker.sock", "tts-vllm",
                             "Qwen/Qwen3-TTS-12Hz-0.6B-Base", start=False)
    )
    paths = [p for _, p in calls]
    assert "/containers/create" in paths
    assert "/containers/new/start" not in paths  # the engine is local: stay stopped
    assert state["model"] == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def test_set_model_stops_when_the_same_checkpoint_should_not_run(monkeypatch):
    calls = []
    _daemon(monkeypatch, [{"Id": "abcdef1234567890", "Names": ["/tts-vllm"],
                           "_inspect": _inspect(CMD)}], calls)
    asyncio.run(
        docker_ctl.set_model("/var/run/docker.sock", "tts-vllm",
                             "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", start=False)
    )
    assert ("POST", "/containers/abcdef1234567890/stop") in calls
