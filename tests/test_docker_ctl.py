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
            calls.append(("CREATE_BODY", body))
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


def _daemon_entries(monkeypatch, containers, calls):
    """Fake daemon whose listing entries carry their own inspect payload."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        path = request.url.path
        if path == "/containers/json":
            listing = [{k: v for k, v in c.items() if k != "_inspect"} for c in containers]
            return httpx.Response(200, json=listing)
        if path.endswith("/json"):
            cid = path.split("/")[2]
            for c in containers:
                if c["Id"] == cid:
                    # The daemon answers with the container's real id.
                    return httpx.Response(200, json=dict(c["_inspect"], Id=c["Id"]))
            return httpx.Response(404, json={"message": "no such container"})
        if path == "/containers/create":
            return httpx.Response(201, json={"Id": "new"})
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


def test_set_model_preserves_the_network_aliases(monkeypatch):
    calls = []
    inspect = _inspect(CMD)
    # Compose attaches the service under its own name too (`tts-vllm`).
    inspect["NetworkSettings"]["Networks"] = {
        "aiudio-net": {"Aliases": ["aiudio-tts-vllm-1", "tts-vllm"]}
    }
    _daemon(monkeypatch, [{"Id": "abcdef1234567890", "Names": ["/tts-vllm"],
                           "_inspect": inspect}], calls)
    asyncio.run(
        docker_ctl.set_model(
            "/var/run/docker.sock", "tts-vllm", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
    )
    body = [c[1] for c in calls if c[0] == "CREATE_BODY"][0]
    # Without the aliases the gateway can no longer resolve `tts-vllm`.
    assert body["NetworkingConfig"]["EndpointsConfig"] == {
        "aiudio-net": {"Aliases": ["aiudio-tts-vllm-1", "tts-vllm"]}
    }


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


def test_with_flags_replaces_adds_and_removes_arguments():
    assert docker_ctl.flag_value(CMD, "--max-model-len") is None
    assert not docker_ctl.has_flag(CMD, "--enforce-eager")

    cmd = docker_ctl.with_flags(
        CMD,
        values={"--max-model-len": "2048", "--stage-overrides": '{"0": {"gpu_memory_utilization": 0.13}}'},
        switches={"--enforce-eager": True},
    )
    assert docker_ctl.flag_value(cmd, "--max-model-len") == "2048"
    assert docker_ctl.flag_value(cmd, "--stage-overrides") == '{"0": {"gpu_memory_utilization": 0.13}}'
    assert docker_ctl.has_flag(cmd, "--enforce-eager")
    assert cmd[:3] == CMD[:3]  # the checkpoint and the deployment flags are intact

    # Replacing keeps a single copy; removing drops the flag *and* its value.
    again = docker_ctl.with_flags(cmd, values={"--max-model-len": "4096"})
    assert again.count("--max-model-len") == 1
    assert docker_ctl.flag_value(again, "--max-model-len") == "4096"
    stripped = docker_ctl.with_flags(again, values={"--max-model-len": None}, switches={"--enforce-eager": False})
    assert not docker_ctl.has_flag(stripped, "--max-model-len")
    assert "4096" not in stripped
    assert not docker_ctl.has_flag(stripped, "--enforce-eager")


def test_set_model_can_change_only_the_flags(monkeypatch):
    calls = []
    _daemon(monkeypatch, [{"Id": "abcdef1234567890", "Names": ["/tts-vllm"],
                           "_inspect": _inspect(CMD)}], calls)
    asyncio.run(
        docker_ctl.set_model(
            "/var/run/docker.sock", "tts-vllm", None,
            values={"--stage-overrides": '{"0": {"gpu_memory_utilization": 0.3}}'},
            switches={"--enforce-eager": True},
        )
    )
    created = [c for c in calls if c[0] == "CREATE"][0][1]
    assert created[2] == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"  # same checkpoint
    assert "--stage-overrides" in created and "--enforce-eager" in created


def test_set_model_does_nothing_when_the_profile_already_matches(monkeypatch):
    calls = []
    _daemon(monkeypatch, [{"Id": "abcdef1234567890", "Names": ["/tts-vllm"],
                           "_inspect": _inspect(CMD)}], calls)
    asyncio.run(
        docker_ctl.set_model("/var/run/docker.sock", "tts-vllm", None,
                             values={"--max-model-len": None})  # already absent
    )
    assert "/containers/create" not in [p for _, p in calls]


def test_compose_prefixed_names_are_resolved(monkeypatch):
    """Compose names containers <project>-<service>-<index>."""
    calls = []
    containers = [
        {"Id": "aaaa11112222", "Names": ["/aiudio-tts-vllm-previous"],
         "State": "exited",
         "Labels": {"com.docker.compose.project": "aiudio", "com.docker.compose.service": "tts-vllm"},
         "_inspect": _inspect(CMD, running=False)},
        {"Id": "bbbb33334444", "Names": ["/aiudio-tts-vllm-1"],
         "State": "running",
         "Labels": {"com.docker.compose.project": "aiudio", "com.docker.compose.service": "tts-vllm"},
         "_inspect": _inspect(CMD)},
    ]
    _daemon_entries(monkeypatch, containers, calls)

    state = asyncio.run(docker_ctl.status("/var/run/docker.sock", "tts-vllm"))
    assert state["running"] is True
    # The live container wins over our own `<name>-previous` leftover.
    assert ("GET", "/containers/bbbb33334444/json") in calls

    # Stopping it works too (that was the reported bug).
    asyncio.run(docker_ctl.stop("/var/run/docker.sock", "tts-vllm", timeout=5))
    assert ("POST", "/containers/bbbb33334444/stop") in calls


def test_a_missing_container_lists_what_the_daemon_sees(monkeypatch):
    calls = []
    _daemon_entries(monkeypatch, [{"Id": "bbbb33334444", "Names": ["/aiudio-other-1"],
                                   "State": "running", "Labels": {}}], calls)

    import pytest

    with pytest.raises(docker_ctl.DockerError) as excinfo:
        asyncio.run(docker_ctl.start("/var/run/docker.sock", "tts-vllm"))
    message = str(excinfo.value)
    assert "docker.container" in message
    assert "aiudio-other-1" in message  # what to fix the setting to


def test_set_model_recreates_when_start_fails_on_a_missing_network(monkeypatch):
    """A stale network id must not leave the service unstartable.

    Docker records the network id in some cases; when the compose network is
    recreated the id is gone and `/start` answers 404 (`network <id> not found`).
    The gateway then recreates the container, re-attaching it by network name.
    """
    calls = []
    stale = "50ce6248282c5d22f952a54d0af6574709198a4ad6c4c20e7014d86a698a128a"
    inspect = _inspect(CMD, running=False)
    inspect["HostConfig"]["NetworkMode"] = stale
    inspect["NetworkSettings"]["Networks"] = {"aiudio-net": {"Aliases": ["tts-vllm"]}}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append((request.method, path))
        if path == "/containers/json":
            listing = [{"Id": "abcdef1234567890", "Names": ["/tts-vllm"]}]
            return httpx.Response(200, json=listing)
        if path.endswith("/json"):
            return httpx.Response(200, json=inspect)
        if path == "/containers/abcdef1234567890/start":
            return httpx.Response(
                404, json={"message": f"failed to set up container networking: network {stale} not found"}
            )
        if path == "/containers/create":
            body = json.loads(request.content)
            calls.append(("CREATE_BODY", body))
            return httpx.Response(201, json={"Id": "new"})
        if path == "/containers/new/start":
            inspect["State"] = {"Running": True, "Status": "running"}
            return httpx.Response(204)
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        docker_ctl, "client",
        lambda socket, timeout=30.0: httpx.AsyncClient(transport=transport, base_url="http://docker"),
    )

    state = asyncio.run(docker_ctl.set_model("/var/run/docker.sock", "tts-vllm"))

    body = [c[1] for c in calls if c[0] == "CREATE_BODY"][0]
    assert body["HostConfig"]["NetworkMode"] == "aiudio-net"  # the name, not the dead id
    assert body["NetworkingConfig"]["EndpointsConfig"]["aiudio-net"]["Aliases"] == ["tts-vllm"]
    assert ("POST", "/containers/new/start") in calls
    assert state["running"] is True
