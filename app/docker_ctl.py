"""Pilot a side container (the vLLM-Omni service) through the Docker API.

Opt-in: the gateway needs the Docker socket mounted, which is root-equivalent
power over the host, so nothing happens unless ``docker.control`` is enabled and
only the container named by ``docker.container`` is ever touched. Everything is
best effort and read-through: the UI shows what the daemon answers.

The Docker HTTP API is spoken directly over the unix socket (``httpx`` supports
UDS transports), so no extra dependency is involved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger(__name__)


class DockerError(RuntimeError):
    """Raised when the daemon refuses a call (socket missing, no permission…)."""


def client(socket: str, timeout: float = 30.0) -> httpx.AsyncClient:
    """Async Docker API client bound to the unix socket."""
    transport = httpx.AsyncHTTPTransport(uds=socket)
    return httpx.AsyncClient(transport=transport, base_url="http://docker", timeout=timeout)


async def _call(
    socket: str, method: str, path: str, *, timeout: float = 30.0, **kwargs: Any
) -> Any:
    async with client(socket, timeout) as api:
        try:
            resp = await api.request(method, path, **kwargs)
        except (httpx.HTTPError, OSError) as exc:
            raise DockerError(f"Docker API unreachable ({socket}): {exc}") from exc
        if resp.status_code >= 400:
            detail = resp.text.strip()[:300]
            raise DockerError(f"{method} {path} → {resp.status_code} {detail}")
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:  # pragma: no cover - some endpoints answer plain text
            return resp.text


def served_model(cmd: Optional[List[str]]) -> Optional[str]:
    """Checkpoint argument of a ``vllm serve <model> …`` command line."""
    args = list(cmd or [])
    try:
        return args[args.index("serve") + 1]
    except (ValueError, IndexError):
        return None


def with_model(cmd: Optional[List[str]], model: str) -> List[str]:
    """Same command line with the ``vllm serve`` argument replaced.

    The flags of the deployment (``--omni``, ``--enforce-eager``,
    ``--stage-overrides``…) are preserved: only the checkpoint changes.
    """
    args = list(cmd or [])
    if "serve" not in args:
        raise DockerError("the container command is not `vllm serve <model> …`")
    index = args.index("serve") + 1
    if index >= len(args):
        raise DockerError("the container command has no model argument")
    args[index] = model
    return args


async def find(socket: str, name: str) -> Optional[Dict[str, Any]]:
    """Inspect the configured container (None when it does not exist)."""
    filter_json = json.dumps({"name": [name]})
    containers = await _call(
        socket, "GET", "/containers/json", params={"all": 1, "filters": filter_json}
    ) or []
    target: Optional[Dict[str, Any]] = None
    for entry in containers:
        names = [str(n).lstrip("/") for n in (entry.get("Names") or [])]
        if name in names:  # the daemon filters by substring: require the exact name
            target = entry
            break
        if name and name in str(entry.get("Id", ""))[:12]:
            target = target or entry
    if target is None:
        return None
    return await _call(socket, "GET", f"/containers/{target['Id']}/json")


def describe(info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact, JSON-friendly state of the service container."""
    if info is None:
        return {"exists": False, "running": False, "model": None}
    state = info.get("State") or {}
    cmd = (info.get("Config") or {}).get("Cmd")
    return {
        "exists": True,
        "id": str(info.get("Id", ""))[:12],
        "name": str(info.get("Name", "")).lstrip("/"),
        "running": bool(state.get("Running")),
        "status": state.get("Status"),
        "started_at": state.get("StartedAt"),
        "image": (info.get("Config") or {}).get("Image"),
        "model": served_model(cmd),
        "command": cmd,
    }


async def status(socket: str, name: str) -> Dict[str, Any]:
    return describe(await find(socket, name))


async def start(socket: str, name: str) -> Dict[str, Any]:
    info = await find(socket, name)
    if info is None:
        raise DockerError(f"container {name!r} not found")
    if not (info.get("State") or {}).get("Running"):
        await _call(socket, "POST", f"/containers/{info['Id']}/start")
    return await status(socket, name)


async def stop(socket: str, name: str, timeout: float = 60.0) -> Dict[str, Any]:
    info = await find(socket, name)
    if info is None:
        raise DockerError(f"container {name!r} not found")
    if (info.get("State") or {}).get("Running"):
        await _call(
            socket, "POST", f"/containers/{info['Id']}/stop",
            params={"t": int(timeout)}, timeout=timeout + 30.0,
        )
    return await status(socket, name)


async def restart(socket: str, name: str, timeout: float = 60.0) -> Dict[str, Any]:
    info = await find(socket, name)
    if info is None:
        raise DockerError(f"container {name!r} not found")
    await _call(
        socket, "POST", f"/containers/{info['Id']}/restart",
        params={"t": int(timeout)}, timeout=timeout + 60.0,
    )
    return await status(socket, name)


async def set_model(socket: str, name: str, model: str, timeout: float = 300.0) -> Dict[str, Any]:
    """Recreate the container serving ``model``, keeping its configuration.

    Docker cannot change a container's command in place, so the current
    container is renamed out of the way, a new one is created from its inspected
    configuration (image, env, mounts, GPU reservations, network…) with the new
    checkpoint, started, and only then is the old one removed — a failure leaves
    the previous container available under ``<name>-previous``.
    """
    info = await find(socket, name)
    if info is None:
        raise DockerError(f"container {name!r} not found")
    config = dict(info.get("Config") or {})
    cmd = with_model(config.get("Cmd"), model)
    if served_model(config.get("Cmd")) == model:
        # Already that checkpoint: just make sure it runs.
        if (info.get("State") or {}).get("Running"):
            return describe(info)
        return await start(socket, name)

    host_config = dict(info.get("HostConfig") or {})
    for key in ("Name", "Created", "State", "NetworkSettings", "Mounts"):
        host_config.pop(key, None)
    networks = list(((info.get("NetworkSettings") or {}).get("Networks") or {}).keys())
    body: Dict[str, Any] = {
        "Image": config.get("Image"),
        "Cmd": cmd,
        "Entrypoint": config.get("Entrypoint"),
        "Env": config.get("Env"),
        "Labels": config.get("Labels"),
        "WorkingDir": config.get("WorkingDir"),
        "User": config.get("User"),
        "HostConfig": host_config,
    }
    if networks:
        body["NetworkingConfig"] = {"EndpointsConfig": {net: {} for net in networks}}

    previous = f"{name}-previous"
    old_id, renamed = info["Id"], False
    if (info.get("State") or {}).get("Running"):
        await _call(socket, "POST", f"/containers/{old_id}/stop",
                    params={"t": 20}, timeout=120.0)
    try:
        await _call(socket, "POST", f"/containers/{old_id}/rename",
                    params={"name": previous})
        renamed = True
        created = await _call(socket, "POST", "/containers/create",
                              params={"name": name}, json=body)
    except DockerError:
        if renamed:  # put the old container back under its name
            await _call(socket, "POST", f"/containers/{old_id}/rename",
                        params={"name": name})
        raise
    new_id = (created or {}).get("Id")
    try:
        await _call(socket, "POST", f"/containers/{new_id}/start", timeout=timeout)
    except DockerError:
        log.error("The new %s container failed to start; the previous one is %r",
                  name, previous)
        raise
    try:
        await _call(socket, "DELETE", f"/containers/{old_id}", params={"v": 1})
    except DockerError:  # pragma: no cover - cosmetic
        log.warning("Could not remove the previous container %s", previous)
    return await status(socket, name)


async def wait_ready(url: str, timeout: float = 300.0, interval: float = 3.0) -> bool:
    """Poll ``{url}/health`` until the service answers (vLLM needs ~1 min)."""
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=5.0) as api:
        while time.monotonic() < deadline:
            try:
                resp = await api.get(f"{url.rstrip('/')}/health")
                if resp.status_code < 400:
                    return True
            except (httpx.HTTPError, OSError):
                pass
            await asyncio.sleep(interval)
    return False
