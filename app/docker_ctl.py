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
import re
import time
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger(__name__)


class DockerError(RuntimeError):
    """Raised when the daemon refuses a call (socket missing, no permission…)."""


def missing_network(exc: BaseException) -> bool:
    """Does this error mean the container references a network that is gone?"""
    message = str(exc).lower()
    return "network" in message and "not found" in message


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


def flag_value(cmd: Optional[List[str]], flag: str) -> Optional[str]:
    """Value of ``--flag value`` in a command line (None when absent)."""
    args = list(cmd or [])
    if flag not in args:
        return None
    index = args.index(flag) + 1
    if index >= len(args) or args[index].startswith("--"):
        return ""  # present but it is a switch
    return args[index]


def has_flag(cmd: Optional[List[str]], flag: str) -> bool:
    return flag in list(cmd or [])


def with_flags(
    cmd: Optional[List[str]],
    *,
    values: Optional[Dict[str, Optional[str]]] = None,
    switches: Optional[Dict[str, bool]] = None,
) -> List[str]:
    """Set/replace/remove ``--flag value`` and switch arguments in place.

    Only the named flags move: everything else (the checkpoint, the deployment
    flags, their order) is preserved. A value of ``None`` in ``values`` removes the
    flag *and* its value; a switch set to False removes it.
    """
    args = list(cmd or [])
    for flag, value in (values or {}).items():
        if flag in args:
            index = args.index(flag)
            has_value = index + 1 < len(args) and not args[index + 1].startswith("--")
            if value is None:
                del args[index : index + (2 if has_value else 1)]
            elif has_value:
                args[index + 1] = value
            else:
                args.insert(index + 1, value)
        elif value is not None:
            args += [flag, value]
    for flag, enabled in (switches or {}).items():
        if enabled and flag not in args:
            args.append(flag)
        elif not enabled and flag in args:
            del args[args.index(flag)]
    return args


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


def _rank(entry: Dict[str, Any], name: str) -> int:
    """How well a daemon listing entry matches the configured container name.

    Compose names containers ``<project>-<service>-<index>``, and a recreate leaves
    a ``<name>-previous`` behind: prefer the exact name, then the compose service
    label, then the project-prefixed name, then any name carrying it as a word.
    Lower is better; ``-1`` means "not it".
    """
    names = [str(n).lstrip("/") for n in (entry.get("Names") or [])]
    labels = entry.get("Labels") or {}
    score = -1
    if name in names:
        score = 0
    elif labels.get("com.docker.compose.service") == name:
        score = 1
    elif any(
        re.search(rf"(^|[-_/]){re.escape(name)}([-_/]|$)", candidate) for candidate in names
    ):
        score = 3
    elif name and name in str(entry.get("Id", ""))[:12]:
        score = 4
    if score < 0:
        return -1
    if any(candidate.endswith("-previous") for candidate in names):
        score += 10  # our own leftover from a recreate
    if (entry.get("State") or "").lower() == "running":
        score -= 1  # at equal specificity, the live one wins
    return score


async def find(socket: str, name: str) -> Optional[Dict[str, Any]]:
    """Inspect the configured container (None when it does not exist)."""
    filter_json = json.dumps({"name": [name]})
    containers = await _call(
        socket, "GET", "/containers/json", params={"all": 1, "filters": filter_json}
    ) or []
    ranked = [(rank, entry) for entry in containers if (rank := _rank(entry, name)) >= 0]
    if not ranked:
        return None
    best = min(ranked, key=lambda item: (item[0], item[1].get("Id", "")))[1]
    return await _call(socket, "GET", f"/containers/{best['Id']}/json")


async def candidates(socket: str, name: str) -> List[str]:
    """Names the daemon knows that could be the configured container (for errors)."""
    filter_json = json.dumps({"name": [name]})
    try:
        containers = await _call(
            socket, "GET", "/containers/json", params={"all": 1, "filters": filter_json}
        ) or []
    except DockerError:  # pragma: no cover - the caller is already failing
        return []
    return sorted(
        str(entry.get("Names", [""])[0]).lstrip("/") for entry in containers
    )


async def _container_id(socket: str, name: str) -> Dict[str, Any]:
    info = await find(socket, name)
    if info is None:
        seen = await candidates(socket, name)
        hint = f" — the daemon sees: {', '.join(seen)}" if seen else ""
        raise DockerError(
            f"container {name!r} not found{hint}. Set docker.container to the exact "
            "name (compose prefixes it with the project: <project>-<service>-1)."
        )
    return info


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


async def start(socket: str, name: str, timeout: float = 30.0) -> Dict[str, Any]:
    info = await _container_id(socket, name)
    if not (info.get("State") or {}).get("Running"):
        await _call(socket, "POST", f"/containers/{info['Id']}/start", timeout=timeout)
    return await status(socket, name)


#: Alias for :func:`start`: ``set_model`` has a boolean ``start`` parameter of its
#: own, which would shadow the function name inside its body.
_start_container = start


async def stop(socket: str, name: str, timeout: float = 60.0) -> Dict[str, Any]:
    info = await _container_id(socket, name)
    if (info.get("State") or {}).get("Running"):
        await _call(
            socket, "POST", f"/containers/{info['Id']}/stop",
            params={"t": int(timeout)}, timeout=timeout + 30.0,
        )
    return await status(socket, name)


async def restart(socket: str, name: str, timeout: float = 60.0) -> Dict[str, Any]:
    info = await _container_id(socket, name)
    await _call(
        socket, "POST", f"/containers/{info['Id']}/restart",
        params={"t": int(timeout)}, timeout=timeout + 60.0,
    )
    return await status(socket, name)


async def set_model(
    socket: str,
    name: str,
    model: Optional[str] = None,
    timeout: float = 300.0,
    *,
    values: Optional[Dict[str, Optional[str]]] = None,
    switches: Optional[Dict[str, bool]] = None,
    start: bool = True,
) -> Dict[str, Any]:
    """Recreate the container on another checkpoint and/or launch flags.

    Docker cannot change a container's command in place, so the current container
    is renamed out of the way, a new one is created from its inspected
    configuration (image, env, mounts, GPU reservations, network…) with the new
    ``vllm serve <model>`` and the given flags, started (unless ``start`` is false:
    the engine is local, the service is only prepared), and only then is the old one
    removed — a failure leaves the previous container available under
    ``<name>-previous``.
    """
    info = await _container_id(socket, name)
    config = dict(info.get("Config") or {})
    current = list(config.get("Cmd") or [])
    wanted = with_flags(
        with_model(current, model) if model else current, values=values, switches=switches
    )
    # Attach the new container to the same networks **with the same aliases**:
    # compose publishes the service name (`tts-vllm`) as an alias, and dropping it
    # makes `http://tts-vllm:8091` unresolvable from the gateway ([Errno -3]
    # Temporary failure in name resolution) even though the service is running.
    network_settings = (info.get("NetworkSettings") or {}).get("Networks") or {}
    names = [net for net in network_settings if net]
    if not names:
        # A container that never started can report no network at all: compose
        # still labels the network it belongs to.
        label = str((config.get("Labels") or {}).get("com.docker.compose.network") or "").strip()
        if label:
            names = [label]
    if wanted == current:
        # Nothing to change: just make sure it is in the wanted state.
        if not (info.get("State") or {}).get("Running"):
            if not start:
                return describe(info)
            try:
                return await _start_container(socket, name, timeout=timeout)
            except DockerError as exc:
                # A container can point at a network *id* that no longer exists (the
                # compose network was recreated with a new id): starting it then
                # fails with `network <id> not found`. Recreate it below instead,
                # which re-attaches it by network name — but only when we know that
                # name, or the replacement would be broken too.
                if not missing_network(exc):
                    raise
                if not names:
                    raise DockerError(
                        f"{exc} — the container's network is gone and its name is "
                        "unknown: recreate the stack (docker compose up -d --force-recreate)"
                    ) from exc
                log.warning(
                    "Container %s references a missing network (%s); recreating it",
                    name, exc,
                )
        else:
            return await stop(socket, name) if not start else describe(info)
    cmd = wanted

    host_config = dict(info.get("HostConfig") or {})
    for key in ("Name", "Created", "State", "NetworkSettings", "Mounts"):
        host_config.pop(key, None)
    endpoints: Dict[str, Any] = {}
    for net, spec in network_settings.items():
        endpoint: Dict[str, Any] = {}
        aliases = [a for a in ((spec or {}).get("Aliases") or []) if a]
        if aliases:
            endpoint["Aliases"] = aliases
        if (spec or {}).get("IPAMConfig"):
            endpoint["IPAMConfig"] = spec["IPAMConfig"]
        endpoints[net] = endpoint
    for net in names:  # e.g. recovered from the compose label
        endpoints.setdefault(net, {})
    # `NetworkMode` may hold a network *id* (it changes when the network is
    # recreated, and the start then fails with `network <id> not found`). Always
    # reference the stable network name we attach to.
    mode = str(host_config.get("NetworkMode") or "")
    if names and (not mode or re.fullmatch(r"[0-9a-f]{12,64}", mode) or mode not in names):
        host_config["NetworkMode"] = names[0]
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
    if endpoints:
        body["NetworkingConfig"] = {"EndpointsConfig": endpoints}

    # Compose names containers <project>-<service>-<index>: keep that identity, or
    # the next `docker compose up` would create a second one next to ours.
    service = str(info.get("Name") or "").lstrip("/") or name
    previous = f"{service}-previous"
    old_id, renamed = info["Id"], False
    if (info.get("State") or {}).get("Running"):
        await _call(socket, "POST", f"/containers/{old_id}/stop",
                    params={"t": 20}, timeout=120.0)
    try:
        await _call(socket, "POST", f"/containers/{old_id}/rename",
                    params={"name": previous})
        renamed = True
        created = await _call(socket, "POST", "/containers/create",
                              params={"name": service}, json=body)
    except DockerError:
        if renamed:  # put the old container back under its name
            await _call(socket, "POST", f"/containers/{old_id}/rename",
                        params={"name": service})
        raise
    new_id = (created or {}).get("Id")
    if start:
        try:
            await _call(socket, "POST", f"/containers/{new_id}/start", timeout=timeout)
        except DockerError:
            log.error("The new %s container failed to start; the previous one is %r",
                      service, previous)
            raise
    try:
        await _call(socket, "DELETE", f"/containers/{old_id}", params={"v": 1})
    except DockerError:  # pragma: no cover - cosmetic
        log.warning("Could not remove the previous container %s", previous)
    return await status(socket, name)


async def start_or_recreate(socket: str, name: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Start the container, recreating it if it references a missing network.

    The UI's Start button and the engine toggle go through here: a container that
    points at a network id that no longer exists (the compose network was recreated)
    fails `/start` with `network <id> not found` and would otherwise stay
    unstartable, while the gateway could rebuild it in place.
    """
    try:
        return await start(socket, name, timeout=timeout)
    except DockerError as exc:
        if not missing_network(exc):
            raise
        log.warning(
            "Container %s references a missing network (%s); recreating it to "
            "re-attach it by network name", name, exc,
        )
    try:
        return await set_model(socket, name, start=True, timeout=timeout)
    except DockerError as exc:
        if not missing_network(exc):
            raise
        raise DockerError(
            f"{exc} — the compose network is gone: recreate the stack "
            "(docker compose up -d --force-recreate)"
        ) from exc


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
