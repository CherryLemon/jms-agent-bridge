"""The MCP server: the tool surface the agent talks to.

Runs as a local stdio process launched by Claude Code, Codex or dsh. Tools are grouped by plane:

* control plane (:mod:`jms_bridge.api`) — ``jms_list_assets``, ``jms_audit_query``,
  ``jms_create_ticket``
* data plane (:mod:`jms_bridge.ssh`, :mod:`jms_bridge.jobs`) — ``jms_probe``,
  ``jms_fleet_summary``, ``jms_exec``, ``jms_sync_push`` / ``jms_sync_pull``, and the
  job lifecycle

State (config, inventory, the SSH runner, the API client) is built lazily on first
use so that ``import`` never fails just because the inventory is not written yet.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
import subprocess
from collections.abc import Callable
from typing import Any, TypedDict, TypeVar

import yaml
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import __version__
from .api import JumpServerAPI, JumpServerError
from .cluster import (
    DEFAULT_RDZV_PORT,
    is_heterogeneous,
    plan_islands,
    preflight,
    resolve_rendezvous,
)
from .config import BastionConfig, ConfigError, load_config
from .discover import DEFAULT_REPO_MATCH
from .inventory import Host, Inventory, InventoryError, shell_quote
from .jobs import JobManager, validate_job_id
from .ssh import (
    OUTPUT_LIMIT,
    REACH_PROBE,
    REACH_SENTINEL,
    ExecResult,
    SSHRunner,
    failure_reason,
    reached,
)
from .workspace import CONTAINER_RE, DEFAULT_CONTAINER, SOURCES

log = logging.getLogger("jms_bridge.server")


# --- structured output: the schemas a client sees ----------------------------------
#
# A tool's published `outputSchema` is derived from its return *annotation*, so these
# TypedDicts exist to give the machine-readable tools a shape. They are deliberately
# partial (``total=False``): the tools return a union of shapes — a success payload, a
# refusal carrying `error`, or a guard return — and the schema's job is to make the
# *predictable* fields known to a client, not to reject the rest. Pydantic drops any key
# the schema does not mention, so a TypedDict that omits a field silently hides it from a
# structured-output consumer; every key any branch of these tools can return must be
# listed here.


class _Refusal(TypedDict, total=False):
    """The shape every tool falls back to when it refuses or matches no host."""

    ok: bool
    error: str


class JobStatusOut(_Refusal, total=False):
    """``jms_job_status``: one status dict per host, plus the per-host `ok`."""

    status: list[dict]


class ClusterPreflightOut(_Refusal, total=False):
    """``jms_cluster_preflight``: readiness, world size and the busy gate."""

    world_size_estimate: int
    rendezvous: dict
    heterogeneous: bool
    islands: list[dict]
    hosts: list[dict]
    platforms: list[str]
    busy_hosts: list[str]
    note: str


class DevRunOut(_Refusal, total=False):
    """``jms_dev_run``: the cross-chip consistency table and its verification gates."""

    ref: str
    stage: str
    columns: list[str]
    rows: list[dict]
    failed_hosts: list[str]
    revisions: dict
    revisions_consistent: bool
    revisions_unverified: list[str]
    attestation: dict
    attestation_consistent: bool
    attestation_diverged: list[str]
    attestation_unverified: list[str]
    hosts: dict
    results: list[dict]


class ClusterSubmitOut(_Refusal, total=False):
    """``jms_cluster_submit``: the rendezvous, per-rank wiring and job id."""

    command: str
    rendezvous: dict
    world_size: int
    nnodes: int
    # None on a refusal that never reached the launch (e.g. an empty selection).
    job_id: str | None
    ranks: list[dict]
    islands: list[dict]
    hint: str
    unready: list[dict]
    ready_targets: list[str]
    busy: list[dict]


# The version is what a client shows in `/mcp` and what a bug report quotes back, so it
# has to be the installed package's, not the SDK default (an empty string).
#
# `instructions` is injected into the agent's context by most clients, and this surface is
# wide with a non-obvious selector grammar — spelling the mental model out once here is
# cheaper (and less error-prone) than hoping the agent infers it from the tool docstrings.
# Keep it to what an agent must know *before* choosing a tool.
_INSTRUCTIONS = """\
JumpServer bridge for a heterogeneous accelerator fleet (Ascend/NVIDIA/domestic NPUs).

Selector grammar: every `targets` / `workspace` argument takes host or workspace *names*,
*group* names, or *tags*; `all` (or `*`) selects everything. An unmatched selector is an
error, and an empty selection is never reported as success.

Choosing a tool:
- Long work (training) must go through `jms_submit_job` (same command on each host) or
  `jms_cluster_submit` (one coordinated collective across hosts) — never `jms_exec`, which
  is for short inspection commands and is bounded by its own timeout.
- Before a distributed run: `jms_cluster_preflight` (readiness, world size, busy cards),
  then `jms_cluster_plan` if the fleet is mixed-vendor — HCCL/NCCL islands cannot form one
  collective and must federate.
- Jobs are detached and answered by id: `jms_job_status` / `jms_job_logs` (poll with the
  returned `next_offset`) / `jms_job_wait` (bounded; a timeout cancels nothing) /
  `jms_cluster_cancel`.
- After editing `hosts.yaml` or logging in, call `jms_inventory_reload` — the server caches
  the inventory and credentials until you do.

`jms_exec` / `jms_submit_job` / `jms_cluster_submit` / `jms_dev_run` refuse a command that
matches a small destructive-pattern denylist unless `allow_dangerous=true`. That is a guard
against obvious mistakes, not a security boundary — JumpServer's own policy is the backstop.
"""

server = MCPServer(
    "jms-agent-bridge",
    version=__version__,
    title="JumpServer Bridge",
    instructions=_INSTRUCTIONS,
)

_F = TypeVar("_F", bound=Callable[..., Any])


def _tool(
    *,
    read_only: bool = False,
    destructive: bool = False,
    idempotent: bool = False,
) -> Callable[[_F], _F]:
    """``@server.tool()``, with setup failures turned into a readable ``ToolError``.

    ``ConfigError`` (no bastion URL, unreadable config) and ``InventoryError``
    (missing/malformed ``hosts.yaml``, or a selector that matched nothing) are raised
    from the lazy state helpers and from ``resolve()`` inside almost every tool. The
    SDK forwards the *text* of a ``ToolError`` to the client; anything else reaches the
    agent as the opaque ``Error executing tool <name>`` — which is exactly, and
    uselessly, what a freshly-cloned-but-unconfigured checkout would see. Translating
    keeps the actionable message ("No inventory at …; run `jms-bridge init` or set
    JMS_BRIDGE_INVENTORY") in front of the agent.

    The three flags become MCP ``ToolAnnotations``, which is how a client decides whether
    to prompt the user. ``read_only`` is the useful one: a client may auto-approve a
    read-only tool, so a session that only inspects the fleet stops asking. It must be
    set honestly — a tool that changes remote state must not claim it.
    """

    def decorate(fn: _F) -> _F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except (ConfigError, InventoryError) as exc:
                raise ToolError(str(exc)) from exc

        annotations = ToolAnnotations(
            readOnlyHint=read_only,
            destructiveHint=destructive,
            idempotentHint=idempotent,
            openWorldHint=True,  # every tool talks to a remote fleet
        )
        return server.tool(annotations=annotations)(wrapper)  # type: ignore[return-value]

    return decorate


async def _progress(ctx: Context | None, done: float, total: float, message: str) -> None:
    """Best-effort MCP progress notification for a long tool.

    A long fan-out (`jms_dev_run` across a fleet) otherwise looks indistinguishable from a
    hang: the client shows a spinner and the user cannot tell "still working" from "dead".
    Reporting each completed host turns that into visible movement.

    Progress is **client-opt-in**: MCP delivers it only when the client registers a progress
    callback (which is what puts a progress token on the request). A client that does not
    simply receives nothing, which is why this is a nicety rather than a contract.

    Swallowed on failure by design. ``Context.report_progress`` raises when there is no live
    request (a direct call from a test or an embedder), and a client that does not implement
    progress is not an error — neither is worth failing the work over, so progress is
    advisory in exactly the way the work is not.
    """
    if ctx is None:
        return
    try:
        await ctx.report_progress(done, total, message)
    except Exception:  # noqa: BLE001 - progress must never break the operation
        log.debug("progress report failed", exc_info=True)


# --- lazily-initialised singletons ------------------------------------------------

_state: dict[str, Any] = {}


def _cfg() -> BastionConfig:
    if "cfg" not in _state:
        _state["cfg"] = load_config()
    return _state["cfg"]


def _inv() -> Inventory:
    if "inv" not in _state:
        _state["inv"] = Inventory.load()
    return _state["inv"]


def _ssh() -> SSHRunner:
    if "ssh" not in _state:
        cfg = _cfg()
        _state["ssh"] = SSHRunner(cfg, max_parallel=cfg.max_parallel)
    return _state["ssh"]


def _jobs() -> JobManager:
    if "jobs" not in _state:
        _state["jobs"] = JobManager(_ssh())
    return _state["jobs"]


def _api() -> JumpServerAPI:
    if "api" not in _state:
        _state["api"] = JumpServerAPI(_cfg())
    return _state["api"]


def _resolve(selectors: list[str]) -> list[Host]:
    return _inv().resolve(selectors)


def _empty_refusal(selectors: list[str], **empty: Any) -> dict[str, Any]:
    """The reply for a fan-out whose selectors matched no host.

    ``resolve([])`` returns an empty list *without* raising (only an *unmatched* selector
    raises), and ``all()`` over nothing is ``True`` — so a fan-out that reached no host
    would otherwise come back green, and for the job tools that green would be a claim
    about hosts nobody asked ("every rank is terminal"). Callers pass their tool's usual
    keys with their empty value (``hosts=[]``) so a client reading the reply finds the
    shape it expects instead of a missing key.
    """
    return {
        "ok": False,
        "error": f"selectors {selectors!r} matched no host; nothing was done",
        **empty,
    }


# --- control plane: discovery / audit / tickets -----------------------------------


@_tool(read_only=True)
async def jms_list_assets(search: str | None = None) -> dict:
    """List the assets this account can reach in JumpServer.

    Use this to discover the ``asset_id`` values the inventory needs. Returns each
    asset's id, name, address and platform. Bound accounts are *not* included — this
    call is one REST request, and enumerating the ``#account`` part of every login
    would be one request per asset. For one asset's accounts (the ``#account`` in a
    login), call `jms_asset_detail`.
    """
    api = _api()
    try:
        assets, truncated = await api.list_assets(search=search)
    except JumpServerError as exc:
        return {"ok": False, "error": str(exc)}
    out = []
    for a in assets:
        out.append(
            {
                "id": a.get("id"),
                "name": a.get("name"),
                "address": a.get("address") or a.get("ip"),
                "platform": (a.get("platform") or {}).get("name")
                if isinstance(a.get("platform"), dict)
                else a.get("platform"),
            }
        )
    result: dict = {"ok": True, "count": len(out), "assets": out, "truncated": truncated}
    if truncated:
        # Never let a prefix read as the whole fleet. A *warning*, not an `error`: the call
        # succeeded, and `error` is the key this surface reserves for a refusal — pairing it
        # with `ok: true` would be self-contradictory and could trip a client that treats
        # any `error` as failure.
        result["warning"] = (
            f"listing stopped after {len(out)} assets while the server still offered more: "
            "this is a prefix, not the full inventory. Narrow it with `search`."
        )
    return result


@_tool(read_only=True)
async def jms_asset_detail(asset_id: str, host_name: str | None = None) -> dict:
    """One asset's bound accounts and a paste-ready `hosts.yaml` entry.

    `jms_list_assets` gives ids, names and addresses, but *not* the ``account`` half of
    the per-asset login ``<login>#<account>#<asset_id>`` — which is exactly the field a
    new inventory is missing, and the one a user otherwise has to hunt for in the
    JumpServer UI. The asset's own detail endpoint carries it, so this is still **one**
    REST request.

    Use the account this returns (``accounts[].username``, preferring an entry whose
    ``secret_type`` is set) as ``account:`` in the host entry. `inventory_yaml` is the
    block to paste under ``hosts:``, and its ``platform`` is a **placeholder on purpose**:
    the accelerator tag cannot be inferred from JumpServer's platform name, and a wrong
    guess probes with the wrong tool and reports a healthy host with no accelerator (see
    `jms_dev_discover` / the platform list in the README).

    Prefer this over a per-asset account *list* endpoint, which in practice is not
    exposed to a normal account (it answers 403); the detail call is.
    """
    api = _api()
    try:
        asset = await api.get_asset(asset_id)
    except JumpServerError as exc:
        return {"ok": False, "error": str(exc), "asset_id": asset_id}

    accounts = []
    for acc in asset.get("permed_accounts") or []:
        accounts.append(
            {
                "username": acc.get("username"),
                "name": acc.get("name"),
                "id": acc.get("id"),
                # A `secret_type` means the account has a credential JumpServer can use.
                "secret_type": acc.get("secret_type"),
                "has_secret": acc.get("has_secret"),
            }
        )
    platform = asset.get("platform")
    name = host_name or asset.get("name") or asset_id
    # Prefer an account that actually has a secret; that is the one that can log in.
    usable = [a for a in accounts if a.get("has_secret")] or accounts
    account = usable[0]["username"] if usable else None
    entry = {
        "login": "<your JumpServer login>",
        "account": account or "<account>",
        "asset_id": asset_id,
        "platform": "<accelerator profile, e.g. ascend-910c | cuda | kunlunxin>",
        # Placeholders are wrapped in <> so a half-edited paste is obvious; a bare
        # `~/your-workdir` reads like a real path and would ship as one.
        "env_setup": "<source /path/to/env.sh>",
        "workdir": "<~/your-workdir>",
        "tags": ["<tag>"],
    }
    return {
        "ok": True,
        "asset": {
            "id": asset.get("id"),
            "name": asset.get("name"),
            "address": asset.get("address"),
            "platform": platform.get("name") if isinstance(platform, dict) else platform,
            "active": asset.get("is_active"),
            "connectivity": (asset.get("connectivity") or {}).get("value")
            if isinstance(asset.get("connectivity"), dict)
            else asset.get("connectivity"),
        },
        "accounts": accounts,
        # One representation, not two: the YAML *is* the paste-ready form, and shipping the
        # same content as a dict as well would double the payload for no gain (the mistake
        # `jms_job_logs` was just fixed for). Callers that want a dict can parse this.
        "inventory_yaml": yaml.safe_dump({name: entry}, sort_keys=False),
    }


@_tool(read_only=True)
async def jms_audit_query(
    asset_id: str | None = None, user: str | None = None, days: int = 5, limit: int = 30
) -> dict:
    """Recent command records from JumpServer's audit log.

    This is how the agent (and you) answer 'what has actually been executed?' —
    every data-plane command is attributed here because it goes through KoKo.
    `days` is the look-back window (JumpServer defaults to the last 5 days).
    """
    api = _api()
    try:
        records = await api.command_records(
            asset_id=asset_id, user=user, days=days, limit=limit
        )
    except JumpServerError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "count": len(records),
        "records": [
            {
                "user": r.get("user"),
                "asset": r.get("asset"),
                "account": r.get("account"),
                "input": r.get("input"),
                "timestamp": r.get("timestamp"),
                "session": r.get("session"),
            }
            for r in records
        ],
    }


@_tool()
async def jms_create_ticket(
    title: str,
    ticket_type: str = "login_asset_confirm",
    comment: str = "",
    org_id: str | None = None,
    fields: dict | None = None,
) -> dict:
    """Raise a JumpServer approval ticket via the subtype's ``open`` action.

    `ticket_type` is one of: `login_asset_confirm`, `command_confirm`, `apply_asset`,
    `login_confirm`. `fields` carries the type-specific payload, e.g. for a
    login-asset ticket ``{"apply_login_user": "<uuid>", "apply_login_asset":
    "<uuid>", "apply_login_account": "root"}``.
    """
    api = _api()
    # `fields` is spread into the payload alongside the named arguments, so a key that
    # repeats one (`title`, `comment`, `org_id`, `type`) is a call that cannot be built.
    # Left to Python it surfaces as a bare `TypeError` from inside the API method, which
    # the SDK shows the agent as the opaque "Error executing tool jms_create_ticket".
    collisions = sorted(set(fields or {}) & {"title", "comment", "org_id", "type"})
    if collisions:
        return {
            "ok": False,
            "error": (
                f"fields may not repeat the named arguments {collisions}; "
                "pass them as top-level tool arguments instead"
            ),
        }
    try:
        return {
            "ok": True,
            "ticket": await api.create_ticket(
                title=title,
                ticket_type=ticket_type,
                comment=comment,
                org_id=org_id,
                **(fields or {}),
            ),
        }
    except JumpServerError as exc:
        return {"ok": False, "error": str(exc)}


# --- data plane: probe / exec -----------------------------------------------------


def _per_host_output_budget(host_count: int) -> int:
    """The stdout/stderr budget for one host in a fan-out of ``host_count``.

    A per-host cap alone does not bound the reply: 40 hosts at the 16 KB cap is still
    ~650 KB (≈160 k tokens), and a 90-host fleet is worse. So a fan-out shares one
    aggregate budget (see :data:`_AGGREGATE_OUTPUT_BUDGET`) split equally across hosts,
    with a floor so a wide fleet still shows each host a useful preview rather than a few
    bytes. Every clipped host reports its true `*_bytes`, so the caller can always tell it
    is looking at a slice.
    """
    if host_count <= 0:
        return OUTPUT_LIMIT
    # Never *more* than one host's own cap, even for a single-host fan-out: the aggregate
    # budget is a fleet-level bound, not a licence for one host to emit 400 KB.
    return min(OUTPUT_LIMIT, max(_MIN_PER_HOST, _AGGREGATE_OUTPUT_BUDGET // host_count))


#: Total characters the `stdout`/`stderr` previews of one fan-out reply may spend across
#: all hosts — roughly 100 k tokens, large enough to diagnose a fleet and small enough
#: that a single call cannot evict a client's context.
_AGGREGATE_OUTPUT_BUDGET = 400_000

#: Floor for one host's slice, so a 90-host fan-out still shows each host something.
_MIN_PER_HOST = 2_000


def _fanout_reply(results: list[ExecResult]) -> list[dict]:
    """Per-host reply entries, sharing one aggregate output budget across the fleet."""
    budget = _per_host_output_budget(len(results))
    return [r.to_dict(limit=budget) for r in results]


@_tool(read_only=True)
async def jms_probe(targets: list[str]) -> dict:
    """Probe hosts for liveness, accelerator type and driver/runtime version.

    ``targets`` are inventory selectors: host names, group names, or tags. Runs the
    platform-appropriate command (``npu-smi`` / ``nvidia-smi`` / ``uname``) on every
    host in parallel and parses the result.

    `ok` means the host was actually *reached*, which is stricter than "the ssh command
    exited 0": KoKo rejects a connection (no permission, asset offline, bad address) with
    exit 0 and the error on stdout, so a status-only check reports a dead asset green.
    The probe command is prefixed with the ``jms-bridge-ok`` sentinel and the reply must
    carry it back.
    """
    hosts = _resolve(targets)
    if not hosts:
        return _empty_refusal(targets, hosts=[])

    # A per-host raise becomes that host's entry, so one unreachable node does not erase
    # the probe reply for the other 39 (the same isolation `_per_host` gives the job
    # tools). The probe itself is a fan-out — sequential probing made a wide selector cost
    # one round trip per host.
    async def one(host: Host) -> dict:
        platform = host.platform_def
        # The sentinel rides the same round trip as the probe: whatever the platform
        # command prints, a reply without the sentinel means the far side never ran it.
        cmd = f"{REACH_PROBE} && {{ {platform.probe_cmd}; }}"
        try:
            res = await _ssh().exec_on(host, cmd, timeout=45, workdir=False)
        except Exception as exc:  # noqa: BLE001 - isolating one host is the point
            return {
                "host": host.name,
                "platform": platform.name,
                "ok": False,
                "reached": False,
                "exit_code": None,
                "error": f"{type(exc).__name__}: {exc}"[:400],
            }
        served = reached(res)
        # Strip the sentinel line so `raw` (and any parser fed by it) sees only the
        # platform command's own output.
        body = res.stdout.replace(f"{REACH_SENTINEL}\n", "", 1)
        parsed = platform.parse_probe(body) if served else {}
        # Report *why* a host did not answer. A KoKo rejection puts its explanation on
        # stdout ("必须是唯一的账号 root"), so taking stderr only when `res.ok` is false
        # would leave the caller with a bare `reached: false` and nothing to act on.
        detail = ""
        if not (served and res.ok):
            detail = failure_reason(res)
        return {
            "host": host.name,
            "platform": platform.name,
            "ok": served and res.ok,
            "reached": served,
            "exit_code": res.exit_code,
            **parsed,
            # Named `error`, not `stderr`: a KoKo rejection puts its reason on stdout, so
            # the field that explains a failure does not always come from stderr.
            "error": detail,
        }

    results = list(await asyncio.gather(*(one(h) for h in hosts)))
    return {"ok": all(r["ok"] for r in results), "hosts": results}


def _fleet_rollup(rows: list[dict]) -> dict:
    """Aggregate per-host reachability/device rows into per-platform totals.

    Pure, so the counting rules are unit-tested without SSH: an unreachable host is
    counted in ``hosts`` but not in ``reachable``/``devices``; a reachable host whose
    device count is unknown contributes 0 devices (never guessed).
    """
    by: dict[str, dict[str, int]] = {}
    unreachable = []
    for r in rows:
        b = by.setdefault(r["platform"], {"hosts": 0, "reachable": 0, "devices": 0})
        b["hosts"] += 1
        if r["reachable"]:
            b["reachable"] += 1
            if r["device_count"] is not None:
                b["devices"] += r["device_count"]
        else:
            unreachable.append({"host": r["host"], "reason": r["reason"]})
    return {"by_platform": by, "unreachable": unreachable}


@_tool(read_only=True)
async def jms_fleet_summary(targets: list[str], probe: bool = True) -> dict:
    """Roll up the selected fleet: how many hosts are reachable and how many
    accelerators they hold, grouped by platform.

    ``targets`` are inventory selectors (names, groups, tags); pass ``all`` to cover the
    whole inventory. Reachability is proven by the ``jms-bridge-ok``
    sentinel coming back — a KoKo rejection (no permission, asset offline, bad address)
    exits **0** with the error on stdout, so trusting the exit code would count a dead
    asset as green. With ``probe`` on, each reachable host is also asked for its device
    count, so the total is the accelerators actually present rather than the assets
    JumpServer lists. A host that reports no device count is reachable but contributes
    0 devices, and is named under ``device_count_unknown`` rather than assumed.
    """
    hosts = _resolve(targets)
    if not hosts:
        # Same trap as the other fan-outs: `all()` over nothing is True, so an empty
        # selection would roll up as a reachable, device-holding fleet of nobody.
        return _empty_refusal(
            targets,
            hosts_total=0,
            hosts_reachable=0,
            devices_total=0,
            by_platform={},
            device_count_unknown=[],
            unreachable=[],
        )
    ssh = _ssh()

    async def one(host: Host) -> dict:
        platform = host.platform_def
        try:
            res = await ssh.exec_on(host, f"{REACH_PROBE} && hostname", timeout=45)
            is_up = reached(res)
            reason = "" if is_up else failure_reason(res)[:200]
        except Exception as exc:  # noqa: BLE001 - a transport failure is just unreachable
            is_up, reason = False, f"{type(exc).__name__}: {exc}"[:200]
        count = None
        if is_up and probe:
            pr = await ssh.exec_on(host, platform.probe_cmd, timeout=45, workdir=False)
            count = platform.parse_probe(pr.stdout).get("device_count")
        return {
            "host": host.name,
            "platform": platform.name,
            "reachable": is_up,
            "device_count": count,
            "reason": reason,
        }

    rows = list(await asyncio.gather(*(one(h) for h in hosts)))
    roll = _fleet_rollup(rows)
    return {
        # `rows` cannot be empty here — the guard above returns before any SSH is
        # attempted — so this is a plain "did every host answer", and a partial fleet
        # is deliberately `ok: false` with the detail under `unreachable`.
        "ok": all(r["reachable"] for r in rows),
        "hosts_total": len(rows),
        "hosts_reachable": sum(1 for r in rows if r["reachable"]),
        "devices_total": sum(v["devices"] for v in roll["by_platform"].values()),
        "by_platform": roll["by_platform"],
        "device_count_unknown": [
            r["host"] for r in rows if r["reachable"] and r["device_count"] is None
        ],
        "unreachable": roll["unreachable"],
    }


@_tool(destructive=True)
async def jms_exec(
    targets: list[str], command: str, timeout: float = 60.0, allow_dangerous: bool = False
) -> dict:
    """Run a SHORT command on one or many hosts, in parallel, and return its output.

    Use this for inspection (`nvidia-smi`, `ls`, `git status`, `df -h`), not for
    training — anything long-running must go through `jms_submit_job`. ``timeout``
    is per host; a host that exceeds it is reported as timed out, not an error.
    """
    if _looks_dangerous(command) and not allow_dangerous:
        return {
            "ok": False,
            "error": "command matched the safety denylist; pass allow_dangerous=true to override",
            "command": command,
        }
    hosts = _resolve(targets)
    if not hosts:
        return _empty_refusal(targets, results=[])
    results = await _ssh().exec_many(hosts, command, timeout=timeout)
    return {"ok": all(r.ok for r in results), "results": _fanout_reply(results)}


# --- data plane: file sync --------------------------------------------------------


@_tool(destructive=True)
async def jms_sync_push(
    targets: list[str],
    local: str,
    remote: str,
    excludes: list[str] | None = None,
    delete: bool = False,
) -> dict:
    """rsync a local path to every targeted host (through the bastion).

    Defaults exclude common noise (`.git`, virtualenvs, caches) so a push does not
    drag a workstation's junk onto a cluster node.
    """
    hosts = _resolve(targets)
    excludes = excludes if excludes is not None else _DEFAULT_EXCLUDES
    if not hosts:
        return _empty_refusal(targets, results=[])
    results = await _gather_rsync(hosts, local, remote, "push", excludes, delete)
    return {"ok": all(r.ok for r in results), "results": _fanout_reply(results)}


@_tool(idempotent=True)
async def jms_sync_pull(
    targets: list[str], remote: str, local: str, excludes: list[str] | None = None
) -> dict:
    """rsync a remote path from every targeted host back to the local machine."""
    hosts = _resolve(targets)
    if not hosts:
        return _empty_refusal(targets, results=[])
    results = await _gather_rsync(hosts, local, remote, "pull", excludes or [], False)
    return {"ok": all(r.ok for r in results), "results": _fanout_reply(results)}


async def _gather_rsync(hosts, local, remote, direction, excludes, delete):  # noqa: ANN001
    return list(
        await asyncio.gather(
            *(
                _ssh().rsync(
                    h,
                    local,
                    remote,
                    direction=direction,
                    delete=delete,
                    excludes=excludes,
                )
                for h in hosts
            )
        )
    )


#: Where `jms_pull_artifacts` puts what it pulls, one directory per host.
_ARTIFACT_ROOT = "~/.jms-bridge/artifacts"

#: Artifacts are checkpoints and can be gigabytes, so the fan-out is bounded far below
#: the SSHRunner's own cap: a handful of concurrent transfers keeps a shared uplink
#: usable for everyone else on the cluster.
_ARTIFACT_PARALLELISM = 4

#: Machine-generated noise, not artifacts. Deliberately *not* `_DEFAULT_EXCLUDES` — that
#: list drops whole directories, and pulling a run's output directory must not silently
#: skip a `.git`- or cache-named subdirectory the run actually wrote.
_ARTIFACT_EXCLUDES = ["*.pyc", "__pycache__", ".pytest_cache"]


def _artifact_remote_error(remote: str) -> str | None:
    """Reject a remote path that could reach outside the host workdir.

    The path is agent-supplied and the pull runs as the asset's account, so an absolute
    path or a `..` segment would rsync something like `/etc` back to the workstation.
    Artifacts belong under the run's workdir, so require a relative, `..`-free path; if
    a run wrote elsewhere, this is the point to say so rather than to guess.
    """
    if not remote.strip():
        return "remote is empty"
    if remote.startswith(("/", "~")):
        return "remote must be relative to the host workdir (no leading '/' or '~')"
    if ".." in remote.replace("\\", "/").split("/"):
        return "remote must not contain a '..' segment"
    return None


def _artifact_subdir(name: str) -> str:
    """A filesystem-safe directory name for one host.

    Host names are operator-supplied and may contain characters that mean something to
    a path (or be `.`/`..`). One directory per host is what stops same-named files from
    different hosts overwriting each other on the way back.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip(".")
    return safe or "host"


@_tool(idempotent=True)
async def jms_pull_artifacts(
    targets: list[str],
    remote: str,
    local: str | None = None,
    excludes: list[str] | None = None,
) -> dict:
    """rsync one artifact path from every targeted host into its own local directory.

    `jms_sync_pull` sends every host to the *same* local path, so pulling the same
    relative path off ten nodes means ten transfers into one directory and nine silent
    overwrites. This gives each host its own subdirectory instead —
    `~/.jms-bridge/artifacts/<host>/` by default, or `<local>/<host>/` — so a
    per-node checkpoint, log or metric dump survives the trip back.

    `remote` must be relative to the host workdir (no leading `/` or `~`, no `..`): the
    copy runs as the asset's account, and an absolute path would pull arbitrary system
    files off the cluster. `excludes` defaults to machine-generated noise only — a
    checkpoint pull should not also drag back a run's caches. Transfers are capped at
    four at a time and a host that fails is reported on its own, not at the expense of
    the others.
    """
    hosts = _resolve(targets)
    bad = _artifact_remote_error(remote)
    if bad:
        return {"ok": False, "error": bad, "remote": remote}
    root = os.path.expanduser(local or _ARTIFACT_ROOT)
    excludes = _ARTIFACT_EXCLUDES if excludes is None else excludes
    sem = asyncio.Semaphore(_ARTIFACT_PARALLELISM)

    async def one(host: Host) -> dict:
        dest = os.path.join(root, _artifact_subdir(host.name))
        async with sem:
            try:
                res = await _ssh().rsync(host, dest, remote, direction="pull", excludes=excludes)
            except Exception as exc:  # noqa: BLE001 - one host must not hide the rest
                return {"host": host.name, "local": dest, "ok": False, "error": str(exc)}
        return {
            "host": host.name,
            "local": dest,
            "ok": res.ok,
            "stderr": res.stderr.strip()[:400] if not res.ok else "",
        }

    results = list(await asyncio.gather(*(one(h) for h in hosts)))
    return {
        "ok": bool(results) and all(r["ok"] for r in results),
        "remote": remote,
        "root": root,
        "results": results,
    }


async def _per_host(hosts: list[Host], fn: Callable[..., Any], *, key: str) -> list[dict]:
    """Run ``fn(host)`` on every host; one host's failure becomes that host's entry.

    A plain ``gather`` lets a single unreachable host raise and erase every other host's
    result — on a 40-node cluster one bad node hides the other 39. ``CancelledError`` is
    a ``BaseException``, so a real cancellation still propagates.
    """

    async def one(host: Host) -> dict:
        try:
            return {"host": host.name, key: await fn(host)}
        except Exception as exc:  # noqa: BLE001 - isolating one host is the whole point
            return {"host": host.name, key: None, "error": f"{type(exc).__name__}: {exc}"}

    return list(await asyncio.gather(*(one(h) for h in hosts)))


# --- data plane: jobs -------------------------------------------------------------


@_tool(destructive=True)
async def jms_submit_job(targets: list[str], command: str, allow_dangerous: bool = False) -> dict:
    """Launch a long-running command (e.g. Torch-FL training) detached on each host.

    Returns immediately with a job id per host. The command runs inside a tmux
    session with its env activated, writing to a log you poll with `jms_job_logs`.
    Use this — never `jms_exec` — for anything that outlives a tool call.
    """
    if _looks_dangerous(command) and not allow_dangerous:
        return {
            "ok": False,
            "error": "command matched the safety denylist; pass allow_dangerous=true to override",
            "command": command,
        }
    hosts = _resolve(targets)
    if not hosts:
        return _empty_refusal(targets, jobs=[])
    jobs = _jobs()
    out = []
    submitted = await asyncio.gather(*(jobs.submit(h, command) for h in hosts))
    for handle, res in submitted:
        out.append(
            {
                "host": handle.host if handle else "?",
                "job_id": handle.job_id if handle else None,
                "launched": handle is not None,
                "stderr": res.stderr.strip()[:400] if handle is None else "",
            }
        )
    return {"ok": all(o["launched"] for o in out), "jobs": out}


@_tool(read_only=True)
async def jms_job_status(targets: list[str], job_id: str) -> JobStatusOut:
    """Check a job's status on each targeted host (status, exit code, log size, alive).

    `ok` is false when a targeted host could not be asked. A job that is merely
    `missing` on a reachable host is an answer, not a failure — the job may simply be
    one this bridge never launched, or one whose status file is gone.
    """
    hosts = _resolve(targets)
    bad = validate_job_id(job_id)
    if bad:
        return {"ok": False, "error": bad}
    if not hosts:
        return _empty_refusal(targets, status=[])
    results = await asyncio.gather(*(_jobs().status(h, job_id) for h in hosts))
    return {"ok": all(r.get("ok") for r in results), "status": list(results)}


@_tool(read_only=True)
async def jms_job_logs(
    host: str, job_id: str, offset: int = 0, max_bytes: int = 65536
) -> dict:
    """Read a job's log from `offset` onward. Poll with the returned `next_offset`.

    Deliberately *unstructured* (no `outputSchema`): a tool whose payload is a text blob
    gets no field-discoverability benefit from a schema, but does pay its cost — a
    structured result is sent alongside the text block, so the log would cross the wire
    twice. Measured at ~100 KB of log that is a doubled reply for nothing. The tools whose
    payload is a small record per host (`jms_job_status`, `jms_cluster_preflight`, …) keep
    their schema.
    """
    bad = validate_job_id(job_id)
    if bad:
        return {"ok": False, "error": bad}
    hosts = _resolve([host])
    if not hosts:
        return _empty_refusal([host], host=host, job_id=job_id)
    read = await _jobs().logs(hosts[0], job_id, offset=offset, max_bytes=max_bytes)
    # `logs()` reports an empty `size=0` body for an unreachable host exactly as it does
    # for a job that has not written a line yet, so the tool has to carry the read's own
    # transport status or a dead host reads as "nothing new to read". `ok` therefore comes
    # from the read itself (the same way `jms_job_cancel` takes it from `cancel()`), and
    # the host-level entry keeps the tri-state DetailResult-style fields.
    return {"ok": read.get("ok", False), **read}


@_tool(destructive=True, idempotent=True)
async def jms_job_cancel(host: str, job_id: str) -> dict:
    """Kill a running job on one host."""
    bad = validate_job_id(job_id)
    if bad:
        return {"ok": False, "error": bad}
    hosts = _resolve([host])
    if not hosts:
        return _empty_refusal([host], host=host, job_id=job_id)
    return {"ok": True, **(await _jobs().cancel(hosts[0], job_id))}


@_tool(read_only=True)
async def jms_job_wait(
    targets: list[str], job_id: str, timeout: float = 50.0, interval: float = 2.0
) -> dict:
    """Block until a job has stopped on every targeted host, or `timeout` elapses.

    Polls `jms_job_status` for you and returns the moment every targeted host reports a
    terminal status (`done` / `failed` / `missing`) — the same job id on each host, which
    is what a coordinated `jms_cluster_submit` hands out. `finished` says whether they
    all stopped; a timeout is not a failure and **kills nothing**, so on `finished=false`
    the jobs are still running and you can call this again or keep polling manually.

    The default is deliberately below the tightest client tool-call cap we know of (dsh's
    60 s), so a bounded wait returns its partial answer *itself* instead of being cut off
    by the client first — the difference between `finished: false` and an opaque timeout.
    Raising `timeout` past that also requires raising the client's cap (`toolCallTimeoutMs`
    in dsh), and the gateway buffers a non-streaming response for ~200 s, so stay under it.
    """
    hosts = _resolve(targets)
    bad = validate_job_id(job_id)
    if bad:
        return {"ok": False, "error": bad}
    if not hosts:
        # `wait([])` polls nothing and `all()` over nothing is True, so an empty
        # selection would report "every job finished" about hosts never asked.
        return _empty_refusal(targets, job_id=job_id, finished=False, status=[])
    latest, finished = await _jobs().wait(
        [(h, job_id) for h in hosts], timeout=timeout, interval=interval
    )
    return {"ok": finished, "finished": finished, "job_id": job_id, "status": latest}


@_tool(read_only=True)
async def jms_job_list(targets: list[str]) -> dict:
    """List the job ids currently running on each targeted host.

    Answers "what did I leave running on this cluster?" — useful after a lost session,
    when the job ids you were handed are gone but the tmux sessions are not.

    Reads `tmux ls`, so it sees jobs launched by `tmux` only: a host that fell back to
    `setsid` (no tmux installed) or a job nested inside a container is invisible here,
    and a job is only listed while its tmux session lives — a finished job is gone from
    this list even though its `status` file and logs remain readable. Pass the ids you
    find to `jms_job_status` / `jms_job_wait` / `jms_job_cancel`.
    """
    hosts = _resolve(targets)
    per_host = await _per_host(hosts, _jobs().list_jobs, key="job_ids")
    return {
        # A host that could not be asked reports its own error; that must not read as
        # "this host has no jobs". An empty selection is likewise not a green listing.
        "ok": bool(per_host) and all("error" not in entry for entry in per_host),
        "hosts": per_host,
        "note": (
            "tmux-launched jobs only; a host with no tmux falls back to setsid and lists nothing"
        ),
    }


@_tool(destructive=True, idempotent=True)
async def jms_cluster_cancel(targets: list[str], job_id: str) -> dict:
    """Stop every rank of one coordinated launch.

    A `jms_cluster_submit` hands every rank the *same* job id (on a different host each),
    so one cancel per host is what actually stops a collective — and doing that by hand
    from the per-rank reply is easy to get half-right, leaving a rank alive still holding
    the rendezvous port and the cards. Pass the same `targets` you submitted with and the
    one job id. Hosts are cancelled in parallel and independently; a host where nothing
    was killed reports `killed=false` rather than failing the batch.
    """
    hosts = _resolve(targets)
    bad = validate_job_id(job_id)
    if bad:
        return {"ok": False, "error": bad}
    if not hosts:
        # `resolve([])` is empty without raising, and `all()` over nothing is True — a
        # cancel that reached no host must not report success.
        return {
            "ok": False,
            "job_id": job_id,
            "error": "no hosts matched the selection; nothing was cancelled",
            "cancelled": [],
            "results": [],
        }
    results = await _jobs().cancel_many([(h, job_id) for h in hosts])
    return {
        "ok": bool(results) and all(r.get("ok") for r in results),
        "job_id": job_id,
        "cancelled": [r for r in results if r.get("killed")],
        "results": results,
    }


# --- dev loop: workspace sync / run / logs -----------------------------------------


@_tool(idempotent=True)
async def jms_inventory_reload() -> dict:
    """Re-read `hosts.yaml` (and the config) and drop everything cached from them.

    The advertised dev loop is *emit YAML → paste into `hosts.yaml` → run*, but every
    tool reads the inventory through a cache populated on first use. Without this the
    paste step silently needs an MCP restart before the new hosts/workspaces appear.
    Call this after editing the inventory; the next tool call re-reads from disk.

    The config is dropped too, and for the same reason: `jms-bridge login` caches a
    session token *into the config file*, so a server started before logging in would
    otherwise hold a credential-less config until it was restarted.

    Returns the freshly loaded summary so the caller can confirm the paste landed.
    """
    stale_ssh = _state.pop("ssh", None)
    if stale_ssh is not None:
        await stale_ssh.close()  # else each reload leaks a ControlMaster socket dir
    for key in ("inv", "cfg", "jobs", "api"):
        _state.pop(key, None)
    try:
        inv = _inv()
    except Exception as exc:  # noqa: BLE001 - surface any load error to the agent
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "hosts": sorted(inv.hosts),
        "workspaces": sorted(inv.workspaces),
        "groups": {k: list(v) for k, v in inv.groups.items()},
    }


def _workspaces(selectors: list[str]):  # noqa: ANN202
    return _inv().resolve_workspaces(selectors)


def _host_of(ws):  # noqa: ANN001, ANN202
    return _inv().hosts[ws.host]


#: A git object name as `git rev-parse` emits it: 7..40 lowercase hex chars.
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _local_sha(path: str) -> str | None:
    """Best-effort HEAD sha of a *local* git tree (for ``push`` workspaces).

    The container has no git for a ``push`` workspace, but the developer's tree
    does, and a fork and the local tree share commit ids — so this is what lets a
    mixed git+push run detect that the pushed tree is not at ``ref``. Validated
    against :data:`_SHA_RE` so a banner or a short/full mismatch cannot slip
    through as a divergence (or false agreement).
    """
    try:
        out = subprocess.run(
            # expanduser: `local:` in the inventory is often `~/...`, and this path
            # goes straight to exec (no shell), so `~` would never be expanded.
            ["git", "-C", os.path.expanduser(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip() if out.returncode == 0 else ""
    return sha if _SHA_RE.match(sha) else None


def _last_sha(stdout: str) -> str | None:
    """The last sha-shaped line of an SSH stdout, or None.

    Scanning for a *sha-shaped* line (bottom-up) rather than the last non-empty
    line means a container logout banner or login notice printed to stdout after
    ``git rev-parse`` is ignored instead of being mistaken for the commit.
    """
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if _SHA_RE.match(candidate):
            return candidate
    return None


async def _deliver(ws, ref: str) -> dict:  # noqa: ANN001
    """Deliver ``ref`` to one workspace, never raising.

    The fan-out over workspaces is a ``gather``: an uncaught raise from one workspace
    (a workspace naming a host the inventory no longer has, a transport error) would
    discard every other workspace's result and reach the agent as an opaque tool error.
    Converting it into this workspace's failure record keeps the per-chip picture
    intact and names the workspace that broke.
    """
    try:
        return await _deliver_one(ws, ref)
    except Exception as exc:  # noqa: BLE001 - isolating one workspace is the point
        return {
            "workspace": ws.name,
            "source": getattr(ws, "source", "?"),
            "ok": False,
            "sha": None,
            "ref_pinned": False,
            "stderr": f"delivery raised: {exc}",
        }


async def _deliver_one(ws, ref: str) -> dict:  # noqa: ANN001
    """Put the workspace's repo at ``ref`` using its configured source.

    Returns the delivered commit under ``sha`` (best-effort) so a caller can spot
    revision divergence across chips. For ``push`` the code arrives out-of-band and
    the container has no git, so ``ref_pinned`` is False and the sha — when the
    local tree could be read — is the *local* HEAD, not the container's.
    """
    host = _host_of(ws)
    if ws.source == "push":
        # `local:` is commonly `~/...` (the shipped example uses it), and rsync runs
        # via exec with no shell, so `~` would stay literal and the whole delivery
        # would silently push a path named `~`. `_local_sha` expands it for the same
        # reason; the path that actually reaches rsync must match.
        res = await _ssh().rsync(
            host,
            os.path.expanduser(ws.local),
            ws.mount,
            direction="push",
            excludes=_DEFAULT_EXCLUDES,
        )
        sha = await asyncio.to_thread(_local_sha, ws.local)
        return {
            "workspace": ws.name,
            "source": "push",
            "ok": res.ok,
            # Push never applies `ref` in-container; `sha` (below) is the *local*
            # tree's HEAD, which is why ref_pinned is always False here.
            "ref_pinned": False,
            "sha": sha,
            "stderr": res.stderr.strip()[:400],
        }
    cmd = ws.delivery_cmd(ref)
    # One round trip: fetch + checkout write to stderr, rev-parse to stdout.
    res = await _ssh().exec_on(
        host,
        ws.container_cmd(f"{cmd} && git rev-parse HEAD"),
        timeout=300,
        workdir=False,
        env_setup=False,
    )
    sha = _last_sha(res.stdout)
    return {
        "workspace": ws.name,
        "source": ws.source,
        # A delivery that cannot report which commit it landed on is a failure:
        # a run over unverified code must not report ok.
        "ok": res.ok and sha is not None,
        "sha": sha,
        "ref_pinned": True,
        "stderr": res.stderr.strip()[:400],
    }


@_tool(destructive=True)
async def jms_dev_sync(targets: list[str], ref: str) -> dict:
    """Deliver a git `ref` to every selected workspace's container.

    Each workspace uses its configured source (git / proxy-git / push), so hosts with
    direct GitHub access, hosts behind a proxy, and offline hosts all work from this
    one call. `ref` is a branch on the workspace's `origin` (the developer's fork).
    """
    workspaces = _workspaces(targets)
    results = await asyncio.gather(*(_deliver(w, ref) for w in workspaces))
    # `bool(results)`: delivering nothing (empty selector) is not success —
    # all([]) is True, the same trap the consistency table guards against.
    return {
        "ok": bool(results) and all(r["ok"] for r in results),
        "ref": ref,
        "results": list(results),
    }


@_tool(destructive=True, idempotent=True)
async def jms_dev_run(
    targets: list[str],
    ref: str,
    timeout: float = 600.0,
    ut_cmd: str | None = None,
    allow_dangerous: bool = False,
    ctx: Context | None = None,
) -> DevRunOut:
    """Deliver `ref` to every workspace, run the UT suite in each container, and
    return ONE cross-chip consistency table.

    This is the development loop: write once, verify on every chip. Each workspace
    runs its configured `ut_cmd` (override with `ut_cmd`) and reports per-test
    outcomes; results are aligned by test id. A FAIL carries a one-line summary
    inline — fetch full output with `jms_dev_logs`.

    An `ut_cmd` override MUST emit pytest `-rA` short-summary lines
    (`PASSED/FAILED/ERROR/SKIPPED ... <id> - <msg>`): the consistency table is
    built from them, so a command without `-rA` yields no rows and the run is
    reported as not-ok even on a clean pass.

    Long training runs do NOT go here; use `jms_submit_job` / `jms_cluster_submit`.
    """
    # `ut_cmd` is the one caller-supplied string this tool interpolates into a container
    # shell, so it gets the same denylist as `jms_exec`'s `command`. The workspace's own
    # `ut_cmd` / `env_setup` come from the hand-written inventory, the same trust level
    # as the rest of `hosts.yaml`, and are deliberately not screened here.
    if ut_cmd and _looks_dangerous(ut_cmd) and not allow_dangerous:
        return {
            "ok": False,
            "stage": "guard",
            "error": "ut_cmd matched the safety denylist; pass allow_dangerous=true to override",
            "ref": ref,
            "columns": [],
            "rows": [],
            "revisions": {},
            "revisions_consistent": False,
            "revisions_unverified": [],
            "failed_hosts": [],
            "results": [],
            "attestation": {},
            "attestation_consistent": False,
            "attestation_diverged": [],
            "attestation_unverified": [],
            "hosts": {},
        }
    from .attest import (
        build_attest_script,
        check_attestation_parity,
        parse_attestation,
    )
    from .workspace import DEFAULT_UT_CMD, build_consistency_table, parse_pytest_summary

    workspaces = _workspaces(targets)
    delivered = await asyncio.gather(*(_deliver(w, ref) for w in workspaces))
    # The early returns carry the *same* keys as the success path (`attestation`, the
    # `attestation_*` trio, `hosts` included) so a caller that reads any of them
    # uniformly never KeyErrors on a delivery failure.
    empty_attest = {
        "attestation": {},
        "attestation_consistent": False,
        "attestation_diverged": [],
        "attestation_unverified": [],
        "hosts": {},
    }
    if not delivered:
        # No workspace matched: report it the same shape as a normal run (empty
        # table) rather than a bare success, so the caller can always read `rows`.
        return {
            "ok": False,
            "stage": "deliver",
            "error": "no workspace matched the selector",
            "ref": ref,
            "columns": [],
            "rows": [],
            "revisions": {},
            "revisions_consistent": False,
            "revisions_unverified": [],
            "failed_hosts": [],
            "results": [],
            **empty_attest,
        }
    if not all(d["ok"] for d in delivered):
        return {
            "ok": False,
            "stage": "deliver",
            "ref": ref,
            "columns": [],
            "rows": [],
            "revisions": {},
            "revisions_consistent": False,
            "revisions_unverified": [d["workspace"] for d in delivered if not d.get("sha")],
            "failed_hosts": [d["workspace"] for d in delivered if not d["ok"]],
            "results": list(delivered),
            **empty_attest,
        }

    async def _run_one(ws):  # noqa: ANN001, ANN202
        host = _host_of(ws)
        # Run the suite from inside the repo: the container's default cwd is not
        # the checkout, so `python -m pytest tests/unit` would not find the tests.
        # `cd --` so a repo path beginning with `-` is not read as a cd option.
        # `set -o pipefail` *inside the subshell* is essential: the inner runs under
        # `docker exec ... bash -lc`, a fresh shell that inherits neither `-e` nor
        # pipefail from the host wrapper, so a piped `ut_cmd` (`... | tee log`) would
        # return tee's 0 and mask a fleet-wide failure.
        # The attestation probe rides the same round trip, *after* the suite. It must
        # never `exit` (it would clobber pytest's status), so the UT run's exit code is
        # captured and re-raised explicitly — otherwise exit_code would always be the
        # probe's and `failed` could never fire. An empty `ut_cmd` (a YAML null override)
        # falls back to the default rather than yielding a `()` syntax error.
        # Activate the container's conda env *before* the repo `cd`, so the UT
        # command and the attestation probe both use the interpreter the built
        # `_C` extension was compiled for (the container default is the wrong ABI
        # on most chips). `env_setup` is caller-controlled spec text, not argv.
        cmd = ut_cmd or ws.ut_cmd or DEFAULT_UT_CMD
        env_prefix = f"{ws.env_setup}; " if ws.env_setup else ""
        inner = (
            env_prefix
            + f"cd -- {shell_quote(ws.repo)} && {{ (set -o pipefail; {cmd}); rc=$?; "
            + build_attest_script(ws.repo)
            + "exit $rc; }"
        )
        res = await _ssh().exec_on(
            host,
            # The UT run must not inherit the delivery proxy.
            ws.container_cmd(inner, with_proxy=False),
            timeout=timeout,
            workdir=False,
            env_setup=False,
        )
        return ws.name, res, parse_pytest_summary(res.stdout)

    # Progress: a fleet-wide UT run takes minutes, and without this the client sees only a
    # spinner — indistinguishable from a hang. Count chips as they finish so the user sees
    # movement (and, when one chip is stuck, which one is outstanding).
    total = len(workspaces)
    done_count = 0

    async def _one(ws):  # noqa: ANN001, ANN202
        """Run one chip, turning a raise into *that chip's* failure.

        A raise before a result exists (a workspace naming a host the inventory no
        longer has, a transport error) would otherwise escape the fan-out `gather` and
        replace the whole cross-chip table — this tool's entire product — with an
        opaque tool error. Here the chip is reported as failed and the table survives.
        """
        nonlocal done_count
        try:
            return await _run_one(ws)
        except Exception as exc:  # noqa: BLE001 - isolating one chip is the point
            return ws.name, ExecResult(ws.name, 1, "", f"run failed: {exc}"), {}
        finally:
            done_count += 1
            await _progress(ctx, done_count, total, f"{done_count}/{total} chips done")

    ran = await asyncio.gather(*(_one(w) for w in workspaces))
    per_ws = {name: parsed for name, _, parsed in ran}
    table = build_consistency_table(per_ws)
    # A chip that timed out or exited nonzero (often with empty stdout, hence no
    # parsed rows) must fail the whole run, not just be reported as informational.
    failed = [name for name, res, _ in ran if not res.ok]
    shas = {d["workspace"]: d["sha"] for d in delivered if d.get("sha")}
    # A workspace whose revision could not be read must not be silently dropped:
    # comparing only the survivors could report agreement while a chip ran
    # unverified code.
    unverified = [d["workspace"] for d in delivered if not d.get("sha")]
    revisions_consistent = len(set(shas.values())) <= 1 and len(shas) == len(delivered)

    # Attestation: the same sha is not the same *code path*. Parse each chip's record
    # out of the UT stdout and require the op->backend map (and plugin version, where
    # readable) to agree. A chip whose map could not be read is `unverified`, never
    # coerced into agreement — the same rule `revisions` follows.
    attestation = {name: parse_attestation(res.stdout) for name, res, _ in ran}
    parity = check_attestation_parity(attestation)

    table.update(
        {
            "ok": (
                table["ok"]
                and not failed
                and revisions_consistent
                and parity["consistent"]
            ),
            "ref": ref,
            "failed_hosts": failed,
            "revisions": shas,
            "revisions_consistent": revisions_consistent,
            "revisions_unverified": unverified,
            "attestation": attestation,
            "attestation_consistent": parity["consistent"],
            "attestation_diverged": parity["diverged"],
            "attestation_unverified": parity["unverified"],
            "hosts": {
                name: {
                    "ok": res.ok,
                    "exit_code": res.exit_code,
                    "stderr": res.stderr.strip()[:400],
                }
                for name, res, _ in ran
            },
        }
    )
    return table


@_tool(idempotent=True)
async def jms_dev_logs(workspace: str, test_id: str, timeout: float = 300.0) -> dict:
    """Full output for one `test_id` on one `workspace` (drill into a FAIL).

    Re-runs just that test in the container with a full traceback, so it works for
    any suite without depending on a saved report file.
    """
    candidates = _inv().resolve_workspaces([workspace])
    if len(candidates) != 1:
        # `resolve_workspaces` treats a selector as a union, so a *tag* can match many
        # workspaces. Silently drilling into the first would answer about a different
        # chip than the one whose FAIL the caller meant to investigate — so an ambiguous
        # selector is refused by name, and the caller picks one.
        return {
            "ok": False,
            "workspace": workspace,
            "test_id": test_id,
            "error": (
                f"{workspace!r} matched {len(candidates)} workspaces "
                f"({sorted(w.name for w in candidates)}); name exactly one"
            ),
            "matched": sorted(w.name for w in candidates),
        }
    ws = candidates[0]
    host = _host_of(ws)
    # Activate the container env *before* the repo `cd`, exactly as jms_dev_run does.
    # The whole point of this tool is to say why a test failed on a chip; re-running
    # it under the container's default interpreter (the wrong ABI on most chips)
    # would report the cause of a failure that is not the one the fleet run hit.
    env_prefix = f"{ws.env_setup}; " if ws.env_setup else ""
    inner = (
        env_prefix
        + f"cd -- {shell_quote(ws.repo)} && python -m pytest "
        + f"{shell_quote(test_id)} -q --tb=long -rA"
    )
    res = await _ssh().exec_on(
        host,
        ws.container_cmd(inner, with_proxy=False),
        timeout=timeout,
        workdir=False,
        env_setup=False,
    )
    return {
        "ok": res.ok,
        "workspace": ws.name,
        "test_id": test_id,
        "exit_code": res.exit_code,
        "output": res.stdout[-20000:],
        "stderr": res.stderr.strip()[:2000],
    }


@_tool(read_only=True)
async def jms_dev_discover(
    targets: list[str],
    container: str = DEFAULT_CONTAINER,
    source: str = "git",
    repo_match: str = DEFAULT_REPO_MATCH,
    max_depth: int = 5,
    proxy: str | None = None,
    mount: str | None = None,
    local: str | None = None,
    timeout: float = 120.0,
    ctx: Context | None = None,
) -> dict:
    """Inspect each host's container and emit a paste-ready `workspaces:` block.

    This is the front door to the dev loop. `jms_dev_run` needs to know, per chip,
    which container holds the repo, where the checkout lives inside it, and whether
    the container can reach the git remote — and that is exactly the per-host
    busywork worth automating. One call fans out `docker exec` discovery to every
    host and returns the inventory YAML to paste into `hosts.yaml`.

    `container` is normally the same name on every host (`lvyufeng-torch-fl`);
    `source` seeds the `source:` field (git / proxy-git / push) and `repo_match` is
    a case-insensitive substring a checkout path must contain. Because the emitted
    block must be *loadable*, `source='proxy-git'` requires `proxy` and
    `source='push'` requires `mount` + `local` — without them the pasted entry would
    make `Inventory.load` reject the whole inventory, so this call refuses rather
    than emit a block that cannot be pasted.

    Every host's raw findings are in `hosts`; anything the probe could not confirm
    is a `warnings` entry (a timed-out or unreachable origin, a missing pytest, a
    missing docker, a container that is not there) and a host that could not be
    inspected is listed in `skipped` — `ok` is true only when every requested host
    produced at least one workspace.

    The emitted `source` is a *guess* from reachability alone; confirm it before
    relying on it.
    """
    from .discover import (
        build_discovery_script,
        parse_discovery,
        render_workspaces_yaml,
    )

    if not CONTAINER_RE.fullmatch(container):
        return {"ok": False, "error": f"invalid container name {container!r}"}
    if source not in SOURCES:
        return {"ok": False, "error": f"source must be one of {SOURCES}, got {source!r}"}
    # The emitted block is advertised as paste-ready, so it must survive
    # Inventory.load: Workspace rejects proxy-git without a proxy and push without
    # both mount and local, and a single bad entry aborts the whole inventory.
    if source == "proxy-git" and not proxy:
        return {"ok": False, "error": "source 'proxy-git' requires a 'proxy' argument"}
    if source == "push" and not (mount and local):
        return {"ok": False, "error": "source 'push' requires both 'mount' and 'local'"}

    hosts = _resolve(targets)
    if not hosts:
        # Nothing was inspected, which is not "every requested host produced a
        # workspace": say so in the tool's own shape (as the other fan-outs do) so the
        # caller can tell an empty selection from a discovery that found nothing.
        return _empty_refusal(
            targets,
            container=container,
            source=source,
            hosts=[],
            skipped=[],
            warnings=[],
            workspaces=[],
            yaml="",
        )
    script = build_discovery_script(container, repo_match=repo_match, max_depth=max_depth)
    # Discovery runs one `docker exec` per host and can take minutes on a wide fleet, so
    # report progress per host rather than leaving the client with a bare spinner.
    total = len(hosts)
    done_count = 0

    async def _discover_one(h: Host):
        nonlocal done_count
        try:
            # Discovery must not depend on the host toolchain (env_setup) nor the
            # host workdir — it inspects a container, which is host-toolchain-free.
            return await _ssh().exec_on(
                h, script, timeout=timeout, workdir=False, env_setup=False
            )
        finally:
            done_count += 1
            await _progress(ctx, done_count, total, f"{done_count}/{total} hosts inspected")

    results = await asyncio.gather(*(_discover_one(h) for h in hosts))
    reports = [
        {
            "host": res.host,
            "exit_code": res.exit_code,
            "timed_out": res.timed_out,
            "timeout": timeout,
            "report": parse_discovery(res.stdout),
            "stderr": res.stderr.strip()[:400],
        }
        for res in results
    ]
    rendered = render_workspaces_yaml(
        reports,
        container=container,
        source=source,
        proxy=proxy,
        mount=mount,
        local=local,
        host_tags={h.name: h.tags for h in hosts},
    )
    return {
        "ok": rendered["ok"],
        "container": container,
        "source": source,
        "hosts": reports,
        "skipped": rendered["skipped"],
        "warnings": rendered["warnings"],
        "workspaces": rendered["workspaces"],
        "yaml": rendered["yaml"],
    }


# --- cluster: preflight + coordinated launch ---------------------------------------


@_tool(read_only=True)
async def jms_cluster_preflight(
    targets: list[str], port: int = DEFAULT_RDZV_PORT
) -> ClusterPreflightOut:
    """Check a cluster is ready for a distributed run, host by host.

    For every host: reachability, accelerator visibility and version (via the
    platform probe), whether the rendezvous `port` is already bound (a stale previous
    run), and how many accelerators already hold a compute process. Returns an
    estimated `world_size` and a suggested rendezvous. Run this before
    `jms_cluster_submit`.

    `ok` is false when a host is unreachable, holds the rendezvous port, **or** already
    has accelerators in use — i.e. exactly when a default `jms_cluster_submit` would
    refuse. `busy_devices` is `null` when occupancy is unknown: a platform with no
    occupancy probe, or a host whose driver tool did not answer. Unknown is never
    reported as busy, so a probe gap degrades to the old behaviour instead of blocking
    a launch. Set `allow_busy=true` on the submit to override the busy gate.
    """
    hosts = _resolve(targets)
    if not hosts:
        # Nothing to preflight is not "ready": say so in the tool's own shape rather than
        # letting `resolve_rendezvous([])` raise an InventoryError the caller has to read
        # as text.
        return _empty_refusal(targets, hosts=[], world_size_estimate=0, busy_hosts=[])
    health = await preflight(_ssh(), hosts, port=port)
    # Match `jms_cluster_submit` exactly: a reachable host with no device count still gets
    # `nproc=1` there, so it must count 1 here. Counting 0 under-reported the world size
    # the submit would really launch, which is the number this estimate exists to predict.
    ranks = sum(max(1, h.device_count or 1) for h in health if h.reachable)
    rdv = resolve_rendezvous(hosts, port=port)
    busy = [h for h in health if h.busy]
    return {
        # `bool(health)` matters: this is the gate the README puts in front of
        # `jms_cluster_submit`, and a selector that matched no host preflights nothing —
        # reporting that as ready would wave an empty cluster through.
        "ok": bool(health) and all(h.ok for h in health) and not busy,
        "world_size_estimate": ranks,
        "rendezvous": rdv.to_dict(),
        "heterogeneous": is_heterogeneous(hosts),
        "islands": [i.to_dict() for i in plan_islands(hosts)],
        "hosts": [h.to_dict() for h in health],
        "platforms": sorted({h.platform for h in health}),
        "busy_hosts": [h.host for h in busy],
        "note": (
            f"{len(busy)} host(s) already have accelerators in use; jms_cluster_submit "
            "refuses a set containing them unless you pass allow_busy=true"
        )
        if busy
        else "",
    }


_FEDERATION_NOTE = (
    "Hosts sharing a collective engine in one island CAN form a single collective. "
    "Cross-island training must federate (exchange updates between rounds), not share "
    "a collective — HCCL and NCCL do not interoperate, and neither do two vendors' "
    "respective NCCL builds."
)


@_tool(read_only=True)
async def jms_cluster_plan(targets: list[str], port: int = DEFAULT_RDZV_PORT) -> dict:
    """Plan a federated layout for a target set, grouped by collective engine.

    Ascend (HCCL) and NVIDIA (NCCL) nodes cannot form a single collective, and neither
    can two vendors whose torch builds each ship their own NCCL. This
    partitions the targets into **islands** — one per engine — each of which runs
    its own coordinated job, and returns the federation plan: per-island rendezvous,
    world size and job wiring. Run `jms_cluster_submit` once per island, then let the
    islands exchange model updates between rounds (the FL exchange is the agent's
    application logic; the bridge provides per-island launch + rendezvous).
    """
    hosts = _resolve(targets)
    if not hosts:
        # `plan_islands([])` returns `[]` without complaint, so the reply would be an empty
        # `islands` and an `ok` that only stayed honest thanks to the `bool(plan)` below —
        # true, but silent about why. Say it in the tool's own shape, like the other fan-outs.
        return _empty_refusal(
            targets,
            heterogeneous=False,
            islands=[],
            note=_FEDERATION_NOTE,
        )
    plan = []
    for island in plan_islands(hosts):
        rdv = resolve_rendezvous(island.hosts, port=port)
        plan.append(
            {
                **island.to_dict(),
                "rendezvous": rdv.to_dict(),
                "world_size": len(island.hosts),
                "submit_with_targets": [h.name for h in island.hosts],
            }
        )
    return {
        # A selector that matched no host produces no islands, which is not a plan.
        "ok": bool(plan),
        "heterogeneous": is_heterogeneous(hosts),
        "islands": plan,
        "note": _FEDERATION_NOTE,
    }


@_tool(destructive=True)
async def jms_cluster_submit(
    targets: list[str],
    command: str,
    master: str | None = None,
    port: int = DEFAULT_RDZV_PORT,
    allow_busy: bool = False,
    allow_dangerous: bool = False,
) -> ClusterSubmitOut:
    """Launch ONE coordinated distributed job across a heterogeneous cluster.

    Unlike `jms_submit_job` (which runs the command independently on each host),
    this assigns every host a rank and gives them a shared rendezvous, exporting
    `JMS_RANK` (global device rank), `JMS_NODE_RANK` / `JMS_NNODES` (node ordinal and
    node count), `JMS_NPROC`, `JMS_WORLD_SIZE`, `JMS_MASTER_ADDR` and `JMS_MASTER_PORT`
    before `command` runs. All ranks launch concurrently and the call returns
    immediately with **one** job id, shared by every rank — pass it (and the same
    `targets`) to `jms_job_status` / `jms_job_wait` / `jms_job_logs` /
    `jms_cluster_cancel`.

    `command` is your per-node launcher; wire it with the exported env. For torchrun:

        torchrun --nnodes=$JMS_NNODES --node_rank=$JMS_NODE_RANK \\
                 --nproc_per_node=$JMS_NPROC \\
                 --rdzv_endpoint=$JMS_MASTER_ADDR:$JMS_MASTER_PORT \\
                 train.py --rounds 100

    (Use `JMS_NNODES` / `JMS_NODE_RANK` for `torchrun`: those are node counts, whereas
    `JMS_WORLD_SIZE` / `JMS_RANK` count *devices*. A 2-node, 2-chip run is `nnodes=2`,
    not `nnodes=4`.)

    `targets` order defines rank order; each host contributes one rank node whose
    `JMS_NPROC` is its accelerator count. A single host is allowed — that is the
    one-island case `jms_cluster_plan` emits for a mixed cluster (e.g. a lone Ascend
    box federating with a CUDA island) — and launches one rank node. Every target is
    preflighted first, and a host
    that is unreachable, holding a stale rendezvous `port`, or already running work on
    its accelerators fails the call rather than being handed a rank. The busy gate is
    what stops a second training being co-scheduled onto cards that are already in use;
    pass `allow_busy=true` to launch anyway — e.g. to resume a run whose previous job is
    still winding down. Occupancy the probe could not determine is not treated as busy,
    so a missing driver tool never blocks a launch on its own. `jms_cluster_preflight`
    reports the same busy hosts before you get here.
    """
    if _looks_dangerous(command) and not allow_dangerous:
        return {
            "ok": False,
            "error": "command matched the safety denylist; pass allow_dangerous=true to override",
            "command": command,
        }
    hosts = _resolve(targets)
    if not hosts:
        return _empty_refusal(targets, job_id=None, rendezvous={}, world_size=0, ranks=[])
    if is_heterogeneous(hosts):
        islands = plan_islands(hosts)
        return {
            "ok": False,
            "error": (
                "targets span multiple collective engines "
                f"({[i.collective for i in islands]}); a single collective is impossible "
                "(e.g. HCCL and NCCL do not interoperate, and neither do two vendors' "
                "respective NCCL builds). Submit one island at a time and federate "
                "across them."
            ),
            "islands": [i.to_dict() for i in islands],
            "hint": "call jms_cluster_plan, then jms_cluster_submit once per island",
        }
    rdv = resolve_rendezvous(hosts, master=master, port=port)

    health = {h.host: h for h in await preflight(_ssh(), hosts, port=port)}
    # Refuse before assigning ranks, rather than quietly shrinking the collective. An
    # unreachable host reports no `device_count` yet still contributed `nproc=1` to the
    # world size below, so every launched rank was told a world size larger than the set
    # that could join it — the job hangs in rendezvous, or the survivors silently train a
    # different-width collective than the caller asked for. A host with a stale rendezvous
    # port is equally unlaunchable: its rank cannot bind. Same readiness test as
    # `jms_cluster_preflight`, so the two tools never disagree about a target set.
    unready = [h for h in hosts if not health[h.name].ok]
    if unready:
        return {
            "ok": False,
            "error": (
                f"{len(unready)} of {len(hosts)} targets are not ready to join a collective "
                f"({', '.join(h.name for h in unready)}); assigning a rank to a host that "
                "cannot join leaves every other rank waiting on a rendezvous that never "
                "completes."
            ),
            "unready": [health[h.name].to_dict() for h in unready],
            "ready_targets": [h.name for h in hosts if health[h.name].ok],
            "hint": "fix the hosts above, or submit the ready subset by name",
        }
    # Busy is a *separate* gate from `ok`, and it is the one that keeps two trainings off
    # the same cards. It stays out of `HostHealth.ok` because the two failures need
    # different answers, and it is overridable because the most common honest reason a
    # card is busy is this caller's own job: refusing outright would make a legitimate
    # relaunch impossible. Unknown occupancy (`busy_devices` is None) is not busy.
    busy = [h for h in hosts if health[h.name].busy]
    if busy and not allow_busy:
        return {
            "ok": False,
            "error": (
                f"{len(busy)} of {len(hosts)} targets already have accelerators in use "
                f"({', '.join(h.name for h in busy)}); launching now would co-schedule a "
                "second training onto occupied cards."
            ),
            "busy": [health[h.name].to_dict() for h in busy],
            "ready_targets": [h.name for h in hosts if not health[h.name].busy],
            "hint": "pass allow_busy=true to launch anyway (e.g. resuming an unfinished job)",
        }
    assignments = [(h, max(1, health[h.name].device_count or 1)) for h in hosts]
    world_size = sum(n for _, n in assignments)

    handles, results = await _jobs().submit_collab(
        assignments, command, master_addr=rdv.master_addr, port=port, world_size=world_size
    )
    by_host = {h.host: h.job_id for h in handles}
    return {
        "ok": len(handles) == len(hosts),
        "rendezvous": rdv.to_dict(),
        "world_size": world_size,
        # The node ordinal / node count a `torchrun`-style launcher takes, as opposed to
        # the device-level `world_size` above. Hoisted so a caller can read the wiring
        # without re-deriving it from the rank list.
        "nnodes": len(assignments),
        # Every rank shares one job id (`submit_collab`), so it is hoisted out of the
        # per-rank entries and repeated here for convenience.
        "job_id": handles[0].job_id if handles else None,
        "ranks": [
            {
                "host": h.name,
                "node_rank": i,
                "job_id": by_host.get(h.name),
                "nproc": n,
                "launched": h.name in by_host,
            }
            for i, (h, n) in enumerate(assignments)
        ],
    }


# --- helpers ----------------------------------------------------------------------

_DEFAULT_EXCLUDES = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "*.pyc",
    ".mypy_cache",
    ".pytest_cache",
    "*.egg-info",
    ".DS_Store",
]

#: Patterns that are almost certainly a mistake when issued by an autonomous agent.
#: Not a security boundary — JumpServer's own command blacklist is the backstop.
_DANGEROUS_PATTERNS = [
    "rm -rf /",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",
    "shutdown",
    "reboot",
    "> /dev/sda",
    "chmod -R 777 /",
]


def _looks_dangerous(command: str) -> bool:
    flat = " ".join(command.split())
    return any(pat in flat for pat in _DANGEROUS_PATTERNS)


async def _aclose_state() -> None:
    """Tear down the lazily-built clients: multiplexed SSH sockets and the HTTP pool.

    Called once, on shutdown. Nothing else ever closes the runner, so without this the
    process leaves one ControlMaster socket per host it talked to (plus the short
    ``/tmp/jmsb-*`` directory) behind for every session.
    """
    ssh = _state.pop("ssh", None)
    api = _state.pop("api", None)
    _state.pop("jobs", None)
    if api is not None:
        await api.aclose()
    if ssh is not None:
        await ssh.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        server.run("stdio")
    finally:
        # ``server.run`` returns on stdin EOF and on Ctrl-C; either way the loop it
        # created is finished, so a fresh one here is fine. Cleanup is best-effort —
        # a failure must not turn a clean exit into a nonzero one.
        try:
            asyncio.run(_aclose_state())
        except Exception:  # noqa: BLE001 - shutdown must not raise
            log.debug("state cleanup on shutdown failed", exc_info=True)


if __name__ == "__main__":
    main()
