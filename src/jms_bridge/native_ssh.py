"""Process-owned SSH connections, durable results, and optional persistent shells."""

import asyncio
import contextlib
import contextvars
import json
import os
import re
import shlex
import sys
import tempfile
import time
import uuid
from pathlib import Path

import asyncssh

from .config import ConfigError
from .results import atomic_json
from .ssh import ExecResult


def open_output(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


async def pump(reader, path, marker=None, progress=None):
    found, buffer = False, b""
    with open_output(path) as out:
        while True:
            try:
                chunk = await reader.read(65536)
            except BaseException:
                out.write(buffer)
                raise
            if not chunk:
                out.write(buffer)
                return found
            if marker is None or found:
                out.write(chunk)
                out.flush()
                continue
            buffer += chunk
            position = buffer.find(marker)
            if position >= 0:
                out.write(buffer[:position])
                out.write(buffer[position + len(marker) :])
                buffer, found = b"", True
                if progress is not None:
                    progress["started"] = True
                out.flush()
            elif len(buffer) >= len(marker):
                keep = len(marker) - 1
                out.write(buffer[:-keep])
                buffer = buffer[-keep:]


class ShellSession:
    def __init__(self, process, key):
        self.process, self.key = process, key
        self.lock = asyncio.Lock()
        self.lost = False
        self.last_used = time.monotonic()
        self.started = False
        self.buffers = {"stdout": b"", "stderr": b""}

    async def capture(self, stream, path, start, end):
        reader = getattr(self.process, stream)
        buffer = self.buffers[stream]
        active = False
        with open_output(path) as out:
            while True:
                if not active:
                    position = buffer.find(start)
                    if position >= 0:
                        buffer, active = buffer[position + len(start) :], True
                        self.started = True
                    elif len(buffer) >= len(start):
                        buffer = buffer[-(len(start) - 1) :]
                if active:
                    position = buffer.find(end)
                    if position >= 0:
                        suffix = buffer[position + len(end) :]
                        terminal = suffix.find(b"\x1f")
                        if terminal >= 0:
                            out.write(buffer[:position])
                            self.buffers[stream] = suffix[terminal + 1 :]
                            return int(suffix[:terminal])
                    keep = len(end) + 16
                    if len(buffer) > keep:
                        out.write(buffer[:-keep])
                        buffer = buffer[-keep:]
                try:
                    chunk = await reader.read(65536)
                except BaseException:
                    if active:
                        out.write(buffer)
                    raise
                if not chunk:
                    if active:
                        out.write(buffer)
                    self.lost = True
                    raise ConnectionError(
                        "persistent shell exited before completion; session state was lost"
                    )
                buffer += chunk


class NativeSSHRunner:
    def __init__(self, cfg, settings, *, store):
        self.cfg, self.settings, self.store = cfg, settings, store
        self._sem = asyncio.Semaphore(cfg.max_parallel)
        self._connect_lock = asyncio.Lock()
        self._connections = {}
        self._shells = {}
        self._shells_lock = asyncio.Lock()
        self._last_login = 0.0
        self._failed_until = {}
        self._registry = self.store.root.parent / "shell_sessions.json" if self.store else None
        self._known_sessions = set()
        self.result_context = contextvars.ContextVar("execution_results", default=None)
        if self._registry and self._registry.exists():
            self._known_sessions = {
                (tuple(row["target"]), row["session_id"])
                for row in json.loads(self._registry.read_text())
            }

    def save_sessions(self):
        if self._registry:
            atomic_json(
                self._registry,
                [
                    {"target": list(key[0]), "session_id": key[1]}
                    for key in sorted(self._known_sessions)
                ],
            )

    def record_result(self, result_id, host):
        records = self.result_context.get()
        if records is not None:
            records.append({"host": host.name, "result_id": result_id})

    async def open_process(self, host, command):
        """Retry a transport failure once, only before any user script is sent."""
        previous = self._connections.get(self.key(host))
        connection = await self.connection(host)
        reconnected = previous is not None and previous is not connection
        try:
            return await connection.create_process(command, encoding=None), reconnected
        except (
            asyncssh.ConnectionLost,
            asyncssh.DisconnectError,
            BrokenPipeError,
            ConnectionResetError,
        ):
            connection.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(connection.wait_closed(), 5)
            connection = await self.connection(host)
            return await connection.create_process(command, encoding=None), True

    def connection_status(self):
        return {
            "connected": sum(not c.is_closed() for c in self._connections.values()),
            "disconnected": sum(c.is_closed() for c in self._connections.values()),
        }

    def key(self, host):
        if not host.asset_id or not host.account or not host.login:
            raise ConfigError("target needs a permitted REST asset ID and account")
        return (host.login, host.account, str(uuid.UUID(host.asset_id)))

    async def connection(self, host):
        key = self.key(host)
        connection = self._connections.get(key)
        if connection and not connection.is_closed():
            return connection
        async with self._connect_lock:
            connection = self._connections.get(key)
            if connection and not connection.is_closed():
                return connection
            if time.monotonic() < self._failed_until.get(key, 0):
                raise ConfigError(
                    "previous SSH authentication failed; waiting before another attempt"
                )
            await asyncio.sleep(max(0, 5 - (time.monotonic() - self._last_login)))
            cfg = self.cfg
            try:
                options = {
                    "port": cfg.ssh_port,
                    "username": host.login_string(cfg.login_user),
                    "preferred_auth": ["publickey", "password"],
                    "config": None,
                    "keepalive_interval": 30,
                    "keepalive_count_max": 3,
                    "connect_timeout": 30,
                    "login_timeout": 60,
                    "encoding": None,
                }
                if cfg.ssh_key:
                    options["client_keys"] = [str(cfg.ssh_key)]
                if cfg.ssh_password:
                    options["password"] = cfg.ssh_password
                if self.settings.known_hosts:
                    options["known_hosts"] = str(self.settings.known_hosts)
                if self.settings.host_key_alias:
                    options["host_key_alias"] = self.settings.host_key_alias
                if self.settings.ssh_proxy_command:
                    options["proxy_command"] = self.settings.ssh_proxy_command
                connection = await asyncssh.connect(self.settings.gateway_host(cfg), **options)
            except Exception as exc:
                self._failed_until[key] = time.monotonic() + 60
                raise ConfigError(
                    f"SSH connection failed ({type(exc).__name__}); no automatic login retry"
                ) from None
            finally:
                self._last_login = time.monotonic()
            self._connections[key] = connection
            return connection

    def command(self, container=None, image=None, docker_args=None, persistent=False):
        if container and image:
            raise ValueError("choose container or image, not both")
        shell = ["bash", "--noprofile", "--norc"] if persistent else ["bash", "-s"]
        if container or image:
            target = container or image
            if not target or target.startswith("-") or "\x00" in target:
                raise ValueError("invalid Docker target")
            if container:
                args = ["docker", "exec", "-i", *(docker_args or []), target, *shell]
            else:
                args = [
                    "docker",
                    "run",
                    "--rm",
                    "-i",
                    *(docker_args or []),
                    "--entrypoint",
                    "bash",
                    target,
                    *(["--noprofile", "--norc"] if persistent else ["-s"]),
                ]
            return shlex.join(args)
        if docker_args:
            raise ValueError("docker_args requires container or image")
        return shlex.join(shell)

    async def execute_to_store(
        self,
        host,
        script,
        timeout,
        *,
        cwd=None,
        env=None,
        container=None,
        image=None,
        docker_args=None,
        session_id=None,
        wrapped=False,
    ):
        if env and any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in env):
            raise ValueError("invalid environment variable name")
        if any(
            "\x00" in x for x in [script, cwd or "", *(env or {}).values(), *(docker_args or [])]
        ):
            raise ValueError("NUL bytes are not valid shell arguments")
        if session_id and not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", session_id):
            raise ValueError(
                "session_id must contain 1..64 letters, digits, dots, underscores, or hyphens"
            )
        command = self.command(container, image, docker_args, bool(session_id))
        result_id = self.store.create(
            kind="run",
            host=host.name,
            asset_id=host.asset_id,
            container=container,
            image=image,
            session_id=session_id,
        )
        self.record_result(result_id, host)
        paths = {s: self.store.path(result_id, s) for s in ("stdout", "stderr")}
        started_at = time.monotonic()
        prefix = [] if session_id or wrapped else ["set -euo pipefail"]
        prefix += ["export " + k + "=" + shlex.quote(v) for k, v in (env or {}).items()]
        body = "\n".join([*prefix, script]) + "\n"
        if cwd:
            body = (
                "if builtin cd -- " + shlex.quote(cwd) + "; then\n" + body + "\nelse\n false\nfi\n"
            )
        code, executed, timed_out, warning, process = 75, False, False, None, None
        sent, completed, reconnected, shell = False, False, False, None
        progress = {"started": False}
        async with self._sem:
            try:
                if session_id:
                    async with self._shells_lock:
                        key = (self.key(host), session_id)
                        shell = self._shells.get(key)
                        if shell is None and key in self._known_sessions:
                            raise ValueError(
                                "session state was lost after restart; use a new session_id "
                                "or call session_close to reset it"
                            )
                        options = (container, image, tuple(docker_args or []))
                        if shell and shell.key != options:
                            raise ValueError(
                                "session_id already uses a different container/image; use a new ID"
                            )
                        if shell and (shell.lost or shell.process.is_closing()):
                            raise ValueError(
                                "session state was lost; inspect prior result, then use a new "
                                "session_id"
                            )
                        if shell is None:
                            process, reconnected = await self.open_process(host, command)
                            self._known_sessions.add(key)
                            self.save_sessions()
                            shell = ShellSession(process, options)
                            self._shells[key] = shell
                    async with shell.lock:
                        if shell.lost or shell.process.is_closing():
                            raise ValueError(
                                "session state was lost; use a new session_id or session_close "
                                "to reset it"
                            )
                        shell.started = False
                        nonce = uuid.uuid4().hex.encode()
                        start, end = b"\x1eSTART" + nonce + b"\x1f", b"\x1eEND" + nonce + b":"
                        # The caller supplies literal Bash; quote it once for the retained shell.
                        frame_start = f"printf '\\036START{nonce.decode()}\\037'"
                        frame_end = f"printf '\\036END{nonce.decode()}:%s\\037' \"$__jms_rc\""
                        packet = (
                            f"{frame_start}; {frame_start} >&2\n"
                            + "builtin eval -- "
                            + shlex.quote(body)
                            + "\n"
                            + f"__jms_rc=$?; {frame_end}; {frame_end} >&2\n"
                        ).encode()
                        try:
                            sent = True
                            shell.process.stdin.write(packet)
                            await shell.process.stdin.drain()
                            captures = [
                                asyncio.create_task(shell.capture(s, paths[s], start, end))
                                for s in ("stdout", "stderr")
                            ]
                            try:
                                codes = await asyncio.wait_for(asyncio.gather(*captures), timeout)
                            finally:
                                for task in captures:
                                    if not task.done():
                                        task.cancel()
                                await asyncio.gather(*captures, return_exceptions=True)
                            code, executed = codes[0], codes[0] == codes[1]
                            completed = executed
                            shell.last_used = time.monotonic()
                        except BaseException:
                            executed = shell.started
                            shell.lost = True
                            shell.process.close()
                            raise
                else:
                    marker = ("__JMS_EXEC_" + uuid.uuid4().hex + "__\n").encode()
                    process, reconnected = await self.open_process(host, command)
                    tasks = [
                        asyncio.create_task(
                            pump(process.stdout, paths["stdout"], marker, progress)
                        ),
                        asyncio.create_task(pump(process.stderr, paths["stderr"])),
                    ]
                    try:
                        sent = True
                        process.stdin.write(
                            b"printf '%s\\n' "
                            + shlex.quote(marker.decode().strip()).encode()
                            + b"\n"
                            + body.encode()
                        )
                        process.stdin.write_eof()
                        await asyncio.wait_for(process.wait_closed(), timeout)
                        await asyncio.gather(*tasks)
                        executed = progress["started"]
                        completed = process.exit_status is not None and process.exit_status >= 0
                    except BaseException:
                        process.close()
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(process.wait_closed(), 5)
                        raise
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        executed = progress["started"]
                    code = process.exit_status if process.exit_status is not None else 255
                    if not executed and code == 0:
                        code = 75
                    if not completed:
                        warning = (
                            "SSH channel ended without a completion status; remote completion is "
                            "unknown."
                        )
            except asyncio.TimeoutError:
                code, timed_out = 124, True
                warning = (
                    "Execution timed out; remote completion is unknown. Inspect before rerunning."
                )
            except asyncio.CancelledError:
                code, warning = 130, "Call cancelled; remote completion is unknown."
                raise
            except Exception as exc:
                warning = (
                    str(exc)
                    if isinstance(exc, (ValueError, ConfigError, ConnectionError))
                    else f"execution failed ({type(exc).__name__})"
                )
            finally:
                unknown = sent and not completed
                if unknown and warning and "unknown" not in warning:
                    warning += "; remote completion is unknown; do not replay automatically"
                for path in paths.values():
                    if not path.exists():
                        with open_output(path):
                            pass
                if warning:
                    with paths["stderr"].open("ab") as out:
                        out.write(("\n" + warning + "\n").encode())
                meta = self.store.finish(
                    result_id,
                    ok=code == 0 and executed,
                    exit_code=code,
                    timed_out=timed_out,
                    executed=True if executed else (None if unknown else False),
                    completion_unknown=unknown,
                    execution_state="completed"
                    if completed
                    else ("unknown" if unknown else "not_started"),
                    reconnected=reconnected,
                    connection_state="connected"
                    if (
                        self._connections.get(self.key(host))
                        and not self._connections[self.key(host)].is_closed()
                    )
                    else "disconnected",
                    session_state=(
                        "lost"
                        if shell is None or shell.lost or shell.process.is_closing()
                        else "active"
                    )
                    if session_id
                    else None,
                    elapsed_seconds=round(time.monotonic() - started_at, 3),
                    warning=warning,
                )
        return meta

    async def exec_on(self, host, command, *, timeout=60, workdir=True, env_setup=True):
        from .inventory import wrap_command

        script = wrap_command(
            host.env_setup if env_setup else None, host.workdir if workdir else None, command
        )
        meta = await self.execute_to_store(host, script, timeout, wrapped=True)
        result_id = meta["result_id"]
        result = ExecResult(
            host.name,
            meta["exit_code"],
            self.store.path(result_id, "stdout").read_text(errors="replace"),
            self.store.path(result_id, "stderr").read_text(errors="replace"),
            meta["timed_out"],
        )
        result.result_id = result_id
        return result

    async def exec_many(self, hosts, command, *, timeout=60):
        return list(
            await asyncio.gather(*(self.exec_on(host, command, timeout=timeout) for host in hosts))
        )

    async def close_session(self, host, session_id):
        key = (self.key(host), session_id)
        async with self._shells_lock:
            shell = self._shells.get(key)
            if shell and shell.lock.locked():
                raise ValueError("session is executing; wait for its result before closing")
            if shell:
                shell.process.close()
                await shell.process.wait_closed()
            existed = key in self._known_sessions
            self._shells.pop(key, None)
            self._known_sessions.discard(key)
            self.save_sessions()
        return {"ok": True, "closed": existed}

    def sessions(self):
        return [
            {
                "asset_id": key[0][2],
                "session_id": key[1],
                "lost": key not in self._shells
                or self._shells[key].lost
                or self._shells[key].process.is_closing(),
            }
            for key in sorted(self._known_sessions)
        ]

    async def close(self):
        for shell in self._shells.values():
            shell.process.close()
        for connection in self._connections.values():
            connection.close()
        await asyncio.gather(
            *(c.wait_closed() for c in self._connections.values()), return_exceptions=True
        )
        self._shells.clear()
        self._connections.clear()

    async def rsync(
        self, host, local, remote, *, direction="push", delete=False, excludes=None, timeout=900
    ):
        """Keep rsync semantics while its SSH stream uses our authenticated connection."""
        result_id = self.store.create(kind="transfer", host=host.name, asset_id=host.asset_id)
        self.record_result(result_id, host)
        out_path, err_path = (
            self.store.path(result_id, "stdout"),
            self.store.path(result_id, "stderr"),
        )
        started = time.monotonic()
        tasks, channels = set(), []
        code, timed_out = 75, False
        async with self._sem:
            try:
                connection = await self.connection(host)
                with tempfile.TemporaryDirectory(prefix="jms-rsync-") as directory:
                    socket_path = str(Path(directory) / "stream.sock")

                    async def handle(reader, writer):
                        task = asyncio.current_task()
                        tasks.add(task)
                        try:
                            line = await asyncio.wait_for(reader.readline(), 10)
                            args = json.loads(line)
                            if len(args) < 2 or args[0] != host.asset_id or args[1] != "rsync":
                                raise ValueError("invalid rsync transport request")
                            channel = await connection.create_process(
                                shlex.join(args[1:]), encoding=None
                            )
                            channels.append(channel)

                            async def incoming():
                                while data := await reader.read(65536):
                                    channel.stdin.write(data)
                                    await channel.stdin.drain()
                                channel.stdin.write_eof()

                            async def outgoing():
                                while data := await channel.stdout.read(65536):
                                    writer.write(data)
                                    await writer.drain()

                            async def errors():
                                with err_path.open("ab") as err:
                                    while data := await channel.stderr.read(65536):
                                        err.write(data)

                            inbound = asyncio.create_task(incoming())
                            try:
                                await asyncio.gather(outgoing(), errors(), channel.wait_closed())
                            finally:
                                inbound.cancel()
                                await asyncio.gather(inbound, return_exceptions=True)
                        except Exception as exc:
                            with err_path.open("ab") as err:
                                err.write(
                                    f"Transport stream failed ({type(exc).__name__})\n".encode()
                                )
                        finally:
                            writer.close()
                            with contextlib.suppress(Exception):
                                await writer.wait_closed()
                            tasks.discard(task)

                    with (
                        open_output(out_path) as out,
                        os.fdopen(
                            os.open(
                                err_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600
                            ),
                            "ab",
                        ) as err,
                    ):
                        server = await asyncio.start_unix_server(handle, path=socket_path)
                        os.chmod(socket_path, 0o600)
                        async with server:
                            bridge = [
                                sys.executable,
                                str(Path(__file__).with_name("rsync_relay.py")),
                                socket_path,
                            ]
                            argv = ["rsync", "-azs", "-e", shlex.join(bridge)]
                            if delete:
                                argv.append("--delete")
                            for pattern in excludes or []:
                                argv += ["--exclude", pattern]
                            spec = f"{host.asset_id}:{remote}"
                            argv += (
                                ["--", local, spec] if direction == "push" else ["--", spec, local]
                            )
                            proc = await asyncio.create_subprocess_exec(
                                *argv, stdout=out, stderr=err, start_new_session=True
                            )
                            try:
                                await asyncio.wait_for(proc.wait(), timeout)
                                code = proc.returncode
                            except asyncio.TimeoutError:
                                timed_out, code = True, 124
                                from .ssh import _kill

                                await _kill(proc)
                            except BaseException:
                                from .ssh import _kill

                                await _kill(proc)
                                raise
                            finally:
                                for channel in channels:
                                    channel.close()
                                for task in list(tasks):
                                    task.cancel()
                                await asyncio.gather(*list(tasks), return_exceptions=True)
            except Exception as exc:
                for p in [out_path, err_path]:
                    if not p.exists():
                        with open_output(p):
                            pass
                with err_path.open("ab") as err:
                    err.write(f"Transfer failed ({type(exc).__name__})\n".encode())
            self.store.finish(
                result_id,
                ok=code == 0,
                exit_code=code,
                timed_out=timed_out,
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
            result = ExecResult(
                host.name,
                code,
                out_path.read_text(errors="replace"),
                err_path.read_text(errors="replace"),
                timed_out,
            )
            result.result_id = result_id
            return result
