"""Actual shell and rsync processes continue while one identity is recovered."""

import asyncio
import shlex
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from test_execution_scheduling import LocalConnection

from jms_bridge.config import BastionConfig
from jms_bridge.filesystem import Filesystem
from jms_bridge.inventory import Host
from jms_bridge.native_ssh import NativeSSHRunner
from jms_bridge.results import ResultStore
from jms_bridge.runtime_config import BridgeSettings


class TrackedConnection(LocalConnection):
    closed = False
    fail_startup = False
    delay_transfer = False

    def is_closed(self):
        return self.closed

    def close(self):
        self.closed = True
        super().close()

    async def create_process(self, command, **kwargs):
        if self.fail_startup and "--noprofile" in command:
            command = "sleep 0.01"
        if self.delay_transfer and command.startswith("rsync "):
            command = shlex.join(
                ["bash", "-c", 'sleep 0.15; exec "$@"', "bash", *shlex.split(command)]
            )
        return await super().create_process(command, **kwargs)


class TargetRecovery(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ResultStore(self.root / "results")
        self.runner = NativeSSHRunner(
            BastionConfig(url="https://bastion.example.com"),
            BridgeSettings(state_dir=self.root),
            store=self.store,
        )
        self.a, self.b = [
            Host(name=name, asset_id=str(uuid.uuid4()), login="test", account="root")
            for name in ("a", "b")
        ]
        self.old, self.other = TrackedConnection(), TrackedConnection()
        self.connections = [self.old, self.other]
        self.runner._connections.update(
            {self.runner.key(self.a): self.old, self.runner.key(self.b): self.other}
        )

        async def connect(host):
            key = self.runner.key(host)
            connection = self.runner._connections.get(key)
            if connection is None or connection.is_closed():
                connection = TrackedConnection()
                self.connections.append(connection)
                self.runner._connections[key] = connection
            return connection

        self.runner._connection = connect
        self.fs = Filesystem(
            SimpleNamespace(
                ensure=AsyncMock(
                    return_value=SimpleNamespace(
                        resolve=lambda targets: [self.a if targets[0] == "a" else self.b]
                    )
                )
            ),
            self.runner,
            self.store,
        )
        self.path = self.root / "data"
        self.path.write_text("payload\n")

    async def asyncTearDown(self):
        await self.fs.workers.close()
        await self.runner.close()
        for connection in self.connections:
            connection.close()
            await connection.wait_closed()
        self.temp.cleanup()

    async def query(self):
        return await self.fs.execute(target="a", action="read", path=str(self.path))

    async def wait_active(self, host):
        async def waiting():
            while not self.runner._connection_recovery.state(self.runner.key(host)).active:
                await asyncio.sleep(0.001)

        await asyncio.wait_for(waiting(), 2)

    async def test_two_failed_channels_replace_only_target_connection(self):
        self.old.fail_startup = True
        result = await self.query()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["data"]["text"], "payload\n")
        self.assertEqual(result["connection_recovery"]["status"], "reset")
        self.assertTrue(self.old.closed)
        self.assertFalse(self.other.closed)
        self.assertEqual(len(self.old.processes), 2)
        self.assertEqual(self.runner._connection_recovery.resets, 1)

    async def test_same_host_command_finishes_and_other_host_keeps_running(self):
        active = asyncio.create_task(
            self.runner.execute_to_store(self.a, "sleep 0.2; printf ORIGINAL_FINISHED", 2)
        )
        await self.wait_active(self.a)
        self.old.fail_startup = True
        query = asyncio.create_task(self.query())
        while self.runner._connection_recovery.state(self.runner.key(self.a)).phase != "draining":
            await asyncio.sleep(0.001)
        other = await self.runner.execute_to_store(self.b, "printf OTHER_FINISHED", 2)
        self.assertTrue(other["ok"])
        self.assertFalse(self.old.closed)
        original = await active
        self.assertTrue(original["ok"])
        self.assertEqual(
            self.store.path(original["result_id"], "stdout").read_text(), "ORIGINAL_FINISHED"
        )
        self.assertTrue((await query)["ok"])
        self.assertFalse(self.other.closed)

    async def test_same_host_rsync_completes_before_connection_is_closed(self):
        destination = self.root / "downloaded file"
        self.old.delay_transfer = True
        transfer = asyncio.create_task(
            self.runner.rsync(self.a, str(destination), str(self.path), direction="pull")
        )
        await self.wait_active(self.a)
        self.old.fail_startup = True
        result = await self.query()
        receipt = await transfer
        self.assertTrue(self.store.metadata(receipt.result_id)["ok"])
        self.assertEqual(destination.read_bytes(), self.path.read_bytes())
        self.assertTrue(result["ok"])
        self.assertTrue(self.old.closed)

    async def test_public_shell_state_is_preserved_then_recovery_resumes_after_close(self):
        await self.runner.execute_to_store(
            self.a, "export USER_STATE=retained", 2, session_id="keep"
        )
        self.old.fail_startup = True
        result = await self.query()
        self.assertEqual(
            result["connection_recovery"], dict(status="deferred", reason="persistent_shells")
        )
        self.assertFalse(self.old.closed)
        check = await self.runner.execute_to_store(
            self.a, 'printf "%s" "$USER_STATE"', 2, session_id="keep"
        )
        self.assertTrue(check["ok"])
        self.assertEqual(self.store.path(check["result_id"], "stdout").read_text(), "retained")
        await self.runner.close_session(self.a, "keep")
        await self.runner._connection_recovery.state(self.runner.key(self.a)).task
        self.assertTrue(self.old.closed)
        self.assertTrue((await self.query())["ok"])

    async def test_simultaneous_query_failures_share_one_connection_replacement(self):
        self.old.fail_startup = True
        first, second = await asyncio.gather(self.query(), self.query())
        self.assertTrue(first["ok"] and second["ok"], (first, second))
        self.assertEqual(self.runner._connection_recovery.resets, 1)
        self.assertEqual(len(self.connections), 3)

    async def test_persistently_bad_target_has_three_attempts_then_cools_down(self):
        original = self.runner._connection

        async def unavailable(host):
            connection = await original(host)
            connection.fail_startup = True
            return connection

        self.runner._connection = unavailable
        first = await self.query()
        self.assertFalse(first["ok"])
        self.assertEqual(first["connection_recovery"]["status"], "reset")
        self.assertEqual(first["execution_state"], "not_started")
        self.assertFalse(first["completion_unknown"])
        self.assertEqual(sum(len(c.processes) for c in self.connections), 3)
        second = await self.query()
        self.assertFalse(second["ok"])
        self.assertEqual(second["connection_recovery"]["status"], "cooldown")
        self.assertEqual(self.runner._connection_recovery.resets, 1)
        self.assertEqual(len(self.connections), 3)

    async def test_queued_operation_has_not_started_if_recovery_wait_times_out(self):
        await self.runner.connection(self.a)
        manager = self.runner._connection_recovery
        state = manager.state(self.runner.key(self.a))
        state.phase = "draining"
        self.runner.queue_timeout = 0.02
        result = await self.runner.execute_to_store(self.a, "printf MUST_NOT_RUN", 2)
        self.assertTrue(result["queue_timed_out"])
        self.assertEqual(result["failure_stage"], "connection_recovery")
        self.assertEqual(result["execution_state"], "not_started")
        self.assertFalse(result["completion_unknown"])
        self.assertEqual(len(self.old.processes), 0)
        state.phase = "idle"
        manager.notify(self.runner.key(self.a))
