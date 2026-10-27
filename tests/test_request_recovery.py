import asyncio
import json
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from jms_bridge.async_io import offload
from jms_bridge.compact import BridgeService
from jms_bridge.config import BastionConfig, ConfigError
from jms_bridge.inventory import Host
from jms_bridge.recoverable_jobs import RecoverableJobManager
from jms_bridge.request_journal import RequestJournal
from jms_bridge.results import ResultStore
from jms_bridge.runtime_config import BridgeSettings, load_settings
from jms_bridge.ssh import ExecResult


class DurableIO(unittest.IsolatedAsyncioTestCase):
    async def test_retention_status_does_not_wait_for_durable_storage_lock(self):
        with tempfile.TemporaryDirectory() as td:
            store = ResultStore(td)
            entered, release = threading.Event(), threading.Event()

            def hold_io_lock():
                with store._lock:
                    entered.set()
                    release.wait(2)

            task = asyncio.create_task(asyncio.to_thread(hold_io_lock))
            while not entered.is_set():
                await asyncio.sleep(0.001)
            try:
                result = await asyncio.wait_for(asyncio.to_thread(store.retention_status), 0.1)
                self.assertEqual(result["active_results"], 0)
            finally:
                release.set()
                await task

    async def test_cancelled_durable_claim_never_executes_or_replays(self):
        with tempfile.TemporaryDirectory() as td:
            journal = RequestJournal(td)
            original = journal.claim
            entered = threading.Event()

            def slow_claim(*args):
                entered.set()
                time.sleep(0.05)
                return original(*args)

            work = AsyncMock(return_value={"ok": True})
            with patch.object(journal, "claim", side_effect=slow_claim):
                task = asyncio.create_task(journal.execute("claim-cancel", "test", {}, work))
                while not entered.is_set():
                    await asyncio.sleep(0.001)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            status = await journal.execute("claim-cancel", "test", {}, work)
            self.assertTrue(status["duplicate"])
            self.assertEqual(status["request_state"], "interrupted")
            self.assertFalse(status["execution_replayed"])
            work.assert_not_called()

    async def test_async_submission_receipt_is_durable_before_remote_execution(self):
        host = Host(name="test", asset_id=str(uuid.uuid4()), login="test", account="root")
        rows = []

        async def record(row):
            await asyncio.sleep(0.001)
            rows.append(dict(row))

        async def execute(*args, **kwargs):
            self.assertEqual(rows[0]["submission_state"], "prepared")
            self.assertFalse(rows[0]["launch_acknowledged"])
            return ExecResult(host.name, 0, "LAUNCHER=tmux\n", "")

        runner = SimpleNamespace(exec_on=AsyncMock(side_effect=execute))
        manager = RecoverableJobManager(runner, record)
        handle, result = await manager.submit(host, "printf test")
        self.assertTrue(result.ok)
        self.assertEqual(handle.job_id, rows[0]["job_id"])
        self.assertEqual(rows[-1]["submission_state"], "acknowledged")
        self.assertTrue(rows[-1]["launch_acknowledged"])
        runner.exec_on.assert_awaited_once()

    async def test_cancelled_create_finishes_disk_work_and_does_not_leak_active_result(self):
        with tempfile.TemporaryDirectory() as td:
            store = ResultStore(td)
            original = store.create

            def slow_create(**kw):
                time.sleep(0.07)
                return original(**kw)

            with patch.object(store, "create", side_effect=slow_create):
                task = asyncio.create_task(store.acreate())
                ticks = 0
                for _ in range(4):
                    await asyncio.sleep(0.01)
                    ticks += 1
                    task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertEqual(ticks, 4)
            self.assertEqual(store.retention_status()["active_results"], 0)
            meta = json.loads(next(Path(td).glob("*/meta.json")).read_text())
            self.assertEqual(meta["execution_state"], "not_started")

    async def test_concurrent_claim_lookup_and_progress_do_not_replay_or_lose_receipts(self):
        with tempfile.TemporaryDirectory() as td:
            journal = RequestJournal(td)
            original = journal.save
            performed = 0

            def slow_save(receipt):
                time.sleep(0.003)
                return original(receipt)

            async def work():
                nonlocal performed
                performed += 1
                receipts = journal.receipts("execution_results")
                await asyncio.gather(
                    *(
                        offload(receipts.append, {"host": str(i), "result_id": str(i)})
                        for i in range(8)
                    )
                )
                return {"ok": True}

            with patch.object(journal, "save", side_effect=slow_save):
                first = asyncio.create_task(journal.execute("same", "test", {}, work))
                await asyncio.sleep(0.001)
                second = asyncio.create_task(journal.execute("same", "test", {}, work))
                status = await journal.astatus("same")
                self.assertNotEqual(status["request_state"], "interrupted")
                a, b = await asyncio.gather(first, second)
            self.assertEqual(performed, 1)
            self.assertTrue(b["duplicate"])
            self.assertEqual(len(a["execution_results"]), 8)
            self.assertFalse(a["execution_replayed"])

    async def test_cancelled_caller_keeps_worker_owned_until_durable_finish(self):
        with tempfile.TemporaryDirectory() as td:
            journal = RequestJournal(td)
            started, release = asyncio.Event(), asyncio.Event()

            async def work():
                started.set()
                await release.wait()
                return {"ok": True, "job_id": "only-job"}

            task = asyncio.create_task(journal.execute("job", "submit", {}, work))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual((await journal.astatus("job"))["request_state"], "running")
            release.set()
            await journal.drain()
            self.assertEqual((await journal.astatus("job"))["job_id"], "only-job")
            self.assertEqual(journal.tasks, {})


async def test_run_recovery_is_exposed_and_survives_service_restart(tmp_path):
    cfg = BastionConfig(url="https://bastion.example.com", login_user="example-user")
    settings = BridgeSettings(state_dir=tmp_path)
    service = BridgeService(cfg, settings)
    host = Host(name="node", asset_id=str(uuid.uuid4()), login="example-user", account="root")
    service.catalog.ensure = AsyncMock(return_value=SimpleNamespace(resolve=lambda _: [host]))
    calls = 0

    async def execute(*args, **kwargs):
        nonlocal calls
        calls += 1
        rid = service.store.create(host=host.name, session_id=None)
        service.runner.record_result(rid, host)
        service.store.path(rid, "stdout").write_text("only-once")
        return service.store.finish(
            rid,
            ok=True,
            exit_code=0,
            executed=True,
            timed_out=False,
            elapsed_seconds=0,
            execution_state="completed",
            completion_unknown=False,
            connection_state="connected",
            session_state=None,
            reconnected=False,
            timing={},
            queue_timed_out=False,
            failure_stage=None,
        )

    service.runner.execute_to_store = execute
    first = await service.run(["node"], script="printf only-once", request_id="same-id")
    duplicate = await service.run(["node"], script="printf only-once", request_id="same-id")
    assert first["results"][0]["stdout"] == "only-once"
    assert duplicate["duplicate"] and calls == 1
    assert first["execution_results"][0]["result_id"] == first["results"][0]["result_id"]
    restarted = BridgeService(cfg, settings)
    saved = await restarted.call("request_status", {"request_id": "same-id"})
    assert saved["results"][0]["result_id"] == first["results"][0]["result_id"]
    with pytest.raises(ValueError, match="already claimed"):
        await service.run(["node"], script="different", request_id="same-id")
    tools = {t.name: t for t in await service.server.list_tools()}
    assert "request_id" in tools["run"].input_schema["properties"]
    assert (
        "does not grant authorization"
        in tools["run"].input_schema["properties"]["allow_dangerous"]["description"]
    )
    await service.runner.close()
    await restarted.runner.close()


@pytest.mark.parametrize(
    "setting",
    [
        "execution_limit=0",
        "execution_limit=true",
        "inspection_reserved=-1",
        "execution_limit=2\ninspection_reserved=2",
        "queue_timeout=0",
        "result_ttl_seconds=-1",
        "result_cleanup_interval_seconds=true",
    ],
)
def test_runtime_configuration_rejects_invalid_values(setting, tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("[bridge]\n" + setting + "\n")
    monkeypatch.setenv("JMS_BRIDGE_CONFIG", str(config))
    with pytest.raises(ConfigError):
        load_settings()
