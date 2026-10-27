import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from jms_bridge.request_journal import RequestJournal
from jms_bridge.results import ResultStore, atomic_json


class Retention(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ResultStore(
            self.root / "results", ttl_seconds=10, cleanup_interval_seconds=0.02
        )

    def old_result(self, finished=True):
        rid = self.store.create()
        self.store.path(rid, "stdout").write_text("old output")
        if finished:
            self.store.finish(rid)
        meta = self.store.metadata(rid)
        meta["created_at"] = time.time() - 100
        if finished:
            meta["finished_at"] = time.time() - 100
        atomic_json(self.store.directory(rid) / "meta.json", meta)
        return rid

    async def test_ttl_protects_live_and_fresh_retains_receipt_no_replay(self):
        old = self.old_result()
        live = self.old_result(finished=False)
        fresh = self.store.save_json({"ok": True})
        journal = RequestJournal(self.root / "requests")
        perform = AsyncMock(return_value={"ok": True, "result_id": old})
        await journal.execute("a", "run", {}, perform)
        report = self.store.cleanup()
        self.assertEqual(report["removed"], 1)
        self.assertTrue(self.store.directory(live).exists())
        self.assertTrue(self.store.directory(fresh).exists())
        with self.assertRaisesRegex(ValueError, "expired"):
            self.store.read(old)
        duplicate = await journal.execute("a", "run", {}, perform)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(perform.await_count, 1)

    async def test_restart_abandoned_results_and_symlink_safety(self):
        orphan = self.old_result(finished=False)
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep").write_text("keep")
        (self.store.root / ("a" * 32)).symlink_to(outside)
        old = self.old_result()
        (self.store.directory(old) / "external").symlink_to(outside)
        restarted = ResultStore(self.store.root, ttl_seconds=10)
        report = restarted.cleanup()
        self.assertEqual(report["abandoned_removed"], 1)
        self.assertFalse(restarted.directory(orphan).exists())
        self.assertEqual((outside / "keep").read_text(), "keep")
        self.assertTrue((self.store.root / ("a" * 32)).is_symlink())

    async def test_background_cleanup_runs_idle_and_stops(self):
        rid = self.old_result()
        await self.store.start_retention()
        for _ in range(100):
            if not self.store.directory(rid).exists():
                break
            await asyncio.sleep(0.01)
        await self.store.close_retention()
        self.assertFalse(self.store.directory(rid).exists())
        self.assertEqual(self.store.retention_status()["last_cleanup"]["removed"], 1)

    async def test_bad_metadata_and_configuration_validation(self):
        rid = self.old_result()
        atomic_json(
            self.store.directory(rid) / "meta.json",
            {"status": "complete", "finished_at": float("nan")},
        )
        self.assertEqual(self.store.cleanup()["invalid_skipped"], 1)
        self.assertTrue(self.store.directory(rid).exists())
        with self.assertRaises(ValueError):
            ResultStore(self.root / "bad", ttl_seconds=-1)
