import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import jsonschema

from jms_bridge.compact import BridgeService, validate_cached
from jms_bridge.config import BastionConfig
from jms_bridge.filesystem import SCHEMA
from jms_bridge.native_ssh import ConnectionTimings
from jms_bridge.runtime_config import BridgeSettings
from jms_bridge.scheduling import Timings


class ValidationAndMetrics(unittest.IsolatedAsyncioTestCase):
    async def test_cached_validator_preserves_best_match_and_only_validates_fs_once(self):
        with tempfile.TemporaryDirectory() as td:
            service = BridgeService(
                BastionConfig(url="https://bastion.example.com"), BridgeSettings(state_dir=Path(td))
            )
            for args in (
                {},
                {"target": 3, "action": "bad"},
                {"target": "x", "action": "read", "limit": 0},
            ):
                errors = []
                for fn in (
                    lambda args=args: jsonschema.validate(args, SCHEMA["inputSchema"]),
                    lambda args=args: validate_cached(service.fs_validator, args),
                ):
                    with self.assertRaises(jsonschema.ValidationError) as caught:
                        fn()
                    errors.append((list(caught.exception.path), caught.exception.message))
                self.assertEqual(*errors)
            service.filesystem.execute = AsyncMock(return_value={"ok": True})
            with patch("jms_bridge.compact.validate_cached", wraps=validate_cached) as checked:
                await service.call("jms_fs", {"target": "x", "action": "read", "path": "/a"})
                self.assertEqual(checked.call_count, 1)
            await service.filesystem.workers.close()
            await service.runner.close()

    async def test_connection_breakdown_has_no_auth_secrets(self):
        timing = Timings()
        auth = ConnectionTimings(timing, time.monotonic() - 0.02)
        auth.begin_auth("do-not-log-user")
        auth.auth_completed()
        self.assertGreater(timing.connection_snapshot()["proxy_and_handshake_seconds"], 0.019)
        self.assertIn("authentication_seconds", timing.connection_snapshot())
        self.assertNotIn("do-not-log-user", json.dumps(timing.connection_snapshot()))
