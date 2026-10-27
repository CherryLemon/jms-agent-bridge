"""Stdlib-only tests for the remote probe, runnable separately on Python 3.8.

The MCP server still requires Python 3.10+. Do not import the package or install
its server dependencies in this target-interpreter compatibility test.
"""

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "src/jms_bridge/fs_probe.py"
SPEC = importlib.util.spec_from_file_location("jms_bridge.fs_probe", SOURCE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class RemoteFilesystemCompatibility(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root / "example.py"
        self.file.write_text("before\nneedle 汉🙂\nafter\n", encoding="utf-8")

    def test_directory_actions_without_ripgrep(self):
        with patch.object(probe.shutil, "which", return_value=None):
            for action in ("list", "tree", "glob"):
                with self.subTest(action=action):
                    options = dict(action=action, path=str(self.root))
                    if action == "glob":
                        options["pattern"] = "*.py"
                    result = probe.execute(options)
                    self.assertTrue(result["ok"])
                    self.assertEqual(result["action"], action)
                    self.assertEqual(result["entries"][0]["path"], str(self.file))

    def test_python_search_file_and_directory_preserves_context_rows(self):
        with patch.object(probe.shutil, "which", return_value=None):
            for path in (self.file, self.root):
                with self.subTest(path=str(path)):
                    result = probe.execute(
                        dict(
                            action="search",
                            path=str(path),
                            pattern="needle",
                            context_lines=1,
                        )
                    )
                    self.assertTrue(result["ok"])
                    self.assertEqual(
                        [row["kind"] for row in result["entries"]],
                        ["context", "match", "context"],
                    )
                    self.assertEqual(result["entries"][1]["text"], "needle 汉🙂")

    def test_file_actions(self):
        read = probe.execute(dict(action="read", path=str(self.file), line_count=1))
        self.assertEqual(read["text"], "before\n")
        batch = probe.execute(dict(action="read_many", paths=[str(self.file)]))
        self.assertTrue(batch["ok"])
        self.assertEqual(probe.execute(dict(action="stat", path=str(self.file)))["type"], "file")
        digest = probe.execute(dict(action="checksum", path=str(self.file)))["digest"]
        self.assertEqual(digest, hashlib.sha256(self.file.read_bytes()).hexdigest())

    def test_ripgrep_paths_strip_exactly_one_dot_prefix(self):
        event = dict(
            type="match",
            data=dict(
                path=dict(text="./..example.py"),
                lines=dict(text="needle\n"),
                line_number=2,
            ),
        )

        class Process:
            def __init__(self, data):
                self.stdout = io.BytesIO(data)

            def poll(self):
                return 0

            def wait(self):
                return 0

        for action, data in (
            ("glob", b"./..example.py\0"),
            ("search", json.dumps(event).encode() + b"\n"),
        ):
            with ExitStack() as stack:
                stack.enter_context(self.subTest(action=action))
                stack.enter_context(patch.object(probe.shutil, "which", return_value="/usr/bin/rg"))
                stack.enter_context(
                    patch.object(probe.subprocess, "Popen", return_value=Process(data))
                )
                result = probe.execute(
                    dict(
                        action=action,
                        path=str(self.root),
                        pattern="*.py" if action == "glob" else "needle",
                    )
                )
                self.assertTrue(result["ok"])
                self.assertEqual(result["entries"][0]["relative_path"], "..example.py")

    def test_isolated_stdin_protocol(self):
        options = dict(action="list", path=str(self.root))
        script = SOURCE.read_text() + "\nmain(json.loads(" + repr(json.dumps(options)) + "))\n"
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-"],
            input=script,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertTrue(json.loads(result.stdout)["ok"])
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
