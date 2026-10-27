"""Boundary tests run the actual collector, without a target Python interpreter."""

import base64
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fs_harness import execute, without_rg

from jms_bridge import bash_fs


class BashBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "data.txt"
        self.path.write_text("before\nneedle 123\nafter\n")

    def query(self, action, **options):
        return execute(dict(action=action, path=str(self.root), **options))

    def test_all_eight_actions_without_python_or_rg(self):
        with without_rg():
            for action in (
                "list",
                "tree",
                "glob",
                "search",
                "read",
                "read_many",
                "stat",
                "checksum",
            ):
                with self.subTest(action=action):
                    options = dict(action=action, path=str(self.path))
                    if action in ("list", "tree", "glob", "search"):
                        options["path"] = str(self.root)
                    if action == "read_many":
                        options = dict(action=action, paths=[str(self.path)])
                    if action in ("glob", "search"):
                        options["pattern"] = "*.txt" if action == "glob" else "needle"
                    value = execute(options)
                    self.assertTrue(value["ok"], value)
                    self.assertEqual(value["backend"], "bash")

    def test_delimiter_newline_and_metacharacters_are_data(self):
        name = "__JMS_FS_BASH__\n'\" $(touch BAD) `touch BAD` @(+|x)\\.txt"
        path = self.root / name
        path.write_text("__JMS_FS_BASH__\nneedle\n")
        with without_rg():
            found = self.query("glob", pattern=name)
            self.assertEqual(found["entries"][0]["path"], str(path))
            found = self.query("search", pattern="needle")
            self.assertIn(str(path), [row["path"] for row in found["entries"]])
            value = execute(dict(action="read", path=str(path)))
            self.assertEqual(value["text"], path.read_text())
        self.assertFalse((self.root / "BAD").exists())

    def test_symlink_with_trailing_newline_keeps_target(self):
        name = "data\n"
        (self.root / name).write_text("ok\n")
        link = self.root / "link"
        link.symlink_to(name)
        value = execute(dict(action="stat", path=str(link)))
        self.assertEqual(value["link_target"], name)
        self.assertEqual(value["resolved_path"], str(self.root / name))

    def test_non_utf8_filename_metadata_is_lossless(self):
        path = os.fsencode(self.root) + b"/invalid-\xff"
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
        value = self.query("list")
        row = next(row for row in value["entries"] if "path_bytes_b64" in row)
        self.assertEqual(base64.b64decode(row["path_bytes_b64"]), path)

    def test_missing_utility_fails_only_actions_which_need_it(self):
        with without_rg() as commands:
            (commands / "sha256sum").unlink()
            value = execute(dict(action="checksum", path=str(self.path)))
            self.assertEqual(value["error"], "UnsupportedOperation")
            self.assertIn("sha256sum", value["message"])
            self.assertTrue(execute(dict(action="read", path=str(self.path)))["ok"])
            (commands / "timeout").unlink()
            self.assertEqual(self.query("list")["error"], "UnsupportedOperation")

    def test_grep_invalid_and_unsupported_patterns(self):
        with without_rg():
            value = self.query("search", pattern="[", literal=False)
            self.assertFalse(value["complete"])
            self.assertEqual(value["errors"][0]["error"], "grep_failed")
            value = self.query("search", pattern=r"\d+", literal=False)
            self.assertEqual(value["errors"][0]["error"], "UnsupportedPattern")
            value = self.query("search", pattern="[[:digit:]]+", literal=False)
            self.assertEqual(value["entries"][0]["line"], 2)

    def test_failed_search_input_is_not_an_exhaustive_empty_result(self):
        with without_rg() as commands:
            (commands / "head").unlink()
            (commands / "head").write_text("#!/bin/bash\nexit 2\n")
            (commands / "head").chmod(0o755)
            value = self.query("search", pattern="needle")
        self.assertFalse(value["ok"])
        self.assertFalse(value["complete"])
        self.assertEqual(value["errors"][0]["error"], "OSError")

    def test_wire_bound_and_unfinished_records_are_never_complete(self):
        with patch.object(bash_fs, "WIRE_LIMIT", 30):
            value = self.query("tree")
        self.assertFalse(value["complete"])
        self.assertTrue(value["scan_limited"])
        raw = b"JMSFS-BASH1\n" + base64.b64encode(b"H\0bash\000\0\0") + b"\nJMSFS-END 0\n"
        value = bash_fs.decode(raw, dict(action="list", path="/"))
        self.assertFalse(value["complete"])
        value = bash_fs.decode(raw, dict(action="read", path="/"))
        self.assertFalse(value["ok"])
        value = bash_fs.decode(b"JMSFS-BASH1\n", dict(action="list", path="/"))
        self.assertEqual(value["error"], "InvalidProbeResponse")

    def test_grep_context_continuation_counts_rows(self):
        with without_rg():
            first = self.query("search", pattern="needle", context_lines=1, limit=2)
            second = self.query(
                "search", pattern="needle", context_lines=1, offset=first["next_offset"]
            )
        rows = first["entries"] + second["entries"]
        self.assertEqual([row["line"] for row in rows], [1, 2, 3])
        self.assertTrue(second["complete"])

    def test_parent_errexit_and_pipefail_do_not_swallow_timeout_status(self):
        with without_rg() as commands:
            (commands / "grep").unlink()
            (commands / "grep").write_text("#!/bin/bash\nsleep 5\n")
            (commands / "grep").chmod(0o755)
            options = dict(action="search", path=str(self.root), pattern="needle", timeout=1)
            result = subprocess.run(
                ["bash", "-s"],
                input="set -euo pipefail\n" + bash_fs.remote_script(options),
                text=True,
                capture_output=True,
                timeout=5,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        value = bash_fs.decode(result.stdout.encode(), options)
        self.assertTrue(value["timed_out"])
        self.assertFalse(value["complete"])
