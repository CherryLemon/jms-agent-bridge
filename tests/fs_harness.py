"""Run the actual Bash wire protocol against disposable local filesystem fixtures."""

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from jms_bridge.bash_fs import decode, remote_script

TOOLS = (
    "bash",
    "timeout",
    "base64",
    "stat",
    "find",
    "sort",
    "readlink",
    "grep",
    "head",
    "dd",
    "od",
    "awk",
    "wc",
    "tee",
    "sha256sum",
    "sleep",
    "cat",
    "chmod",
)


def execute(options):
    result = subprocess.run(
        ["bash", "-s"],
        input=remote_script(options).encode(),
        capture_output=True,
        timeout=options.get("timeout", 30) + 3,
    )
    if result.returncode:
        raise AssertionError(result.stderr.decode(errors="replace"))
    return decode(result.stdout, options, result.stderr)


@contextmanager
def without_rg():
    # Keep target utilities but omit both rg and Python. An accidental Python
    # invocation fails loudly rather than silently using the test interpreter.
    with tempfile.TemporaryDirectory() as td:
        directory = Path(td)
        for name in TOOLS:
            path = shutil.which(name)
            if path:
                (directory / name).symlink_to(path)
        for name in ("python", "python3"):
            marker = directory / name
            marker.write_text("#!/bin/sh\necho 'UNEXPECTED TARGET PYTHON' >&2\nexit 99\n")
            marker.chmod(0o755)
        with patch.dict(os.environ, {"PATH": str(directory)}):
            yield directory
