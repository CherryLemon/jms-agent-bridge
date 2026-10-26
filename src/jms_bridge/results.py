"""Private, durable results with bounded previews and byte-offset reads."""

import json
import os
import re
import time
import uuid
from pathlib import Path

from .runtime_config import default_state_dir


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, ensure_ascii=False, separators=(",", ":"))
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class ResultStore:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else default_state_dir() / "results"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)

    def directory(self, result_id):
        if not re.fullmatch(r"[0-9a-f]{32}", result_id):
            raise ValueError("invalid result_id")
        return self.root / result_id

    def create(self, **metadata):
        result_id = uuid.uuid4().hex
        directory = self.directory(result_id)
        directory.mkdir(mode=0o700)
        metadata.update(result_id=result_id, created_at=time.time(), status="running")
        atomic_json(directory / "meta.json", metadata)
        return result_id

    def path(self, result_id, stream):
        if stream not in ("stdout", "stderr", "result"):
            raise ValueError("stream must be stdout, stderr, or result")
        return self.directory(result_id) / (stream + (".json" if stream == "result" else ".log"))

    def metadata(self, result_id):
        try:
            return json.loads((self.directory(result_id) / "meta.json").read_text())
        except FileNotFoundError:
            raise ValueError("unknown result_id") from None

    def finish(self, result_id, **values):
        meta = self.metadata(result_id)
        meta.update(values, status="complete", finished_at=time.time())
        meta["bytes"] = {
            s: self.path(result_id, s).stat().st_size
            for s in ("stdout", "stderr", "result")
            if self.path(result_id, s).exists()
        }
        atomic_json(self.directory(result_id) / "meta.json", meta)
        return meta

    def save_json(self, value, **meta):
        result_id = self.create(**meta)
        atomic_json(self.path(result_id, "result"), value)
        self.finish(result_id, ok=not (isinstance(value, dict) and value.get("ok") is False))
        return result_id

    def preview(self, result_id, budget=4096):
        meta = self.metadata(result_id)
        sizes = meta.get("bytes", {})
        # stderr gets a reserved share, so noisy stdout cannot hide the error.
        err_budget = min(sizes.get("stderr", 0), max(0, budget // 2))
        out_budget = min(sizes.get("stdout", 0), max(0, budget - err_budget))
        err_budget = min(sizes.get("stderr", 0), max(0, budget - out_budget))
        result = {}
        for stream, limit in [("stdout", out_budget), ("stderr", err_budget)]:
            size = sizes.get(stream, 0)
            path = self.path(result_id, stream)
            if size and limit:
                with path.open("rb") as fh:
                    # Failure diagnostics usually live at the end of stderr.
                    if stream == "stderr":
                        fh.seek(max(0, size - limit))
                    result[stream] = fh.read(limit).decode("utf-8", errors="replace")
            if size > limit:
                result[stream + "_truncated"] = True
        return result

    def read(self, result_id, stream="stdout", offset=0, limit=4096, tail=False, pattern=None):
        meta = self.metadata(result_id)
        if not 1 <= limit <= 16384 or offset < 0:
            raise ValueError("limit must be 1..16384 bytes; offset must be nonnegative")
        path = self.path(result_id, stream)
        if not path.exists():
            raise ValueError(f"no {stream} stream; available: {list(meta.get('bytes', {}))}")
        size = path.stat().st_size
        if pattern is not None:
            if tail or not pattern or len(pattern.encode()) > 1024:
                raise ValueError(
                    "search needs a nonempty pattern of at most 1024 bytes and tail=false"
                )
            return self._search(path, result_id, stream, pattern, offset, limit, size)
        start = max(0, size - limit) if tail else min(offset, size)
        with path.open("rb") as fh:
            fh.seek(start)
            data = fh.read(limit + 3)
        if tail:
            while data and 0x80 <= data[0] < 0xC0:
                data = data[1:]
                start += 1
        count = min(limit, len(data))
        # Avoid splitting valid UTF-8 between pages. A tiny limit can grow by <=3 bytes.
        while count < len(data) and 0x80 <= data[count] < 0xC0:
            count += 1
        data = data[:count]
        end = start + len(data)
        return {
            "result_id": result_id,
            "stream": stream,
            "offset": start,
            "next_offset": end,
            "total_bytes": size,
            "eof": end >= size,
            "text": data.decode("utf-8", errors="replace"),
        }

    def _search(self, path, result_id, stream, pattern, offset, limit, size):
        needle = pattern.encode()
        start = min(offset, size)
        # Bounded scan and bounded result. Cursor can continue even with zero matches.
        end = min(size, start + 8 * 1024 * 1024)
        matches, spent, cursor = [], 0, start
        with path.open("rb") as fh:
            while cursor < end and spent < limit and len(matches) < 20:
                fh.seek(cursor)
                advance = min(65536, end - cursor)
                data = fh.read(advance + len(needle) - 1)
                found = data.find(needle)
                if found < 0:
                    cursor += advance
                    continue
                position = cursor + found
                lo = max(0, position - 120)
                fh.seek(lo)
                snippet = fh.read(min(360, limit - spent))
                matches.append(
                    {"offset": position, "text": snippet.decode("utf-8", errors="replace")}
                )
                spent += len(snippet)
                cursor = position + len(needle)
        return {
            "result_id": result_id,
            "stream": stream,
            "matches": matches,
            "next_offset": min(cursor, size),
            "total_bytes": size,
            "eof": cursor >= size,
        }
