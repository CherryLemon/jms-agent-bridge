"""Remote stdlib-only filesystem reader. All request values are data, never code."""

import base64
import collections
import fnmatch
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time

SCAN_LIMIT = 100000
SCAN_BYTES = 64 * 1024 * 1024
DEFAULT_BYTES = 4096


def error(exc, path=None):
    return dict(
        ok=False, error=type(exc).__name__, message=str(exc), **({"path": path} if path else {})
    )


def info(path):
    s = os.lstat(path)
    kind = (
        "file"
        if stat.S_ISREG(s.st_mode)
        else "directory"
        if stat.S_ISDIR(s.st_mode)
        else "symlink"
        if stat.S_ISLNK(s.st_mode)
        else "special"
    )
    row = dict(
        path=path,
        name=os.path.basename(path.rstrip("/")),
        type=kind,
        size=s.st_size,
        mtime_ns=s.st_mtime_ns,
        mode=stat.filemode(s.st_mode),
        uid=s.st_uid,
        gid=s.st_gid,
    )
    if kind == "symlink":
        row.update(link_target=os.readlink(path), resolved_path=os.path.realpath(path))
    return row


def regular_file(path):
    # O_NONBLOCK avoids hanging on a FIFO even if the path changes after stat.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("only regular files can be read/checksummed")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def read_file(path, options, budget):
    with regular_file(path) as f:
        before = os.fstat(f.fileno())
        size = before.st_size
        if b"\x00" in f.read(8192):
            raise ValueError("binary file; use sync_pull to download it")
        count = options.get("line_count", 200)
        tail = options.get("tail", False)
        line = options.get("start_line", 1)
        partial_first = False
        if tail:
            start = max(0, size - budget)
            boundary = True
            if start:
                f.seek(start - 1)
                boundary = f.read(1) == b"\n"
            f.seek(start)
            data = f.read(budget)
            if start and not boundary:
                # Drop the first incomplete line when possible.
                pos = data.find(b"\n")
                if pos >= 0:
                    start += pos + 1
                    data = data[pos + 1 :]
                else:
                    partial_first = True
            pieces = data.splitlines(keepends=True)[-count:]
            selected = b"".join(pieces)
            start += len(data) - len(selected)
            data = selected
            line = None
        else:
            f.seek(min(options.get("byte_offset", 0), size))
            if "byte_offset" in options:
                line = None  # Exact global line number is unknown without rescanning.
                start = f.tell()
                if start:
                    f.seek(start - 1)
                    partial_first = f.read(1) != b"\n"
            else:
                for _ in range(line - 1):
                    if f.tell() >= SCAN_BYTES:
                        raise ValueError("start_line scan exceeds 64 MiB; use byte_offset")
                    part = f.readline(SCAN_BYTES - f.tell() + 1)
                    if f.tell() > SCAN_BYTES:
                        raise ValueError("start_line scan exceeds 64 MiB; use byte_offset")
                    if not part:
                        break
            start = f.tell()
            data = f.read(budget + 3)
            n = min(budget, len(data))
            while n < len(data) and 0x80 <= data[n] < 0xC0:
                n += 1
            data = b"".join(data[:n].splitlines(keepends=True)[:count])
        if tail or partial_first:
            while data and 0x80 <= data[0] < 0xC0:
                start += 1
                data = data[1:]
        end = start + len(data)
        after = os.fstat(f.fileno())
        eof = end >= size
        partial_last = bool(data and not data.endswith(b"\n") and not eof)
        return dict(
            ok=True,
            path=path,
            text=data.decode("utf-8", errors="replace"),
            encoding="utf-8",
            decoding_errors="replace",
            start_line=line,
            next_line=(line + data.count(b"\n")) if line is not None else None,
            byte_offset=start,
            next_byte_offset=end,
            total_bytes=size,
            eof=eof,
            truncated=not eof or (tail and start > 0),
            partial_first_line=partial_first,
            partial_last_line=partial_last,
            changed_during_read=(before.st_size, before.st_mtime_ns)
            != (after.st_size, after.st_mtime_ns),
            fingerprint=dict(size=after.st_size, mtime_ns=after.st_mtime_ns),
        )


def matches(path, patterns):
    return any(
        fnmatch.fnmatchcase(path, p)
        or fnmatch.fnmatchcase(os.path.basename(path), p)
        or (p.startswith("**/") and fnmatch.fnmatchcase(path, p[3:]))
        for p in patterns
    )


def walk(path, options, state, depth=128):
    """Deterministic, bounded traversal; never descends into symlink directories."""
    seen = 0

    def descend(root, level):
        nonlocal seen
        try:
            with os.scandir(root) as it:
                entries = []
                for entry in it:
                    seen += 1
                    if seen > SCAN_LIMIT:
                        state.update(scan_limited=True)
                        return
                    entries.append(entry)
            entries.sort(key=lambda e: e.name)
        except TimeoutError:
            raise
        except OSError as exc:
            if len(state["errors"]) < 20:
                state["errors"].append(error(exc, root))
            return
        for e in entries:
            rel = os.path.relpath(e.path, path)
            if (not options.get("hidden") and e.name.startswith(".")) or matches(
                rel, options.get("exclude", [])
            ):
                continue
            try:
                row = info(e.path)
                row.update(relative_path=rel, depth=level)
                yield row
                if row["type"] == "directory" and level < depth:
                    yield from descend(e.path, level + 1)
                elif (
                    row["type"] == "directory"
                    and level == depth
                    and options["action"] in ("glob", "search")
                ):
                    state.update(scan_limited=True, depth_limited=True)
            except OSError as exc:
                if len(state["errors"]) < 20:
                    state["errors"].append(error(exc, e.path))

    yield from descend(path, 1)


def rg_rows(path, options, state, glob=False):
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    operand = "." if os.path.isdir(path) else os.path.basename(path)
    args = ["rg", "--no-config", "--sort", "path", "--threads", "1"]
    args += (
        ["--files", "--null"]
        if glob
        else [
            "--json",
            "--color",
            "never",
            "--max-filesize",
            str(options.get("max_file_bytes", 8388608)),
        ]
    )
    if options.get("hidden"):
        args.append("--hidden")
    if not options.get("respect_ignore", True):
        args.append("--no-ignore")
    # Positive --glob rules override .gitignore in rg. Filter positive patterns
    # after traversal instead, preserving ignore semantics on both operations.
    for p in options.get("exclude", []):
        args += ["--glob", "!" + p, "--glob", "!" + p.rstrip("/") + "/**"]
    if glob:
        args += ["--", operand]
    else:
        if options.get("literal", True):
            args.append("--fixed-strings")
        if not options.get("case_sensitive", True):
            args.append("--ignore-case")
        args += [
            "--context",
            str(options.get("context_lines", 0)),
            "--",
            options["pattern"],
            operand,
        ]
    # Never use shell=True. A private temporary stderr file avoids pipe deadlocks;
    # it is unlinked on exit, and not a persistent file on the target.
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(args, cwd=directory, stdout=subprocess.PIPE, stderr=err)
        try:
            if glob:
                pending = b""
                while True:
                    chunk = proc.stdout.read1(8192)
                    if not chunk:
                        break
                    parts = (pending + chunk).split(b"\0")
                    pending = parts.pop()
                    for part in parts:
                        rel = os.fsdecode(part).removeprefix("./")
                        if not matches(rel, [options["pattern"]]) or (
                            options.get("include") and not matches(rel, options["include"])
                        ):
                            continue
                        yield dict(path=os.path.join(directory, rel), relative_path=rel)
            else:
                while True:
                    raw = proc.stdout.readline(2 * 1024 * 1024)
                    if not raw:
                        break
                    if not raw.endswith(b"\n"):
                        state["errors"].append(
                            dict(
                                error="line_too_large",
                                message="rg JSON event exceeds 2 MiB; narrow the search",
                            )
                        )
                        state["scan_limited"] = True
                        break
                    event = json.loads(raw)
                    if event["type"] not in ("match", "context"):
                        continue
                    data = event["data"]

                    def decode(value):
                        return (
                            value.get("text")
                            if "text" in value
                            else base64.b64decode(value["bytes"]).decode("utf-8", "replace")
                        )

                    rel = decode(data["path"]).removeprefix("./")
                    if options.get("include") and not matches(rel, options["include"]):
                        continue
                    text = decode(data["lines"]).rstrip("\r\n")
                    yield dict(
                        path=os.path.join(directory, rel),
                        relative_path=rel,
                        line=data["line_number"],
                        kind=event["type"],
                        text=text[:2000],
                        line_truncated=len(text) > 2000,
                    )
            if state["scan_limited"] and proc.poll() is None:
                proc.kill()
            rc = proc.wait()
            if rc not in (0, 1):
                err.seek(0)
                state["errors"].append(
                    dict(
                        error="ripgrep_failed",
                        message=err.read(1024).decode("utf-8", "replace"),
                        exit_code=rc,
                    )
                )
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            proc.stdout.close()


def python_search(path, options, state):
    pattern = options["pattern"]
    flags = 0 if options.get("case_sensitive", True) else re.IGNORECASE
    matcher = re.compile(re.escape(pattern) if options.get("literal", True) else pattern, flags)
    scanned = 0
    rows = (
        walk(path, options, state)
        if os.path.isdir(path)
        else iter([info(path) | {"relative_path": os.path.basename(path), "type": "file"}])
    )
    for row in rows:
        if row["type"] != "file" or (
            options.get("include") and not matches(row["relative_path"], options["include"])
        ):
            continue
        if row["size"] > options.get("max_file_bytes", 8388608):
            state["skipped_large_files"] += 1
            continue
        scanned += row["size"]
        if scanned > SCAN_BYTES:
            state["scan_limited"] = True
            break
        try:
            with regular_file(row["path"]) as f:
                if b"\x00" in f.read(8192):
                    state["skipped_binary_files"] += 1
                    continue
                f.seek(0)
                before = collections.deque(maxlen=options.get("context_lines", 0))
                after_until, last_emitted = 0, 0
                for number, raw in enumerate(f, 1):
                    text = raw.decode("utf-8", "replace").rstrip("\r\n")
                    hit = matcher.search(text) is not None
                    current = dict(
                        path=row["path"],
                        relative_path=row["relative_path"],
                        line=number,
                        text=text[:2000],
                        line_truncated=len(text) > 2000,
                    )
                    if hit:
                        for previous in before:
                            if previous["line"] > last_emitted:
                                yield previous | {"kind": "context"}
                        yield current | {"kind": "match"}
                        last_emitted = number
                        after_until = number + options.get("context_lines", 0)
                    elif number <= after_until:
                        yield current | {"kind": "context"}
                        last_emitted = number
                    before.append(current)
        except TimeoutError:
            raise
        except OSError as exc:
            if len(state["errors"]) < 20:
                state["errors"].append(error(exc, row["path"]))


def page(rows, options, state):
    offset, limit = options.get("offset", 0), options.get("limit", 100)
    budget = options.get("max_bytes", DEFAULT_BYTES)
    out, spent, more, examined = [], 0, False, 0
    try:
        for i, row in enumerate(rows):
            examined += 1
            if examined > SCAN_LIMIT:
                state["scan_limited"] = True
                break
            if i < offset:
                continue
            cost = len(json.dumps(row, ensure_ascii=False).encode())
            if len(out) >= limit or (out and spent + cost > budget):
                more = True
                break
            out.append(row)
            spent += cost
    except TimeoutError:
        state.update(scan_limited=True, timed_out=True)
    finally:
        rows.close()
    incomplete = more or state["scan_limited"]
    return dict(
        ok=not state["errors"],
        entries=out,
        returned=len(out),
        offset=offset,
        next_offset=offset + len(out) if more else None,
        truncated=incomplete,
        complete=not incomplete and not state["errors"],
        **state,
    )


def execute(options):
    action, path = options["action"], options.get("path")
    budget = options.get("max_bytes", DEFAULT_BYTES)
    if action == "stat":
        return dict(ok=True, **info(path))
    if action == "read":
        return read_file(path, options, budget)
    if action == "read_many":
        results = []
        for p in options["paths"]:
            try:
                results.append(read_file(p, options, budget // len(options["paths"])))
            except (OSError, ValueError) as exc:
                results.append(error(exc, p))
        return dict(ok=all(r["ok"] for r in results), files=results)
    if action == "checksum":
        with regular_file(path) as f:
            before = os.fstat(f.fileno())
            cap = options.get("max_file_bytes", 8388608)
            if before.st_size > cap:
                raise ValueError("file exceeds max_file_bytes checksum limit")
            h, total = hashlib.sha256(), 0
            while True:
                data = f.read(min(65536, cap - total + 1))
                if not data:
                    break
                total += len(data)
                if total > cap:
                    raise ValueError("file grew beyond checksum limit")
                h.update(data)
            after = os.fstat(f.fileno())
            changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            return dict(
                ok=not changed,
                path=path,
                algorithm="sha256",
                digest=h.hexdigest(),
                bytes=total,
                changed_during_read=changed,
            )
    if not os.path.isdir(path) and not (action == "search" and os.path.isfile(path)):
        raise ValueError(
            "list/tree/glob require a directory; search requires an existing file or directory"
        )
    state = dict(
        errors=[],
        scan_limited=False,
        engine="python",
        ignore_rules_applied=False,
        skipped_large_files=0,
        skipped_binary_files=0,
    )
    if action in ("list", "tree"):
        rows = walk(path, options, state, depth=1 if action == "list" else options.get("depth", 3))
    elif shutil.which("rg"):
        state.update(
            engine="ripgrep",
            ignore_rules_applied=options.get("respect_ignore", True),
            skipped_binary_files=None,
            skipped_large_files=None,
        )
        rows = rg_rows(path, options, state, glob=action == "glob")
    else:
        if options.get("respect_ignore", True):
            state["warnings"] = [
                (
                    "rg unavailable: .gitignore/.ignore rules are not applied; "
                    "use exclude or install rg."
                )
            ]
        if action == "glob":
            rows = (
                r
                for r in walk(path, options, state)
                if r["type"] == "file"
                and matches(r["relative_path"], [options["pattern"]])
                and (not options.get("include") or matches(r["relative_path"], options["include"]))
            )
        else:
            rows = python_search(path, options, state)
    return page(rows, options, state) | {"path": path, "action": action}


def main(options):
    def timeout(signum, frame):
        raise TimeoutError("filesystem operation exceeded time budget")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(options.get("timeout", 30))
    start = time.monotonic()
    try:
        result = execute(options)
    except (OSError, ValueError, TimeoutError, re.error) as exc:
        result = error(exc, options.get("path"))
    finally:
        signal.alarm(0)
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
