"""Generate literal Bash requests and decode bounded records on the MCP host."""

import base64
import json
import os
import shlex
import stat
from pathlib import Path

SCAN_LIMIT = 100000
SCAN_BYTES = 64 * 1024 * 1024
WIRE_LIMIT = 1024 * 1024
DEFAULT_BYTES = 4096
ARITY = dict(H=3, E=3, I=10, G=2, R=8, C=7, L=6, J=1, P=7)


def bash_glob(pattern):
    """Keep fnmatch's literal backslashes and parentheses out of Bash extglobs."""
    result, inside = [], False
    for i, char in enumerate(pattern):
        if char == "[":
            inside = True
        elif char == "]":
            inside = False
        if char == "\\" or (not inside and char in "()|+@!"):
            result.append("\\")
        if inside and char == "^" and i and pattern[i - 1] == "[":
            result.append("\\")
        result.append(char)
    return "".join(result)


def remote_script(options, *, rg_command="rg"):
    for value in [
        options.get("path", ""),
        options.get("pattern", ""),
        rg_command,
        *options.get("paths", []),
        *options.get("include", []),
        *options.get("exclude", []),
    ]:
        if "\0" in value:
            raise ValueError("NUL is not a valid shell argument")
        value.encode("utf-8")
    action = options["action"]
    if action == "search" and "\n" in options.get("pattern", ""):
        raise ValueError("line-based search patterns must be single-line strings")
    depth = 1 if action == "list" else options.get("depth", 3) if action == "tree" else 128
    numbers = [
        options.get("max_bytes", DEFAULT_BYTES),
        options.get("line_count", 200),
        options.get("start_line", 1),
        options.get("byte_offset", -1),
        int(options.get("tail", False)),
        depth,
        options.get("limit", 100),
        options.get("offset", 0),
        options.get("max_file_bytes", 8388608),
        int(options.get("literal", True)),
        int(options.get("case_sensitive", True)),
        int(options.get("hidden", False)),
        int(options.get("respect_ignore", True)),
        SCAN_LIMIT,
        SCAN_BYTES,
        WIRE_LIMIT,
    ]
    if any(type(n) is not int for n in numbers):
        raise ValueError("filesystem numeric options must be integers")
    args = [
        action,
        options.get("path", ""),
        options.get("pattern", ""),
        *map(str, numbers),
        bash_glob(options.get("pattern", "")),
        str(options.get("context_lines", 0)),
        rg_command,
    ]
    for values in (
        options.get("paths", []),
        [bash_glob(p) for p in options.get("include", [])],
        [bash_glob(p) for p in options.get("exclude", [])],
    ):
        args += [str(len(values)), *values]
    timeout = int(options.get("timeout", 30))
    source = Path(__file__).with_name("fs_probe.sh").read_text()
    # User values are argv, outside the fixed here-document; quotes/newlines
    # cannot terminate its delimiter or become executable shell syntax.
    command = shlex.join(
        [
            "timeout",
            "--signal=TERM",
            "--kill-after=1",
            str(timeout),
            "bash",
            "--noprofile",
            "--norc",
            "-s",
            "--",
            *args,
        ]
    )
    return (
        "(\nset +e\nif ! command -v timeout >/dev/null 2>&1; then "
        "printf 'JMSFS-MISSING timeout\\n';\n"
        "elif ! command -v base64 >/dev/null 2>&1; then printf 'JMSFS-MISSING base64\\n';\n"
        "else\nprintf 'JMSFS-BASH1\\n'\n"
        + command
        + " <<'__JMS_FS_BASH__' | base64 -w0\n"
        + source
        + '\n__JMS_FS_BASH__\n__jms_fs_codes=("${PIPESTATUS[@]}")\n'
        "__jms_fs_rc=${__jms_fs_codes[0]}\n"
        "if (( __jms_fs_codes[1] != 0 )); then __jms_fs_rc=${__jms_fs_codes[1]}; fi\n"
        "printf '\\nJMSFS-END %s\\n' \"$__jms_fs_rc\"\nfi\n)\n"
    )


def text(value):
    return value.decode("utf-8", "replace")


def error(kind, message, path=None):
    return dict(ok=False, error=kind, message=message, **({"path": path} if path else {}))


def read_page(fields, options):
    path, size, stamp, start, previous, payload, after_size, after_stamp = fields
    size, stamp, start = int(size), int(stamp), int(start)
    after_size, after_stamp = int(after_size), int(after_stamp)
    data = base64.b64decode(payload, validate=True)
    tail = options.get("tail", False)
    count = options.get("line_count", 200)
    budget = options.get("max_bytes", DEFAULT_BYTES)
    if options["action"] == "read_many":
        budget //= len(options["paths"])
    partial_first = start > 0 and base64.b64decode(previous, validate=True) != b"\n"
    line = None if tail or "byte_offset" in options else options.get("start_line", 1)
    if tail:
        if partial_first:
            position = data.find(b"\n")
            if position >= 0:
                start += position + 1
                data = data[position + 1 :]
                partial_first = False
        selected = b"".join(data.splitlines(keepends=True)[-count:])
        start += len(data) - len(selected)
        data = selected
    else:
        n = min(budget, len(data))
        while n < len(data) and 0x80 <= data[n] < 0xC0:
            n += 1
        data = b"".join(data[:n].splitlines(keepends=True)[:count])
    if tail or partial_first:
        while data and 0x80 <= data[0] < 0xC0:
            start += 1
            data = data[1:]
    end = start + len(data)
    eof = end >= size
    return dict(
        ok=True,
        path=text(path),
        text=text(data),
        encoding="utf-8",
        decoding_errors="replace",
        start_line=line,
        next_line=line + data.count(b"\n") if line is not None else None,
        byte_offset=start,
        next_byte_offset=end,
        total_bytes=size,
        eof=eof,
        truncated=not eof or (tail and start > 0),
        partial_first_line=partial_first,
        partial_last_line=bool(data and not data.endswith(b"\n") and not eof),
        changed_during_read=(size, stamp) != (after_size, after_stamp),
        fingerprint=dict(size=after_size, mtime_ns=after_stamp),
    )


def info(fields, action):
    path, relative, depth, mode, size, stamp, uid, gid, link, resolved = fields
    mode = int(mode, 16)
    kind = (
        "file"
        if stat.S_ISREG(mode)
        else "directory"
        if stat.S_ISDIR(mode)
        else "symlink"
        if stat.S_ISLNK(mode)
        else "special"
    )
    result = dict(
        path=text(path),
        name=text(os.path.basename(path.rstrip(b"/"))),
        type=kind,
        size=int(size),
        mtime_ns=int(stamp),
        mode=stat.filemode(mode),
        uid=int(uid),
        gid=int(gid),
    )
    if kind == "symlink":
        result.update(link_target=text(link), resolved_path=text(resolved))
    if action != "stat":
        result.update(relative_path=text(relative), depth=int(depth))
    if result["path"].encode() != path:
        result.update(path_bytes_b64=base64.b64encode(path).decode(), path_encoding="utf-8-replace")
    return result


def rg_row(raw, root):
    event = json.loads(raw)
    data = event["data"]

    def decode_field(value):
        return value["text"] if "text" in value else text(base64.b64decode(value["bytes"]))

    path = decode_field(data["path"])
    if not os.path.isabs(path):
        path = os.path.join(root, path[2:] if path.startswith("./") else path)
    # Never inspect the remote path on the local server to determine its type.
    relative = (
        os.path.relpath(path, root)
        if path.startswith(root.rstrip("/") + "/")
        else os.path.basename(path)
    )
    content = decode_field(data["lines"]).rstrip("\r\n")
    return dict(
        path=path,
        relative_path=relative,
        line=data["line_number"],
        kind=event["type"],
        text=content[:2000],
        line_truncated=len(content) > 2000,
    )


def decode(raw, options, stderr=b""):
    if raw.startswith(b"JMSFS-MISSING "):
        return error(
            "UnsupportedOperation", "missing target command: " + text(raw.split(None, 1)[1]).strip()
        )
    if len(raw) > WIRE_LIMIT * 2 + 65536:
        return error("InvalidProbeResponse", "filesystem probe exceeded wire limit")
    header, _, encoded = raw.partition(b"\n")
    body, sep, status = encoded.rpartition(b"\nJMSFS-END ")
    if header != b"JMSFS-BASH1" or not sep:
        return error("InvalidProbeResponse", "missing Bash probe completion marker")
    rc = int(status.strip())
    payload = base64.b64decode(body, validate=True)
    tokens = payload.split(b"\0")
    records, index, partial = [], 0, False
    while index < len(tokens) - 1:
        tag = text(tokens[index])
        count = ARITY.get(tag)
        if count is None or index + count >= len(tokens):
            partial = True
            break
        records.append((tag, tokens[index + 1 : index + 1 + count]))
        index += count + 1
    if tokens[-1] or not any(tag == "P" for tag, _ in records):
        partial = True
    state = dict(
        errors=[],
        scan_limited=partial or rc != 0,
        engine="bash",
        ignore_rules_applied=False,
        skipped_large_files=0,
        skipped_binary_files=0,
    )
    rows, files, single, elapsed = [], [], None, 0.0
    for tag, fields in records:
        if tag == "H":
            engine, ignored, warning = map(text, fields)
            state.update(engine=engine, ignore_rules_applied=ignored == "1")
            if warning:
                state["warnings"] = [warning]
        elif tag == "E":
            path, kind, message = map(text, fields)
            item = error(kind, message, path)
            if options["action"] == "read_many":
                files.append(item)
            else:
                single = item
            if len(state["errors"]) < 20:
                state["errors"].append(item)
        elif tag == "I":
            row = info(fields, options["action"])
            if options["action"] == "stat":
                single = dict(ok=True, **row)
            else:
                rows.append(row)
        elif tag == "G":
            rows.append(dict(path=text(fields[0]), relative_path=text(fields[1])))
        elif tag == "R":
            row = read_page(fields, options)
            if options["action"] == "read_many":
                files.append(row)
            else:
                single = row
        elif tag == "C":
            path, digest, count, size, stamp, after_size, after_stamp = fields
            changed = (size, stamp) != (after_size, after_stamp)
            single = dict(
                ok=not changed,
                path=text(path),
                algorithm="sha256",
                digest=text(digest),
                bytes=int(count),
                changed_during_read=changed,
            )
        elif tag == "L":
            path, relative, number, kind, content, clipped = fields
            content = text(content).rstrip("\r\n")
            rows.append(
                dict(
                    path=text(path),
                    relative_path=text(relative),
                    line=int(number),
                    kind=text(kind),
                    text=content[:2000],
                    line_truncated=clipped == b"1" or len(content) > 2000,
                )
            )
        elif tag == "J":
            rows.append(rg_row(fields[0], options["path"]))
        elif tag == "P":
            _, large, binary, limited, depth = map(int, fields[:5])
            elapsed = max(0, round(float(fields[6]) - float(fields[5]), 3))
            state.update(
                skipped_large_files=None if large < 0 else large,
                skipped_binary_files=None if binary < 0 else binary,
            )
            state["scan_limited"] |= bool(limited)
            if depth:
                state["depth_limited"] = True
    if rc in (124, 137):
        state.update(timed_out=True, scan_limited=True)
    elif rc:
        state["errors"].append(
            error("CollectorFailure", "Bash collector failed: " + text(stderr[:1024]))
        )
    if options["action"] == "read_many":
        if rc or partial:
            return error(
                "TimeoutError" if rc in (124, 137) else "InvalidProbeResponse",
                "incomplete read_many response",
            )
        return dict(
            ok=all(row["ok"] for row in files), files=files, backend="bash", elapsed_seconds=elapsed
        )
    if options["action"] in ("read", "stat", "checksum"):
        if rc or partial or single is None:
            return error(
                "TimeoutError" if rc in (124, 137) else "InvalidProbeResponse",
                "incomplete filesystem response",
            )
        return dict(single, backend="bash", elapsed_seconds=elapsed)
    output, spent = [], 0
    budget, limit = options.get("max_bytes", DEFAULT_BYTES), options.get("limit", 100)
    for row in rows:
        size = len(json.dumps(row, ensure_ascii=False).encode())
        if len(output) >= limit or (output and spent + size > budget):
            break
        output.append(row)
        spent += size
    more = len(rows) > len(output)
    offset = options.get("offset", 0)
    incomplete = more or state["scan_limited"]
    return dict(
        ok=not state["errors"],
        entries=output,
        returned=len(output),
        offset=offset,
        next_offset=offset + len(output) if more else None,
        truncated=incomplete,
        complete=not incomplete and not state["errors"],
        path=options["path"],
        action=options["action"],
        backend="bash",
        elapsed_seconds=elapsed,
        **state,
    )
