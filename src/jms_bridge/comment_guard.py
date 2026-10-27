"""Narrow Python-comment analysis. This module contains no executor.

For an already-matched command, narrowly recognize one literal Python heredoc,
then ignore only proven COMMENT tokens. All strings and executable text retain
the original substring checks. Unsupported/malformed input retains the original
denial. The original script is never returned in transformed form or executed.
"""

import ast
import io
import re
import tokenize

HEADER = re.compile(
    r"[ \t]*python3[ \t]+-I[ \t]+-S[ \t]+-[ \t]+<<[ \t]*'"
    r"(?P<delimiter>[A-Za-z_][A-Za-z0-9_]{0,63})'[ \t]*"
)
ENCODING_COOKIE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")


def matches(text, patterns):
    flat = " ".join(text.split())
    return [pattern for pattern in patterns if pattern in flat]


def analyze(command, patterns):
    """Return an advisory decision; never supplies an executable replacement.

    `blocked=False` without original matches preserves the existing guard's
    behavior; it is not an independent safety/authorization determination.
    """
    original = matches(command, patterns)
    result = dict(
        blocked=bool(original),
        original_patterns=original,
        effective_patterns=original,
        ignored_comment_matches=[],
        analysis="no_original_match" if not original else "original_denial_retained",
    )
    if not original:
        return result
    if len(command) > 256 * 1024:
        result["reason"] = "analysis_size_limit"
        return result
    if any(
        (ord(char) < 32 and char not in "\t\n") or char in "\x85\u2028\u2029" for char in command
    ):
        result["reason"] = "unsupported_control_or_line_separator"
        return result
    lines = command.splitlines(keepends=True)
    first = next((i for i, line in enumerate(lines) if line.strip(" \t\n")), None)
    if first is None:
        return result
    header = HEADER.fullmatch(lines[first].rstrip("\n"))
    if header is None:
        result["reason"] = "unsupported_shell_form"
        return result
    delimiter = header["delimiter"]
    # Shell closes a heredoc at the FIRST exact delimiter line. Do not treat a
    # later matching line as the end or overlook an appended second command.
    close = next(
        (i for i in range(first + 1, len(lines)) if lines[i].rstrip("\n") == delimiter), None
    )
    if close is None or any(line.strip(" \t\n") for line in lines[close + 1 :]):
        result["reason"] = "missing_delimiter_or_trailing_shell_content"
        return result
    body_lines = lines[first + 1 : close]
    body = "".join(body_lines)
    # A coding cookie affects Python's byte decoding. This Unicode-only analyzer
    # must not exempt such a command based on a different decoded token stream.
    if any(ENCODING_COOKIE.match(line) for line in body_lines[:2]):
        result["reason"] = "encoding_cookie_not_supported"
        return result
    try:
        ast.parse(body, filename="<offline-heredoc>")
        tokens = list(tokenize.generate_tokens(io.StringIO(body).readline))
    except (SyntaxError, ValueError, tokenize.TokenError, RecursionError, UnicodeError):
        result["reason"] = "python_parse_failed"
        return result
    analyzed = list(body_lines)
    ignored = []
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        row, start = token.start
        end_row, end = token.end
        if row != end_row or not 1 <= row <= len(analyzed):
            result["reason"] = "unsupported_token_span"
            return result
        line = analyzed[row - 1]
        # Keep columns/newlines stable; never join code formerly separated by a
        # comment and never remove '#' inside STRING/f-string tokens.
        analyzed[row - 1] = line[:start] + " " * (end - start) + line[end:]
        hits = matches(token.string, patterns)
        if hits:
            ignored.append(dict(line=first + 1 + row, column=start, patterns=hits))
    remaining = matches("".join(lines[: first + 1] + analyzed + lines[close:]), patterns)
    result.update(
        blocked=bool(remaining),
        effective_patterns=remaining,
        ignored_comment_matches=ignored,
        analysis="executable_or_literal_match" if remaining else "proven_python_comment_only",
    )
    return result


def check(command, patterns):
    """Shared live predicate/diagnostic boundary; unexpected errors fail closed."""
    try:
        decision = analyze(command, patterns)
        if type(decision.get("blocked")) is not bool:
            raise ValueError("invalid guard decision")
        return {**decision, "execution_performed": False}
    except Exception as exc:
        return dict(
            blocked=True,
            analysis="analysis_failed_closed",
            error_type=type(exc).__name__,
            execution_performed=False,
        )
