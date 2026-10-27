from unittest.mock import patch

import pytest

from jms_bridge.comment_guard import check

PATTERNS = ["rm -rf", "kill -9"]


def heredoc(body):
    return "python3 -I -S - <<'PY'\n" + body + "\nPY\n"


@pytest.mark.parametrize(
    ("script", "blocked"),
    [
        ("printf hello", False),
        ("rm -rf /tmp/example", True),
        (heredoc("# rm -rf is documentation\nprint('ok')"), False),
        (heredoc("print('ok')  # kill -9 is documentation"), False),
        (heredoc("print('rm -rf')"), True),
        (heredoc('value = """# rm -rf"""'), True),
        (heredoc("# rm -rf\nprint('ok')") + "kill -9 123\n", True),
        ("python3 - <<'PY'\n# rm -rf\nPY\n", True),
        ("python3 -I -S - <<PY\n# rm -rf\nPY\n", True),
        (heredoc("# coding: utf-8\n# rm -rf\nprint('ok')"), True),
        (heredoc("# rm -rf\ninvalid syntax !"), True),
        (heredoc("# rm -rf\nprint('ok')").replace("\n", "\r\n"), True),
        (heredoc("# rm -rf\u2028\nprint('ok')"), True),
        (heredoc("# rm -rf\n")[:-3], True),
        (heredoc("# rm -rf\n" + "# x\n" * 70000), True),
    ],
)
def test_guard_only_ignores_proven_comments(script, blocked):
    result = check(script, PATTERNS)
    assert result["blocked"] is blocked
    assert result["execution_performed"] is False
    assert "script" not in result


def test_unexpected_analysis_failure_is_closed():
    with patch("jms_bridge.comment_guard.ast.parse", side_effect=RuntimeError("test")):
        result = check(heredoc("# rm -rf\nprint('ok')"), PATTERNS)
    assert result["blocked"] and result["analysis"] == "analysis_failed_closed"


def test_comment_string_removal_does_not_change_quoted_strings():
    result = check(heredoc("# kill -9\nprint('# rm -rf')"), PATTERNS)
    assert result["blocked"]
    assert result["effective_patterns"] == ["rm -rf"]
    assert result["ignored_comment_matches"][0]["patterns"] == ["kill -9"]
