"""The first-run path: what `jms-bridge init` writes and whether it is usable.

The rest of the CLI is covered only against a real bastion (see CONTRIBUTING), but the
templates are pure data and they are what a freshly-cloned checkout hands a new user.
The regression this guards: `init` used to write an inventory with no `workspaces:`
block, so the documented onboarding sequence produced a file on which every
`jms_dev_*` tool fails with "matched no workspace" — the dev loop, which is the point
of the tool, was unreachable from the documented path.
"""

import asyncio
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from jms_bridge.cli import _CONFIG_TEMPLATE, _INVENTORY_TEMPLATE, cmd_init, cmd_ssh_key
from jms_bridge.config import read_toml
from jms_bridge.inventory import Inventory


@pytest.fixture(autouse=True)
def _no_ambient_bridge_env(monkeypatch):
    # `Inventory.load(path)` resolves the path through `config.inventory_path()`, where the
    # env var wins over the explicit argument — so a developer (or CI job) with
    # JMS_BRIDGE_INVENTORY exported, which the README documents as a first-class way to
    # relocate the inventory, would make the template tests below parse *their* file. The
    # regression they pin would then go unguarded. tests/test_config.py and
    # tests/test_secrets.py delenv the same vars for the same reason.
    for name in ("JMS_BRIDGE_INVENTORY", "JMS_BRIDGE_CONFIG", "JMS_BRIDGE_CREDENTIALS"):
        monkeypatch.delenv(name, raising=False)


def _load_template(tmp_path, text):
    p = tmp_path / "hosts.yaml"
    p.write_text(text)
    return Inventory.load(p)


def test_init_inventory_template_loads(tmp_path):
    inv = _load_template(tmp_path, _INVENTORY_TEMPLATE)
    assert set(inv.hosts) == {"ascend-910c", "node-4090-1"}
    assert inv.groups["cluster"] == ["ascend-910c", "node-4090-1"]


def test_init_inventory_template_has_a_workspace(tmp_path):
    # Without this, `jms_dev_run(["all"], ...)` resolves to no workspaces and the
    # documented quickstart cannot deliver code to any chip.
    inv = _load_template(tmp_path, _INVENTORY_TEMPLATE)
    assert inv.workspaces, "init template must ship at least one workspace"
    ws = next(iter(inv.workspaces.values()))
    assert ws.host in inv.hosts
    assert ws.repo and ws.container


def test_init_inventory_template_only_uses_git_source(tmp_path):
    # `git` is the only source that needs no extra field, so it is the only one safe to
    # put in a template a user might run before reading the comments. proxy-git and push
    # would need `proxy:` / `mount:` + `local:` to load at all.
    inv = _load_template(tmp_path, _INVENTORY_TEMPLATE)
    assert {ws.source for ws in inv.workspaces.values()} == {"git"}


def test_init_config_template_has_a_bastion_url(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(_CONFIG_TEMPLATE)
    data = read_toml(p)
    assert data["bastion"]["url"]
    # The terse `NestedKeyError`-style failure a missing url would cause is exactly what
    # this template exists to prevent.
    assert data["bastion"]["login_user"] == "your-login"


def test_init_writes_both_files_and_tightens_the_config(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    inv = tmp_path / "hosts.yaml"
    monkeypatch.setenv("JMS_BRIDGE_CONFIG", str(cfg))
    monkeypatch.setenv("JMS_BRIDGE_INVENTORY", str(inv))

    class Args:
        force = False

    assert cmd_init(Args()) == 0
    assert cfg.exists() and inv.exists()
    # config.toml will hold a session token after `login`, and hosts.yaml holds asset
    # ids; both should be owner-only from the moment init writes them.
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o600


def test_init_does_not_clobber_without_force(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "config.toml"
    monkeypatch.setenv("JMS_BRIDGE_CONFIG", str(cfg))
    monkeypatch.setenv("JMS_BRIDGE_INVENTORY", str(tmp_path / "hosts.yaml"))
    cfg.write_text(textwrap.dedent("""\
        [bastion]
        url = "https://mine.example.com"
    """))

    class Args:
        force = False

    assert cmd_init(Args()) == 0
    assert "skipping" in capsys.readouterr().out
    assert "mine.example.com" in cfg.read_text()


def test_init_chmods_hosts_yaml_too(tmp_path, monkeypatch):
    # It names every host and its asset id, so it gets the same treatment as the config.
    cfg = tmp_path / "config.toml"
    inv = tmp_path / "hosts.yaml"
    monkeypatch.setenv("JMS_BRIDGE_CONFIG", str(cfg))
    monkeypatch.setenv("JMS_BRIDGE_INVENTORY", str(inv))

    class Args:
        force = False

    cmd_init(Args())
    assert stat.S_IMODE(inv.stat().st_mode) == 0o600
    # ...and the template comments survive the round trip (they are the documentation).
    assert "workspaces:" in inv.read_text()


# --- `python -m jms_bridge`: the CLI without an install -------------------------------


def test_module_entry_point_delegates_to_the_cli():
    # `[project.scripts]` puts `jms-bridge` on PATH only after an install, so a freshly
    # cloned checkout needed `python -c 'from jms_bridge.cli import main; ...'` to reach
    # `init` / `login` / `doctor`. `__main__.py` is the same `main`, reached the ordinary way.
    from jms_bridge import __main__ as entry
    from jms_bridge.cli import main

    assert entry.main is main


def test_python_m_jms_bridge_help_names_the_commands():
    repo = Path(__file__).resolve().parent.parent
    p = subprocess.run(
        [sys.executable, "-m", "jms_bridge", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
        timeout=60,
    )
    assert p.returncode == 0, p.stderr
    for cmd in ("init", "login", "doctor", "serve"):
        assert cmd in p.stdout


# --- ssh-key: the hardening hand-off ------------------------------------------------
# `ssh-key` exists so the data plane can stop carrying the account password. What the
# tests pin is the part a mistake would silently break: printing the *public* half (never
# the private one), not overwriting or regenerating a key on a second run, and refusing
# to "verify" a key that was never generated.


class _KeyArgs:
    def __init__(self, key, verify=False, targets=None):
        self.key = key
        self.verify = verify
        self.targets = targets


def test_ssh_key_generates_prints_only_the_public_half(tmp_path, capsys):
    if shutil.which("ssh-keygen") is None:
        pytest.skip("ssh-keygen not installed")
    key = tmp_path / "bastion_key"
    assert cmd_ssh_key(_KeyArgs(str(key))) == 0
    out = capsys.readouterr().out

    assert key.exists()
    pub = Path(str(key) + ".pub")
    assert pub.read_text().strip() in out
    # The private half must never be echoed — this command's whole output is meant to be
    # pasted into a web form and into a chat, so a leaked private key would be a real one.
    assert key.read_text().strip() not in out
    assert "SSH 公钥" in out and "--verify" in out


def test_ssh_key_is_owner_only(tmp_path):
    if shutil.which("ssh-keygen") is None:
        pytest.skip("ssh-keygen not installed")
    key = tmp_path / "bastion_key"
    cmd_ssh_key(_KeyArgs(str(key)))
    assert stat.S_IMODE(key.stat().st_mode) == 0o600


def test_ssh_key_reuses_an_existing_key_instead_of_overwriting(tmp_path, capsys):
    if shutil.which("ssh-keygen") is None:
        pytest.skip("ssh-keygen not installed")
    key = tmp_path / "bastion_key"
    cmd_ssh_key(_KeyArgs(str(key)))
    before = key.read_text()
    capsys.readouterr()

    assert cmd_ssh_key(_KeyArgs(str(key))) == 0
    assert "reusing" in capsys.readouterr().out
    # A regenerated key would silently invalidate the key already registered with the
    # bastion, and the failure would show up later as an unexplained auth refusal.
    assert key.read_text() == before


def test_ssh_key_verify_refuses_a_key_that_does_not_exist(tmp_path, capsys):
    # Otherwise `--verify` would report on the *default* key's absence in a confusing way,
    # or worse, pass by falling back to the password.
    rc = cmd_ssh_key(_KeyArgs(str(tmp_path / "nope"), verify=True))
    assert rc == 2
    assert "no key at" in capsys.readouterr().err


# --- `login --sync-ssh-password`: keeping the two planes on one secret -------------
#
# The control plane reads the password saved by `login` (keyring / credentials file); the
# data plane reads `bastion.ssh_password` from config.toml. They are different fields on
# purpose, so rotating the password updates only the first one, and every SSH tool then
# fails with an auth error naming neither file. These tests pin the escape hatch — and,
# just as importantly, that it is opt-in and never fires on its own.


def _login_args(**over):
    class Args:
        username = "someone"
        password = "s3cret"
        save_password = False
        forget = False
        sync_ssh_password = False

    for k, v in over.items():
        setattr(Args, k, v)
    return Args()


@pytest.fixture
def _stub_login(monkeypatch):
    """Make `cmd_login` reachable without a bastion: stub the API and the password store."""

    async def _fake_login(self, username, password):
        return "tok"

    async def _aclose(self):
        return None

    monkeypatch.setattr("jms_bridge.cli.JumpServerAPI.login_with_password", _fake_login)
    monkeypatch.setattr("jms_bridge.cli.JumpServerAPI.aclose", _aclose)
    monkeypatch.setattr("jms_bridge.cli.secrets.get_password", lambda: None)
    monkeypatch.setattr("jms_bridge.cli.secrets.save_password", lambda pw: "nowhere")


def _write_config(tmp_path, monkeypatch, body):
    cfg = tmp_path / "config.toml"
    cfg.write_text(textwrap.dedent(body))
    monkeypatch.setenv("JMS_BRIDGE_CONFIG", str(cfg))
    return cfg


def test_sync_ssh_password_rewrites_the_data_plane_field(
    tmp_path, monkeypatch, _stub_login, capsys
):
    from jms_bridge.cli import cmd_login

    cfg = _write_config(
        tmp_path,
        monkeypatch,
        """\
        # keep me
        [bastion]
        url = "https://bastion.example.com"
        # the data plane credential
        ssh_password = "old-one"
        """,
    )
    assert asyncio.run(cmd_login(_login_args(sync_ssh_password=True))) == 0

    data = read_toml(cfg)
    assert data["bastion"]["ssh_password"] == "s3cret"
    # Rewriting is line-oriented precisely so the user's comments survive; a TOML
    # round-trip would have dropped them.
    assert "# keep me" in cfg.read_text() and "# the data plane credential" in cfg.read_text()
    # The session token still lands, i.e. syncing did not replace the login behaviour.
    assert data["bastion"]["session_token"] == "tok"


def test_sync_ssh_password_backs_up_before_overwriting(tmp_path, monkeypatch, _stub_login):
    from jms_bridge.cli import cmd_login

    _write_config(
        tmp_path,
        monkeypatch,
        """\
        [bastion]
        url = "https://bastion.example.com"
        ssh_password = "old-one"
        """,
    )
    assert asyncio.run(cmd_login(_login_args(sync_ssh_password=True))) == 0

    backups = list(tmp_path.glob("config.toml.bak.*"))
    assert len(backups) == 1
    # If the new password is wrong, the previous one has to still exist somewhere: this
    # command rewrites the file the data plane authenticates with.
    assert read_toml(backups[0])["bastion"]["ssh_password"] == "old-one"


def test_login_without_the_flag_leaves_the_data_plane_alone(
    tmp_path, monkeypatch, _stub_login, capsys
):
    from jms_bridge.cli import cmd_login

    cfg = _write_config(
        tmp_path,
        monkeypatch,
        """\
        [bastion]
        url = "https://bastion.example.com"
        ssh_password = "old-one"
        """,
    )
    assert asyncio.run(cmd_login(_login_args())) == 0

    # Default behaviour is unchanged — the flag is the whole point, and silently
    # overwriting a working data-plane credential would be worse than the drift.
    assert read_toml(cfg)["bastion"]["ssh_password"] == "old-one"
    assert not list(tmp_path.glob("config.toml.bak.*"))
    # But it must not be silent: the mismatch is reported with the remedy.
    assert "--sync-ssh-password" in capsys.readouterr().out


def test_login_skips_the_rewrite_when_already_in_sync(tmp_path, monkeypatch, _stub_login, capsys):
    from jms_bridge.cli import cmd_login

    _write_config(
        tmp_path,
        monkeypatch,
        """\
        [bastion]
        url = "https://bastion.example.com"
        ssh_password = "s3cret"
        """,
    )
    assert asyncio.run(cmd_login(_login_args(sync_ssh_password=True))) == 0

    assert "already matches" in capsys.readouterr().out
    # No backup means no rewrite; an idempotent re-run should not litter the config dir.
    assert not list(tmp_path.glob("config.toml.bak.*"))


def test_sync_ssh_password_escapes_a_value_that_would_break_toml(
    tmp_path, monkeypatch, _stub_login
):
    from jms_bridge.cli import cmd_login

    cfg = _write_config(
        tmp_path,
        monkeypatch,
        """\
        [bastion]
        url = "https://bastion.example.com"
        """,
    )
    # Both a quote and a backslash: the quote needs escaping at all, and the backslash is
    # the one `re.sub` would eat as a replacement-string escape (turning `\\` into `\`
    # and leaving the file unparseable).
    tricky = 'pa"ss\\word'
    assert asyncio.run(cmd_login(_login_args(password=tricky, sync_ssh_password=True))) == 0

    # A quote or backslash written raw would leave config.toml unparseable, and the
    # config loader is the thing every tool call goes through.
    assert read_toml(cfg)["bastion"]["ssh_password"] == tricky


def test_set_bastion_value_survives_replacement_escapes(tmp_path):
    from jms_bridge.cli import _set_bastion_value

    # Regression: the value was spliced in as a `re.sub` *replacement string*, so a
    # backslash in a password was interpreted as an escape (`\\b` -> `\b`) and the
    # rewritten config.toml no longer parsed — bricks the config on a one-character
    # password. Go through the helper directly so this stays pinned even if the CLI
    # plumbing around it changes.
    p = tmp_path / "config.toml"
    p.write_text('[bastion]\nssh_password = "old"\n')
    for value in ("a\\b", 'q"q', 'back\\slash"and\\quote', "\\1", "plain"):
        _set_bastion_value(p, "ssh_password", value)
        assert read_toml(p)["bastion"]["ssh_password"] == value


def test_sync_ssh_password_keeps_the_file_owner_only(tmp_path, monkeypatch, _stub_login):
    from jms_bridge.cli import cmd_login

    cfg = _write_config(
        tmp_path,
        monkeypatch,
        """\
        [bastion]
        url = "https://bastion.example.com"
        ssh_password = "old-one"
        """,
    )
    os.chmod(cfg, 0o600)
    assert asyncio.run(cmd_login(_login_args(sync_ssh_password=True))) == 0
    # The file holds the session token; rewriting it must not be the step that widens it.
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o600


def test_sync_ssh_password_warns_when_the_env_var_wins(tmp_path, monkeypatch, _stub_login, capsys):
    from jms_bridge.cli import cmd_login

    _write_config(
        tmp_path,
        monkeypatch,
        """\
        [bastion]
        url = "https://bastion.example.com"
        ssh_password = "old-one"
        """,
    )
    # JMS_BRIDGE_SSH_PASSWORD outranks the file (config.load_config), so a user who
    # exported it would otherwise watch the sync "work" and the data plane stay broken.
    monkeypatch.setenv("JMS_BRIDGE_SSH_PASSWORD", "exported")
    assert asyncio.run(cmd_login(_login_args(sync_ssh_password=True))) == 0
    assert "JMS_BRIDGE_SSH_PASSWORD is exported" in capsys.readouterr().out
