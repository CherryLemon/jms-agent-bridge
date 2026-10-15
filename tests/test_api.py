import httpx
import pytest

from jms_bridge.api import JumpServerAPI, JumpServerError
from jms_bridge.config import BastionConfig

# A fixed date so the signature is reproducible in tests.
DATE = "Thu, 09 Oct 2026 00:00:00 GMT"


def keyed(**kw):
    return JumpServerAPI(
        BastionConfig(url="https://x", access_key_id="AKID", access_key_secret="SECRET", **kw)
    )


def test_signature_header_shape():
    auth, headers = keyed().sign_access_key(
        "GET", "/api/v1/perms/users/my/assets/", date=DATE, accept="application/json", org=None
    )
    assert auth.startswith('Signature keyId="AKID",algorithm="hmac-sha256",')
    assert 'headers="(request-target) date accept"' in auth
    assert 'signature="' in auth
    assert headers == {"date": DATE, "accept": "application/json"}


def test_signature_binds_method_path_and_org():
    api = keyed()
    a, _ = api.sign_access_key("GET", "/p", date=DATE, accept="application/json", org=None)
    b, _ = api.sign_access_key("POST", "/p", date=DATE, accept="application/json", org=None)
    c, _ = api.sign_access_key("GET", "/q", date=DATE, accept="application/json", org=None)
    d, hd = api.sign_access_key("GET", "/p", date=DATE, accept="application/json", org="ORG")
    assert len({a, b, c, d}) == 4  # every input changes the signature
    assert "x-jms-org" in hd and hd["x-jms-org"] == "ORG"
    assert 'headers="(request-target) date accept x-jms-org"' in d


def test_signature_matches_pyhttpsig_library():
    """Cross-check our hand-rolled signer against the library JumpServer verifies with."""
    httpsig = pytest.importorskip("httpsig")

    api = keyed()
    auth, _ = api.sign_access_key(
        "GET", "/api/v1/assets/assets/", date=DATE, accept="application/json", org=None
    )

    signer = httpsig.HeaderSigner(
        "AKID",
        "SECRET",
        "hmac-sha256",
        headers=["(request-target)", "date", "accept"],
        sign_header="authorization",
    )
    expected = signer.sign(
        {"date": DATE, "accept": "application/json"},
        method="GET",
        path="/api/v1/assets/assets/",
    )
    assert auth == expected["authorization"]


def test_auth_headers_prefers_access_key():
    headers = keyed(private_token="TOK")._auth_headers("GET", "/p")
    assert headers["Authorization"].startswith("Signature")


def test_private_token_uses_token_scheme():
    api = JumpServerAPI(BastionConfig(url="https://x", private_token="TOK"))
    assert api._auth_headers("GET", "/p")["Authorization"] == "Token TOK"


def test_session_token_uses_bearer_scheme():
    api = JumpServerAPI(BastionConfig(url="https://x", private_token="TOK"))
    api._token = "SESSION"
    assert api._auth_headers("GET", "/p")["Authorization"] == "Bearer SESSION"


def test_cached_session_token_uses_bearer_and_beats_private_token():
    """A token cached by `jms-bridge login` is sent as Bearer, taking precedence."""
    api = JumpServerAPI(BastionConfig(url="https://x", session_token="SESS", private_token="TOK"))
    assert api._auth_headers("GET", "/p")["Authorization"] == "Bearer SESS"


def test_live_token_beats_cached_session_token():
    api = JumpServerAPI(BastionConfig(url="https://x", session_token="CACHED"))
    api._token = "LIVE"
    assert api._auth_headers("GET", "/p")["Authorization"] == "Bearer LIVE"


def test_no_credentials_raises():
    api = JumpServerAPI(BastionConfig(url="https://x"))
    with pytest.raises(JumpServerError):
        api._auth_headers("GET", "/p")


def test_create_ticket_uses_open_action(monkeypatch):
    """Tickets are created on the subtype collection's ``open`` action."""
    api = JumpServerAPI(BastionConfig(url="https://x", org_id="O1"))
    captured = {}

    async def fake_request(method, path, *, params=None, json_body=None):
        captured.update(method=method, path=path, json=json_body)
        return {"id": "T1"}

    monkeypatch.setattr(api, "request", fake_request)
    import asyncio

    asyncio.run(api.create_ticket(title="need login", ticket_type="login_asset_confirm"))
    assert captured["method"] == "POST"
    assert captured["path"] == "tickets/apply-login-asset-tickets/open/"
    assert captured["json"]["type"] == "login_asset_confirm"
    assert captured["json"]["org_id"] == "O1"


def test_unknown_ticket_type_raises():
    import asyncio

    api = JumpServerAPI(BastionConfig(url="https://x"))
    with pytest.raises(JumpServerError):
        asyncio.run(api.create_ticket(title="x", ticket_type="nope"))


# --- transport failures are normalised to JumpServerError ---------------------------


def test_transport_failure_is_a_jumpserver_error(monkeypatch):
    """A DNS/refused/TLS failure must be a `JumpServerError`, not the raw httpx one.

    The MCP tools (and `doctor`, and `login`) render `JumpServerError` as a message; the
    underlying `httpx.ConnectError` would reach the agent as the opaque "Error executing
    tool <name>", which is the first thing a mistyped bastion URL produces.
    """
    import asyncio

    api = JumpServerAPI(BastionConfig(url="https://bastion.invalid", private_token="TOK"))

    class _BoomClient:
        async def request(self, *a, **k):
            raise httpx.ConnectError("nodename nor servname provided")

    async def fake_client():
        return _BoomClient()

    monkeypatch.setattr(api, "client", fake_client)
    with pytest.raises(JumpServerError, match="ConnectError"):
        asyncio.run(api.request("GET", "perms/users/my/assets/"))


def test_ping_transport_failure_is_a_jumpserver_error(monkeypatch):
    import asyncio

    api = JumpServerAPI(BastionConfig(url="https://bastion.invalid", private_token="TOK"))

    class _BoomClient:
        async def get(self, *a, **k):
            raise httpx.ConnectTimeout("timed out")

    async def fake_client():
        return _BoomClient()

    monkeypatch.setattr(api, "client", fake_client)
    with pytest.raises(JumpServerError, match="ConnectTimeout"):
        asyncio.run(api.ping())


def test_login_transport_failure_is_a_jumpserver_error(monkeypatch):
    import asyncio

    api = JumpServerAPI(BastionConfig(url="https://bastion.invalid"))

    class _BoomClient:
        async def post(self, *a, **k):
            raise httpx.ConnectError("connection refused")

    async def fake_client():
        return _BoomClient()

    monkeypatch.setattr(api, "client", fake_client)
    with pytest.raises(JumpServerError, match="login failed"):
        asyncio.run(api.login_with_password("u", "p"))


def test_control_plane_tools_turn_a_transport_failure_into_a_readable_refusal(monkeypatch):
    """The three REST tools must return `{ok: false, error: ...}`, never raise."""
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import jms_bridge.server as S

    api = JumpServerAPI(BastionConfig(url="https://bastion.invalid", private_token="TOK"))

    class _BoomClient:
        async def request(self, *a, **k):
            raise httpx.ConnectError("nodename nor servname provided")

    async def fake_client():
        return _BoomClient()

    monkeypatch.setattr(api, "client", fake_client)
    monkeypatch.setitem(S._state, "api", api)

    for coro in (S.jms_list_assets(), S.jms_audit_query(), S.jms_create_ticket(title="t")):
        r = asyncio.run(coro)
        assert r["ok"] is False
        assert "ConnectError" in r["error"]


# --- asset listing must not silently truncate ---------------------------------------


def test_list_assets_follows_pagination(monkeypatch):
    """A second page must be fetched, not dropped.

    The list used to stop at the first ``limit``-sized page, so a fleet larger than the
    limit silently reported a partial inventory as though it were complete.
    """
    import asyncio

    api = JumpServerAPI(BastionConfig(url="https://x", private_token="TOK"))
    calls = []

    async def fake_request(method, path, *, params=None, json_body=None):
        calls.append(dict(params or {}))
        offset = (params or {}).get("offset", 0)
        if offset == 0:
            return {
                "count": 3,
                "next": "https://x/page2",
                "results": [{"id": "a"}, {"id": "b"}],
            }
        return {"count": 3, "next": None, "results": [{"id": "c"}]}

    monkeypatch.setattr(api, "request", fake_request)
    assets, truncated = asyncio.run(api.list_assets(limit=2))
    assert [a["id"] for a in assets] == ["a", "b", "c"]
    assert truncated is False, "a full walk must not report truncation"
    assert [c.get("offset", 0) for c in calls] == [0, 2]


def test_list_assets_stops_on_a_bare_list(monkeypatch):
    """A deployment without a pagination envelope still returns its (single) page."""
    import asyncio

    api = JumpServerAPI(BastionConfig(url="https://x", private_token="TOK"))
    calls = []

    async def fake_request(method, path, *, params=None, json_body=None):
        calls.append(dict(params or {}))
        return [{"id": "a"}, {"id": "b"}]

    monkeypatch.setattr(api, "request", fake_request)
    assets, truncated = asyncio.run(api.list_assets(limit=2))
    assert [a["id"] for a in assets] == ["a", "b"]
    assert truncated is False
    assert len(calls) == 1  # no second request for a bare list


def test_list_assets_bounds_the_walk_and_says_it_was_truncated(monkeypatch):
    """A server that always claims another page must not loop forever — and the caller
    must be told the walk stopped early, not handed a prefix as though it were complete.

    This is the bug the truncation flag exists for: the first version of this fix bounded
    the walk but kept the truncation silent, so a 5000-asset deployment returned 2000 with
    nothing to say so.
    """
    import asyncio

    api = JumpServerAPI(BastionConfig(url="https://x", private_token="TOK"))
    calls = []

    async def fake_request(method, path, *, params=None, json_body=None):
        calls.append(1)
        return {"count": 9999, "next": "https://x/always", "results": [{"id": "x"}]}

    monkeypatch.setattr(api, "request", fake_request)
    assets, truncated = asyncio.run(api.list_assets(limit=1, max_pages=3))
    assert len(calls) == 3, "the runaway guard must still stop the walk"
    assert truncated is True, "stopping early while `next` remains must be reported"
    assert len(assets) == 3


def test_list_assets_reports_no_truncation_on_the_final_page(monkeypatch):
    """Ending exactly on the page limit is not truncation if the server is done."""
    import asyncio

    api = JumpServerAPI(BastionConfig(url="https://x", private_token="TOK"))

    async def fake_request(method, path, *, params=None, json_body=None):
        offset = (params or {}).get("offset", 0)
        if offset == 0:
            return {"count": 2, "next": "https://x/p2", "results": [{"id": "a"}]}
        return {"count": 2, "next": None, "results": [{"id": "b"}]}

    monkeypatch.setattr(api, "request", fake_request)
    assets, truncated = asyncio.run(api.list_assets(limit=1, max_pages=2))
    assert [a["id"] for a in assets] == ["a", "b"]
    assert truncated is False


def test_api_does_not_export_the_json_module():
    """`json` was in `__all__` only because a parameter shadowed the module."""
    import jms_bridge.api as api

    assert "json" not in api.__all__