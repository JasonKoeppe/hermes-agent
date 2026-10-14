"""HERMES_DENY_CLAUDE_SUBSCRIPTION_OAUTH: refuse Claude subscription OAuth outright.

Anthropic limits Claude Pro/Max subscription logins to its own apps. With the
switch on, Hermes must never resolve, seed or send a subscription token and
must use Anthropic API keys only. With the switch off, upstream behavior is
unchanged.
"""

from pathlib import Path

import pytest

FAKE_OAUTH = "sk-ant-oat01-" + "x" * 40
FAKE_API_KEY = "sk-ant-api03-" + "y" * 40
DENY = "HERMES_DENY_CLAUDE_SUBSCRIPTION_OAUTH"


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", DENY):
        monkeypatch.delenv(key, raising=False)
    return hermes_home


def test_switch_is_off_by_default(home):
    from agent.anthropic_credentials import claude_subscription_oauth_denied

    assert claude_subscription_oauth_denied() is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_switch_reads_process_env(home, monkeypatch, value):
    from agent.anthropic_credentials import claude_subscription_oauth_denied

    monkeypatch.setenv(DENY, value)
    assert claude_subscription_oauth_denied() is True


def test_switch_reads_hermes_dotenv(home):
    from agent.anthropic_credentials import claude_subscription_oauth_denied

    (home / ".env").write_text(f"{DENY}=1\n")
    assert claude_subscription_oauth_denied() is True


def test_resolver_returns_api_key_and_ignores_oauth_env(home, monkeypatch):
    import agent.anthropic_credentials as creds

    monkeypatch.setenv(DENY, "1")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", FAKE_OAUTH)
    monkeypatch.setenv("ANTHROPIC_TOKEN", FAKE_OAUTH)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    assert creds.resolve_anthropic_token() == FAKE_API_KEY


def test_resolver_never_reads_claude_code_credentials(home, monkeypatch):
    import agent.anthropic_credentials as creds

    def boom(*_a, **_k):
        raise AssertionError("Claude Code credentials must not be read")

    monkeypatch.setattr(creds, "read_claude_code_credentials", boom)
    monkeypatch.setattr(creds, "_resolve_anthropic_pool_token", boom)
    monkeypatch.setenv(DENY, "1")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", FAKE_OAUTH)
    assert creds.resolve_anthropic_token() is None


def test_resolver_unchanged_when_switch_off(home, monkeypatch):
    import agent.anthropic_credentials as creds

    monkeypatch.setattr(creds, "read_claude_code_credentials", lambda: None)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", FAKE_OAUTH)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    assert creds.resolve_anthropic_token() == FAKE_OAUTH


def test_client_builder_refuses_oauth_token(home, monkeypatch):
    from agent.anthropic_adapter import build_anthropic_client

    monkeypatch.setenv(DENY, "1")
    with pytest.raises(ValueError, match="disabled"):
        build_anthropic_client(FAKE_OAUTH)


def test_client_builder_accepts_api_key(home, monkeypatch):
    from agent.anthropic_adapter import build_anthropic_client

    monkeypatch.setenv(DENY, "1")
    assert build_anthropic_client(FAKE_API_KEY) is not None


def test_env_seeding_skips_subscription_tokens(home, monkeypatch):
    from agent import credential_pool

    monkeypatch.setenv(DENY, "1")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", FAKE_OAUTH)
    monkeypatch.setenv("ANTHROPIC_TOKEN", FAKE_OAUTH)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    entries = []
    _changed, sources = credential_pool._seed_from_env("anthropic", entries)
    assert sources == {"env:ANTHROPIC_API_KEY"}
    assert [e.source for e in entries] == ["env:ANTHROPIC_API_KEY"]


def test_singleton_seeding_drops_autodiscovered_oauth(home, monkeypatch):
    from agent import credential_pool

    monkeypatch.setenv(DENY, "1")
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured", lambda _p: True
    )

    class Entry:
        def __init__(self, source):
            self.source = source

    entries = [Entry("claude_code"), Entry("hermes_pkce"), Entry("manual")]
    changed, sources = credential_pool._seed_from_singletons("anthropic", entries)
    assert changed is True
    assert sources == set()
    assert [e.source for e in entries] == ["manual"]


def test_switch_file_value_wins_over_false_process_value(home, monkeypatch):
    from agent.anthropic_credentials import claude_subscription_oauth_denied

    monkeypatch.setenv(DENY, "0")
    (home / ".env").write_text(f"{DENY}=1\n")
    assert claude_subscription_oauth_denied() is True


def test_claude_code_credentials_are_never_read(home, monkeypatch):
    import agent.anthropic_credentials as creds

    def boom(*_a, **_k):
        raise AssertionError("keychain must not be read")

    monkeypatch.setattr(creds, "_read_claude_code_credentials_from_keychain", boom)
    monkeypatch.setenv(DENY, "1")
    assert creds.read_claude_code_credentials() is None


def test_refresh_and_login_are_refused(home, monkeypatch):
    import agent.anthropic_credentials as creds

    def no_network(*_a, **_k):
        raise AssertionError("no OAuth network call may be made")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setenv(DENY, "1")
    with pytest.raises(ValueError, match="disabled"):
        creds.refresh_anthropic_oauth_pure("refresh-token")
    assert creds._refresh_oauth_token({"refreshToken": "refresh-token"}) is None
    assert creds.run_hermes_oauth_login_pure() is None
    assert creds.run_oauth_setup_token() is None


def test_persisted_oauth_pool_entries_are_dropped_on_load(home, monkeypatch):
    import json

    from agent import credential_pool

    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "credential_pool": {"anthropic": [
            {"id": "oat1", "label": "subscription", "auth_type": "oauth",
             "source": "manual:hermes_pkce", "access_token": FAKE_OAUTH, "priority": 0},
            {"id": "env1", "label": "CLAUDE_CODE_OAUTH_TOKEN", "auth_type": "api_key",
             "source": "env:CLAUDE_CODE_OAUTH_TOKEN", "access_token": FAKE_OAUTH, "priority": 1},
            {"id": "key1", "label": "Finch Hermes", "auth_type": "api_key",
             "source": "manual", "access_token": FAKE_API_KEY, "priority": 2},
        ]},
    }))
    monkeypatch.setenv(DENY, "1")
    pool = credential_pool.load_pool("anthropic")
    assert [e.id for e in pool._entries] == ["key1"]
    stored = json.loads((home / "auth.json").read_text())["credential_pool"]["anthropic"]
    assert [e["id"] for e in stored] == ["key1"]


def test_persisted_oauth_pool_entries_kept_when_switch_off(home):
    import json

    from agent import credential_pool

    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "credential_pool": {"anthropic": [
            {"id": "oat1", "label": "subscription", "auth_type": "oauth",
             "source": "manual:hermes_pkce", "access_token": FAKE_OAUTH, "priority": 0},
            {"id": "key1", "label": "Finch Hermes", "auth_type": "api_key",
             "source": "manual", "access_token": FAKE_API_KEY, "priority": 1},
        ]},
    }))
    pool = credential_pool.load_pool("anthropic")
    assert {e.id for e in pool._entries} == {"oat1", "key1"}


def test_context_length_probe_skips_oauth(home, monkeypatch):
    from agent import model_metadata

    def no_network(*_a, **_k):
        raise AssertionError("no request with a subscription token")

    monkeypatch.setenv(DENY, "1")
    monkeypatch.setattr(model_metadata, "_ensure_requests", no_network)
    assert model_metadata._query_anthropic_context_length(
        "claude-opus-5", "https://api.anthropic.com", "cc-" + "z" * 30
    ) is None


def test_hermes_saved_oauth_is_not_read(home, monkeypatch):
    import json

    import agent.anthropic_credentials as creds

    (home / ".anthropic_oauth.json").write_text(json.dumps({"accessToken": FAKE_OAUTH}))
    monkeypatch.setenv(DENY, "1")
    assert creds.read_hermes_oauth_credentials() is None


def test_setup_flow_and_config_writers_refuse(home, monkeypatch):
    from hermes_cli import config
    from hermes_cli.main import _run_anthropic_oauth_flow

    saved = {}
    monkeypatch.setenv(DENY, "1")
    assert _run_anthropic_oauth_flow(lambda k, v: saved.__setitem__(k, v)) is False
    with pytest.raises(ValueError, match="disabled"):
        config.save_anthropic_oauth_token(FAKE_OAUTH, save_fn=lambda k, v: saved.__setitem__(k, v))
    with pytest.raises(ValueError, match="disabled"):
        config.use_anthropic_claude_code_credentials(save_fn=lambda k, v: saved.__setitem__(k, v))
    assert saved == {}


def test_catalog_never_sends_pool_oauth_token(home, monkeypatch):
    from hermes_cli import models

    monkeypatch.setenv(DENY, "1")
    monkeypatch.setattr(models, "_resolve_anthropic_pool_catalog_credentials",
                        lambda: (FAKE_OAUTH, "https://api.anthropic.com"))

    def no_network(*_a, **_k):
        raise AssertionError("no catalog request with a subscription token")

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", no_network)
    assert models._fetch_anthropic_models(api_key="", base_url="https://api.anthropic.com") is None


def test_root_policy_applies_inside_a_profile(tmp_path, monkeypatch):
    from agent.anthropic_credentials import claude_subscription_oauth_denied

    root = tmp_path / ".hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    (root / ".env").write_text(f"{DENY}=1\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.delenv(DENY, raising=False)
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)
    assert claude_subscription_oauth_denied() is True


def test_api_key_save_rejects_subscription_token(home, monkeypatch):
    from hermes_cli import config

    saved = {}
    monkeypatch.setenv(DENY, "1")
    with pytest.raises(ValueError, match="subscription token"):
        config.save_anthropic_api_key(FAKE_OAUTH, save_fn=lambda k, v: saved.__setitem__(k, v))
    config.save_anthropic_api_key(FAKE_API_KEY, save_fn=lambda k, v: saved.__setitem__(k, v))
    assert saved == {"ANTHROPIC_API_KEY": FAKE_API_KEY, "ANTHROPIC_TOKEN": ""}
