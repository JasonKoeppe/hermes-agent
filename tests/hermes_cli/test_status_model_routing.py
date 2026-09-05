"""Model routing must be visible per surface, not just as one global default.

Regression cover for the 2026-09-04 hardening brief item 3: config said
`model.default: claude-opus-5` while Telegram ran `gpt-5.6-sol` and crons ran
`gpt-5.6-terra`, and `hermes status` showed only the global default, so the
divergence was invisible.
"""
import pytest

from hermes_cli import status as status_mod


@pytest.fixture
def no_live_sessions(monkeypatch):
    monkeypatch.setattr(status_mod, "_last_observed_model_by_source", lambda: {})


def _render(capsys, config, monkeypatch, live=None):
    monkeypatch.setattr(
        status_mod, "_last_observed_model_by_source", lambda: (live or {})
    )
    status_mod._print_model_routing(config)
    return capsys.readouterr().out


def test_reports_each_surface(capsys, monkeypatch):
    out = _render(
        capsys,
        {"model": {"default": "claude-opus-5"}, "cron": {"model": "gpt-5.6-terra"}},
        monkeypatch,
        live={"telegram": "gpt-5.6-sol"},
    )
    assert "Default (config)" in out and "claude-opus-5" in out
    assert "Telegram" in out and "gpt-5.6-sol" in out
    assert "Cron" in out and "gpt-5.6-terra" in out
    assert "Ledger workers" in out


def test_chat_override_is_labelled_as_such(capsys, monkeypatch):
    out = _render(
        capsys,
        {"model": {"default": "claude-opus-5"}},
        monkeypatch,
        live={"telegram": "gpt-5.6-sol"},
    )
    assert "chat override" in out


def test_surface_matching_default_is_not_called_an_override(capsys, monkeypatch):
    out = _render(
        capsys,
        {"model": {"default": "claude-opus-5"}},
        monkeypatch,
        live={"telegram": "claude-opus-5"},
    )
    assert "chat override" not in out


def test_never_labels_a_divergent_observed_model_as_inheriting(capsys, monkeypatch):
    """A subagent on a different model must not read as 'inherits default'."""
    out = _render(
        capsys,
        {"model": {"default": "claude-opus-5"}},
        monkeypatch,
        live={"subagent": "gpt-5.6-sol"},
    )
    line = next(l for l in out.splitlines() if "Delegation" in l)
    assert "gpt-5.6-sol" in line
    assert "inherits default" not in line


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"model": "claude-opus-5"},          # section is a scalar, not a mapping
        {"model": {"default": "x"}, "cron": "nope"},
        {"cron": None},
    ],
)
def test_tolerates_malformed_config(capsys, monkeypatch, config):
    """Sections can be scalars or missing; status must still render."""
    out = _render(capsys, config, monkeypatch)
    assert "Model routing by surface" in out


def test_live_lookup_returns_empty_when_state_db_is_unreadable(monkeypatch, tmp_path):
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: str(tmp_path))
    assert status_mod._last_observed_model_by_source() == {}
