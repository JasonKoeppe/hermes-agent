"""Imported sessions are reference material, not live threads.

Regression cover for the 2026-09-04 hardening brief item 2: session
20260904_162527_e210ea was an imported Claude Code transcript continued by the
desktop agent, which then did code edits, a merge, a production deploy and a
client email inline before hitting agent.max_turns and run_budget_seconds.
"""
import json

import pytest

from hermes_cli import foreign_sessions as fs

WORKER_PREAMBLE = (
    "You are a detached worker for ledger task #28 (\"Build the thing\") on "
    "Jason's Mac mini, dispatched by Finch.\nWorking directory: /Users/x\n"
)


class _FakeDB:
    def __init__(self):
        self.sessions = {}
        self.messages = []
        self.titles = {}

    def create_session(self, session_id, source=None, cwd=None, origin_json=None):
        self.sessions[session_id] = {
            "source": source, "cwd": cwd,
            "origin": json.loads(origin_json) if origin_json else {},
        }

    def append_message(self, session_id, role, content):
        self.messages.append((session_id, role, content))

    def set_session_title(self, session_id, title):
        self.titles[session_id] = title

    def close(self):
        pass


def _import(tmp_path, monkeypatch, first_user, allow_continue=False):
    turns = [
        {"role": "user", "content": first_user},
        {"role": "assistant", "content": "ok"},
    ]
    monkeypatch.setattr(
        fs, "parse_claude_session",
        lambda path: {"turns": turns, "session_id": "abc", "cwd": "/tmp"},
    )
    f = tmp_path / "s.jsonl"
    f.write_text("{}\n")
    db = _FakeDB()
    sid = fs.import_foreign_session("claude", f, db=db, allow_continue=allow_continue)
    return db.sessions[sid]["origin"]


def test_detects_a_ledger_worker_preamble():
    assert fs.looks_like_ledger_worker_transcript(
        [{"role": "user", "content": WORKER_PREAMBLE}]
    )


def test_ignores_an_ordinary_first_turn():
    assert not fs.looks_like_ledger_worker_transcript(
        [{"role": "user", "content": "hey, can you look at the invoice?"}]
    )


def test_looks_only_at_the_first_user_turn():
    """A later mention must not retroactively mark a normal session."""
    assert not fs.looks_like_ledger_worker_transcript([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": WORKER_PREAMBLE},
    ])


def test_import_is_read_only_by_default(tmp_path, monkeypatch):
    origin = _import(tmp_path, monkeypatch, "just a normal conversation")
    assert origin["read_only"] is True
    assert "clone it explicitly" in origin["read_only_reason"]


def test_worker_transcript_gets_the_ledger_reason(tmp_path, monkeypatch):
    origin = _import(tmp_path, monkeypatch, WORKER_PREAMBLE)
    assert origin["read_only"] is True
    assert origin["ledger_worker_transcript"] is True
    assert "dispatch a new ledger task" in origin["read_only_reason"]


def test_allow_continue_opts_out(tmp_path, monkeypatch):
    origin = _import(tmp_path, monkeypatch, "normal", allow_continue=True)
    assert "read_only" not in origin


def test_allow_continue_still_records_worker_provenance(tmp_path, monkeypatch):
    """Opting in must not erase the fact that this was a worker transcript."""
    origin = _import(tmp_path, monkeypatch, WORKER_PREAMBLE, allow_continue=True)
    assert origin["ledger_worker_transcript"] is True
    assert "read_only" not in origin


def test_import_provenance_is_preserved(tmp_path, monkeypatch):
    origin = _import(tmp_path, monkeypatch, "normal")
    assert origin["imported_from"]["foreign_session_id"] == "abc"


@pytest.mark.parametrize("turns", [None, [], [{"role": "assistant", "content": "x"}]])
def test_detector_tolerates_missing_user_turns(turns):
    assert fs.looks_like_ledger_worker_transcript(turns) is False
