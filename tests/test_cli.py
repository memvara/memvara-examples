"""The command: menu, check, and a chat driven end to end with a scripted model."""

import io

from memvara_examples import cli
from memvara_examples.model import ScriptedModel


def test_menu_accepts_a_number_a_name_or_quit():
    out = io.StringIO()
    def feed(*lines):
        it = iter(lines)
        return lambda _prompt: next(it)

    assert cli.choose_agent(inp=feed("2"), out=out) == "engineer"
    assert cli.choose_agent(inp=feed("assistant"), out=out) == "assistant"
    assert cli.choose_agent(inp=feed("9", "q"), out=out) is None
    assert "1. assistant" in out.getvalue() and "2. engineer" in out.getvalue()


def test_check_reports_a_local_store(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    args = cli.build_parser().parse_args(["check"])
    args.user, args.local = "alice", str(tmp_path / "m.db")
    out = io.StringIO()
    assert cli.check(args, out=out) == 0
    text = out.getvalue()
    assert "Memvara: ok" in text and "0 facts visible" in text
    assert "no ANTHROPIC_API_KEY" in text


def test_chat_runs_a_turn_and_the_standing_command(tmp_path):
    args = cli.build_parser().parse_args(
        ["assistant", "--local", str(tmp_path / "m.db"), "--user", "alice"])
    model = ScriptedModel([
        [("memory_remember", {"predicate": "prefers", "object": "tea over coffee",
                              "memory_type": "procedural"})],
        "Noted: tea over coffee.",
    ])
    out = io.StringIO()
    agent = cli.make_agent("assistant", args, model=model, out=out)
    lines = iter(["I prefer tea over coffee", "/standing", "/quit"])
    cli.chat(agent, inp=lambda _: next(lines), out=out)
    text = out.getvalue()
    assert "[memory_remember(" in text
    assert "assistant> Noted: tea over coffee." in text
    assert "- prefers: tea over coffee" in text


def test_main_check_uses_the_parser(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MEMVARA_USER", "alice")
    assert cli.main(["check", "--local", str(tmp_path / "m.db")]) == 0
    assert "Memvara: ok" in capsys.readouterr().out
