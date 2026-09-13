"""The command: menu, check, and a chat driven end to end with a scripted model."""

import io

import pytest

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


MODEL_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "OPENAI_BASE_URL",
              "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY")


@pytest.fixture
def no_model_env(monkeypatch):
    for name in MODEL_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_check_reports_a_local_store_and_a_missing_credential(tmp_path, no_model_env):
    args = cli.build_parser().parse_args(["check", "--local", str(tmp_path / "m.db"),
                                          "--user", "alice"])
    out = io.StringIO()
    assert cli.check(args, out=out) == 1
    text = out.getvalue()
    assert "Memvara: ok" in text and "0 facts visible" in text
    assert "Model: NOT ok" in text and "provider anthropic" in text


def test_check_sees_a_credential_the_sdk_resolves(tmp_path, no_model_env):
    no_model_env.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    args = cli.build_parser().parse_args(["check", "--local", str(tmp_path / "m.db")])
    out = io.StringIO()
    assert cli.check(args, out=out) == 0
    assert "Model: ok. provider anthropic, model claude-opus-5" in out.getvalue()


def test_check_accepts_an_openai_format_endpoint_from_flags(tmp_path, no_model_env):
    args = cli.build_parser().parse_args([
        "check", "--local", str(tmp_path / "m.db"), "--provider", "openai",
        "--base-url", "http://localhost:11434/v1", "--model", "qwen3:8b"])
    out = io.StringIO()
    assert cli.check(args, out=out) == 0
    assert ("Model: ok. provider openai, model qwen3:8b, endpoint http://localhost:11434/v1"
            in out.getvalue())


def test_check_says_when_an_openai_endpoint_has_no_model_named(tmp_path, no_model_env):
    no_model_env.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    args = cli.build_parser().parse_args(["check", "--local", str(tmp_path / "m.db")])
    out = io.StringIO()
    assert cli.check(args, out=out) == 1
    assert "Model: NOT ok. No model named for provider openai" in out.getvalue()


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


def test_main_check_uses_the_parser(tmp_path, no_model_env, capsys):
    no_model_env.setenv("MEMVARA_USER", "alice")
    no_model_env.setenv("LLM_API_KEY", "k")
    assert cli.main(["check", "--local", str(tmp_path / "m.db")]) == 0
    assert "Memvara: ok" in capsys.readouterr().out
