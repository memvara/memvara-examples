"""The ``memvara-examples`` command.

Run it with no arguments and it lists the agents and asks which one to start. Run it
with an agent name to start that one directly. ``check`` tests the Memvara connection
and the Anthropic credential without starting anything.

    memvara-examples                  # menu
    memvara-examples assistant        # start the personal assistant
    memvara-examples engineer --project checkout
    memvara-examples check
    memvara-examples assistant --local ./memory.db   # no account, no network
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from typing import Any

from . import __version__
from .agents import AGENTS
from .agents.base import Agent
from .memory import Memory
from .model import DEFAULT_MODEL, ClaudeModel, Model

HELP = """Commands inside a chat:
  /help       show this
  /standing   show the standing preferences memory holds for you
  /quit       leave (Ctrl-D works too)"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memvara-examples",
        description="Start an AI agent that keeps its memory in Memvara.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    _add_common(sub.add_parser(
        "check", help="Test the Memvara connection and the Anthropic credential."),
        agent=False)
    for name, cls in AGENTS.items():
        p = sub.add_parser(name, help=cls.description)
        _add_common(p)
        if name == "engineer":
            p.add_argument("--project", default="project",
                           help="The project whose record this agent keeps. Default 'project'.")
    # The menu (no subcommand) takes the same options.
    _add_common(parser)
    return parser


def _add_common(p: argparse.ArgumentParser, *, agent: bool = True) -> None:
    """The memory options every command takes, plus the agent options for the ones
    that start an agent. ``check`` opens a store but never a model."""
    p.add_argument("--user", default=os.environ.get("MEMVARA_USER") or getpass.getuser(),
                   help="Whose memory to open. Default: MEMVARA_USER or your login name.")
    p.add_argument("--local", metavar="PATH", default=None,
                   help="Use a local SQLite store at PATH instead of the hosted deployment.")
    if not agent:
        return
    p.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
                   help=f"Claude model id. Default {DEFAULT_MODEL}.")
    p.add_argument("--quiet", action="store_true",
                   help="Do not print the tool calls the agent makes.")


def check(args: argparse.Namespace, out: Any = None) -> int:
    """Report whether the two things an agent needs are in place."""
    out = out or sys.stdout
    ok = True
    try:
        memory = Memory.open(user=args.user, local=args.local)
        try:
            info = memory.describe()
        finally:
            memory.close()
        print(f"Memvara: ok. {info['store']}, user {info['user']}, "
              f"{info['claims_visible']} facts visible, extractor {info['extractor']}.",
              file=out)
        if info.get("read_only"):
            print("  This credential is read-only; the agents will not be able to write.",
                  file=out)
    except Exception as exc:
        ok = False
        print(f"Memvara: NOT ok. {exc}", file=out)
    if anthropic_credential_present():
        print("Anthropic: credential found.", file=out)
    else:
        print("Anthropic: no credential found. Export ANTHROPIC_API_KEY (or run "
              "`ant auth login`) before starting an agent.", file=out)
    return 0 if ok else 1


def anthropic_credential_present() -> bool:
    """Whether the Anthropic SDK can find a credential, asked of the SDK itself.

    The client resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN and a profile written by
    ``ant auth login`` in its own order. Constructing it makes no network call, so this
    reads back what it resolved rather than re-implementing that order here.
    """
    import anthropic

    client = anthropic.Anthropic()
    return bool(client.api_key or client.auth_token)


def choose_agent(inp: Any = input, out: Any = sys.stdout) -> str | None:
    """The menu. Returns an agent name, or None if the user leaves."""
    names = list(AGENTS)
    print("Which agent do you want to start?\n", file=out)
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name:<10} {AGENTS[name].description}", file=out)
    print(f"  q. quit\n", file=out)
    while True:
        try:
            raw = inp("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw in ("q", "quit", "exit", ""):
            return None
        if raw in names:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            return names[int(raw) - 1]
        print(f"Type a number from 1 to {len(names)}, an agent name, or q.", file=out)


def make_agent(name: str, args: argparse.Namespace, *, model: Model | None = None,
               out: Any = sys.stdout) -> Agent:
    memory = Memory.open(user=args.user, local=args.local)
    try:
        model = model or ClaudeModel(args.model)
    except Exception:
        memory.close()
        raise

    def trace(tool: str, arguments: dict[str, Any], result: str) -> None:
        if getattr(args, "quiet", False):
            return
        shown = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        first = result.strip().splitlines()[0] if result.strip() else ""
        print(f"    [{tool}({shown}) -> {first[:120]}]", file=out)

    extra: dict[str, Any] = {}
    if name == "engineer":
        extra["project"] = getattr(args, "project", "project")
    return AGENTS[name](memory, model, trace=trace, **extra)


def chat(agent: Agent, *, inp: Any = input, out: Any = sys.stdout) -> None:
    """The read-reply loop, until /quit or end of input."""
    info = agent.memory.describe()
    print(f"\n{agent.name}: {agent.description}", file=out)
    print(f"Memory: {info['store']}, user {info['user']}, "
          f"{info['claims_visible']} facts visible.", file=out)
    print("Type /help for commands.\n", file=out)
    while True:
        try:
            text = inp("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=out)
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/help":
            print(HELP, file=out)
            continue
        if text == "/standing":
            rows = agent.memory.standing()
            if not rows:
                print("No standing preferences stored yet.", file=out)
            for c in rows:
                print(f"  - {c.predicate}: {c.object}", file=out)
            continue
        try:
            reply = agent.turn(text)
        except Exception as exc:  # keep the session alive; the user can retry
            print(f"error: {type(exc).__name__}: {exc}", file=out)
            continue
        print(f"\n{agent.name}> {reply}\n", file=out)
    agent.memory.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        return check(args)
    name = args.command or choose_agent()
    if name is None:
        return 0
    try:
        agent = make_agent(name, args)
    except Exception as exc:
        print(f"Could not start {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Run `memvara-examples check` to see what is missing.", file=sys.stderr)
        return 1
    chat(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
