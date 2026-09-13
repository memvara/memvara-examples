"""The model layer: Claude through the Anthropic SDK, behind a small interface.

An agent asks a :class:`Model` for one completion at a time and gets back a
:class:`Completion`: the text the model wrote, the tools it wants called, and the raw
content to append to the conversation. The agent runs the tools and asks again until the
model stops calling tools. Keeping the model behind this interface is what lets the tests
drive an agent with a scripted model and no API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-opus-5"


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Completion:
    """One reply from the model."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: The assistant content to append to the conversation before tool results. For
    #: Claude this is the response's content blocks, thinking blocks included, so the
    #: next request carries them back unchanged.
    content: Any = None
    stop_reason: str = "end_turn"


class Model(Protocol):
    def complete(self, *, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]]) -> Completion: ...


class ClaudeModel:
    """Claude, through ``anthropic.Anthropic()``.

    The client resolves its credential from the environment: ``ANTHROPIC_API_KEY``, or
    ``ANTHROPIC_AUTH_TOKEN``, or a profile written by ``ant auth login``. Nothing is
    hard-coded here.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, max_tokens: int = 8000) -> None:
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def complete(self, *, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]]) -> Completion:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            why = getattr(details, "explanation", None) or "no explanation given"
            return Completion(text=f"The model declined to answer ({why}).",
                              content=response.content, stop_reason="refusal")
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input))
                 for b in response.content if b.type == "tool_use"]
        return Completion(text=text, tool_calls=calls, content=response.content,
                          stop_reason=response.stop_reason)


class ScriptedModel:
    """A model that replays completions handed to it in order.

    For tests and for checking the memory integration without spending tokens. Each
    entry is either a string (a final text reply) or a list of ``(name, input)`` tool
    calls; the calls are given ids so an agent can match results to them.
    """

    def __init__(self, script: list[str | list[tuple[str, dict[str, Any]]]]) -> None:
        self._script = list(script)
        self.requests: list[dict[str, Any]] = []

    def complete(self, *, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]]) -> Completion:
        self.requests.append({"system": system, "messages": list(messages),
                              "tools": tools})
        if not self._script:
            raise AssertionError("ScriptedModel ran out of scripted replies")
        step = self._script.pop(0)
        if isinstance(step, str):
            return Completion(text=step, content=[{"type": "text", "text": step}])
        calls = [ToolCall(f"call_{i}", name, dict(args))
                 for i, (name, args) in enumerate(step)]
        content = [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input}
                   for c in calls]
        return Completion(text="", tool_calls=calls, content=content,
                          stop_reason="tool_use")
