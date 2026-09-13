"""The model layer: any Anthropic-format or OpenAI-format endpoint, behind one interface.

An agent asks a :class:`Model` for one completion at a time and gets back a
:class:`Completion`: the text the model wrote, the tools it wants called, and the
provider's own representation of the reply, which goes back into the conversation
unchanged on the next request. The agent runs the tools and asks again until the model
stops calling tools.

The agent keeps the conversation in a provider-neutral form (see :func:`to_anthropic`
and :func:`to_openai`), so the same agent runs against Claude through the Anthropic SDK,
against OpenAI, or against anything that speaks the OpenAI chat-completions format:
Ollama, vLLM, LM Studio, OpenRouter, a company gateway. Which one is decided by
:func:`resolve_config`, from flags first and then the environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

DEFAULT_CLAUDE_MODEL = "claude-opus-5"
PROVIDERS = ("anthropic", "openai")


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
    #: The provider's own representation of this reply, appended to the conversation
    #: unchanged on the next request. For Claude that is the list of content blocks,
    #: thinking blocks included; for an OpenAI-format endpoint it is the assistant
    #: message with its tool_calls.
    content: Any = None
    stop_reason: str = "end_turn"


class Model(Protocol):
    def complete(self, *, system: str, transcript: list[dict[str, Any]],
                 tools: list[dict[str, Any]]) -> Completion: ...


# -- the neutral transcript and its two renderings -------------------------------------
#
# Each entry is one of:
#   {"role": "user", "content": "<text>"}
#   {"role": "assistant", "completion": Completion}
#   {"role": "tool", "results": [{"id", "name", "content", "is_error"}, ...]}


def to_anthropic(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The transcript as Anthropic Messages API messages."""
    out: list[dict[str, Any]] = []
    for entry in transcript:
        if entry["role"] == "user":
            out.append({"role": "user", "content": entry["content"]})
        elif entry["role"] == "assistant":
            out.append({"role": "assistant", "content": entry["completion"].content})
        else:
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"],
                 "is_error": r["is_error"]} for r in entry["results"]]})
    return out


def to_openai(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The transcript as OpenAI chat-completions messages."""
    out: list[dict[str, Any]] = []
    for entry in transcript:
        if entry["role"] == "user":
            out.append({"role": "user", "content": entry["content"]})
        elif entry["role"] == "assistant":
            out.append(entry["completion"].content)
        else:
            for r in entry["results"]:
                out.append({"role": "tool", "tool_call_id": r["id"], "content": r["content"]})
    return out


def openai_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """A tool declared in Anthropic form, as an OpenAI function tool."""
    return {"type": "function", "function": {
        "name": schema["name"], "description": schema["description"],
        "parameters": schema["input_schema"]}}


# -- the two providers ---------------------------------------------------------------


class ClaudeModel:
    """Claude, or any Anthropic-format endpoint, through ``anthropic.Anthropic()``.

    ``base_url`` and ``api_key`` left as None fall through to the SDK's own resolution:
    ``ANTHROPIC_BASE_URL``, then ``ANTHROPIC_API_KEY`` or ``ANTHROPIC_AUTH_TOKEN`` or a
    profile written by ``ant auth login``. Nothing is hard-coded here.
    """

    provider = "anthropic"

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, *, base_url: str | None = None,
                 api_key: str | None = None, max_tokens: int = 8000) -> None:
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def credential_present(self) -> bool:
        """Whether the SDK resolved a credential, asked of the SDK itself."""
        return bool(self._client.api_key or self._client.auth_token)

    def complete(self, *, system: str, transcript: list[dict[str, Any]],
                 tools: list[dict[str, Any]]) -> Completion:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=to_anthropic(transcript),
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


class OpenAIModel:
    """OpenAI, or any endpoint that speaks its chat-completions format, through
    ``openai.OpenAI()``.

    ``base_url`` and ``api_key`` left as None fall through to ``OPENAI_BASE_URL`` and
    ``OPENAI_API_KEY``. A local server such as Ollama or vLLM usually ignores the key but
    the SDK insists on one, so when a base URL is given and no key is, a placeholder is
    sent.
    """

    provider = "openai"

    def __init__(self, model: str, *, base_url: str | None = None,
                 api_key: str | None = None, max_tokens: int = 8000,
                 client: Any = None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
            return
        import openai

        base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        api_key = api_key or os.environ.get("OPENAI_API_KEY") or None
        if api_key is None and base_url:
            api_key = "not-needed"
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def credential_present(self) -> bool:
        return bool(self._client.api_key)

    def complete(self, *, system: str, transcript: list[dict[str, Any]],
                 tools: list[dict[str, Any]]) -> Completion:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = [openai_tool(t) for t in tools]
        response = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, *to_openai(transcript)],
            **kwargs,
        )
        choice = response.choices[0]
        message = choice.message
        calls: list[ToolCall] = []
        raw_calls: list[dict[str, Any]] = []
        for tc in message.tool_calls or []:
            arguments = tc.function.arguments or "{}"
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {"_unparseable_arguments": arguments}
            calls.append(ToolCall(tc.id, tc.function.name, parsed))
            raw_calls.append({"id": tc.id, "type": "function",
                              "function": {"name": tc.function.name,
                                           "arguments": arguments}})
        content: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if raw_calls:
            content["tool_calls"] = raw_calls
        return Completion(text=message.content or "", tool_calls=calls, content=content,
                          stop_reason=choice.finish_reason or "stop")


class ScriptedModel:
    """A model that replays completions handed to it in order.

    For tests and for checking the memory integration without spending tokens. Each
    entry is either a string (a final text reply) or a list of ``(name, input)`` tool
    calls; the calls are given ids so an agent can match results to them. Every request
    is recorded, rendered in Anthropic form, so a test can read what the model was
    handed.
    """

    provider = "scripted"

    def __init__(self, script: list[str | list[tuple[str, dict[str, Any]]]]) -> None:
        self._script = list(script)
        self.requests: list[dict[str, Any]] = []

    def credential_present(self) -> bool:
        return True

    def complete(self, *, system: str, transcript: list[dict[str, Any]],
                 tools: list[dict[str, Any]]) -> Completion:
        self.requests.append({"system": system, "messages": to_anthropic(transcript),
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


# -- choosing a provider ---------------------------------------------------------------


@dataclass
class ModelConfig:
    provider: str
    model: str | None
    base_url: str | None
    api_key: str | None

    def describe(self) -> str:
        where = self.base_url or "the provider's default endpoint"
        return f"provider {self.provider}, model {self.model or '(none)'}, endpoint {where}"


def resolve_config(*, provider: str | None = None, model: str | None = None,
                   base_url: str | None = None, api_key: str | None = None,
                   env: Mapping[str, str] | None = None) -> ModelConfig:
    """Which endpoint to talk to, from flags first and then the environment.

    Order for each field: the flag, then ``LLM_PROVIDER`` / ``LLM_MODEL`` /
    ``LLM_BASE_URL`` / ``LLM_API_KEY``, then the provider SDK's own variables
    (``ANTHROPIC_*`` or ``OPENAI_*``), which the SDK reads itself.

    With no provider named anywhere, ``openai`` is chosen when an OpenAI credential or
    base URL is set and no Anthropic one is; otherwise ``anthropic``. The Anthropic
    provider has a default model; an OpenAI-format endpoint has none, because every
    server names its models differently, so ``model`` is required there and
    :func:`build_model` says so.
    """
    environ = os.environ if env is None else env
    provider = (provider or environ.get("LLM_PROVIDER") or "").strip().lower()
    if not provider:
        has_openai = bool(environ.get("OPENAI_API_KEY") or environ.get("OPENAI_BASE_URL"))
        has_anthropic = bool(environ.get("ANTHROPIC_API_KEY")
                             or environ.get("ANTHROPIC_AUTH_TOKEN"))
        provider = "openai" if has_openai and not has_anthropic else "anthropic"
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; use one of {', '.join(PROVIDERS)}.")
    model = model or environ.get("LLM_MODEL") or None
    if model is None and provider == "anthropic":
        model = DEFAULT_CLAUDE_MODEL
    return ModelConfig(provider=provider, model=model,
                       base_url=base_url or environ.get("LLM_BASE_URL") or None,
                       api_key=api_key or environ.get("LLM_API_KEY") or None)


def build_model(config: ModelConfig) -> Model:
    """A model client for ``config``. Makes no network call."""
    if config.model is None:
        raise ValueError(
            f"No model named for provider {config.provider}. Pass --model or set "
            "LLM_MODEL (for example qwen3:8b on Ollama, or gpt-4.1 on OpenAI).")
    if config.provider == "anthropic":
        return ClaudeModel(config.model, base_url=config.base_url, api_key=config.api_key)
    return OpenAIModel(config.model, base_url=config.base_url, api_key=config.api_key)
