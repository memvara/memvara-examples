"""The model layer without a network: transcript rendering for both wire formats, the
OpenAI-format client against a stub, and how the endpoint is chosen."""

import json
from types import SimpleNamespace

import pytest

from memvara_examples.model import (Completion, ModelConfig, OpenAIModel, ToolCall,
                                    build_model, openai_tool, resolve_config,
                                    to_anthropic, to_openai)

TRANSCRIPT = [
    {"role": "user", "content": "Where do I live?"},
    {"role": "assistant", "completion": Completion(
        text="", tool_calls=[ToolCall("c1", "memory_recall", {"query": "home"})],
        content="<provider-specific>", stop_reason="tool_use")},
    {"role": "tool", "results": [{"id": "c1", "name": "memory_recall",
                                  "content": "- user lives_in Lisbon", "is_error": False}]},
]


def test_anthropic_rendering_puts_tool_results_in_a_user_message():
    msgs = to_anthropic(TRANSCRIPT)
    assert msgs[0] == {"role": "user", "content": "Where do I live?"}
    assert msgs[1] == {"role": "assistant", "content": "<provider-specific>"}
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["tool_use_id"] == "c1"


def test_openai_rendering_puts_each_tool_result_in_its_own_message():
    msgs = to_openai(TRANSCRIPT)
    assert msgs[1] == "<provider-specific>"
    assert msgs[2] == {"role": "tool", "tool_call_id": "c1", "content": "- user lives_in Lisbon"}


def test_openai_tool_wraps_the_anthropic_schema():
    schema = {"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}
    assert openai_tool(schema) == {"type": "function", "function": {
        "name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}}


class StubClient:
    """Enough of the OpenAI client to answer one chat completion and record the request."""

    def __init__(self, message, finish_reason="stop"):
        self.calls = []
        self.api_key = "k"
        client = self

        class Completions:
            def create(self, **kwargs):
                client.calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=message,
                                                                finish_reason=finish_reason)])

        self.chat = SimpleNamespace(completions=Completions())


def test_openai_model_turns_tool_calls_into_completions_and_echoes_them_back():
    message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="memory_recall",
                                              arguments=json.dumps({"query": "home"})))])
    client = StubClient(message, finish_reason="tool_calls")
    model = OpenAIModel("local-model", client=client)
    tools = [{"name": "memory_recall", "description": "d",
              "input_schema": {"type": "object", "properties": {}}}]
    reply = model.complete(system="sys", transcript=[{"role": "user", "content": "hi"}],
                           tools=tools)
    assert reply.tool_calls == [ToolCall("call_1", "memory_recall", {"query": "home"})]
    assert reply.content["tool_calls"][0]["function"]["name"] == "memory_recall"
    sent = client.calls[0]
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["tools"][0]["type"] == "function"
    # The next request carries the assistant message back in OpenAI form.
    transcript = [{"role": "user", "content": "hi"},
                  {"role": "assistant", "completion": reply},
                  {"role": "tool", "results": [{"id": "call_1", "name": "memory_recall",
                                                "content": "Lisbon", "is_error": False}]}]
    assert to_openai(transcript)[1]["tool_calls"][0]["id"] == "call_1"
    assert to_openai(transcript)[2] == {"role": "tool", "tool_call_id": "call_1", "content": "Lisbon"}


def test_openai_model_survives_unparseable_arguments():
    message = SimpleNamespace(content="", tool_calls=[SimpleNamespace(
        id="c", function=SimpleNamespace(name="memory_recall", arguments="{not json"))])
    reply = OpenAIModel("m", client=StubClient(message)).complete(
        system="s", transcript=[{"role": "user", "content": "x"}], tools=[])
    assert reply.tool_calls[0].input == {"_unparseable_arguments": "{not json"}


def test_resolve_config_prefers_flags_then_llm_variables_then_the_sdk():
    env = {"LLM_PROVIDER": "openai", "LLM_MODEL": "m-env", "LLM_BASE_URL": "http://env",
           "LLM_API_KEY": "k-env"}
    c = resolve_config(model="m-flag", env=env)
    assert (c.provider, c.model, c.base_url, c.api_key) == ("openai", "m-flag", "http://env", "k-env")
    c = resolve_config(env={})
    assert (c.provider, c.model, c.base_url, c.api_key) == ("anthropic", "claude-opus-5", None, None)


def test_resolve_config_picks_openai_when_only_openai_variables_are_set():
    assert resolve_config(env={"OPENAI_API_KEY": "x"}).provider == "openai"
    assert resolve_config(env={"OPENAI_BASE_URL": "http://localhost:11434/v1"}).provider == "openai"
    assert resolve_config(env={"OPENAI_API_KEY": "x", "ANTHROPIC_API_KEY": "y"}).provider == "anthropic"
    with pytest.raises(ValueError):
        resolve_config(provider="gemini", env={})


def test_build_model_needs_a_model_name_for_openai():
    with pytest.raises(ValueError, match="LLM_MODEL"):
        build_model(ModelConfig("openai", None, None, None))
    model = build_model(ModelConfig("openai", "llama", "http://localhost:11434/v1", None))
    assert isinstance(model, OpenAIModel) and model.credential_present()
    claude = build_model(ModelConfig("anthropic", "claude-opus-5", "http://gateway", "k"))
    assert claude.provider == "anthropic" and claude.credential_present()
