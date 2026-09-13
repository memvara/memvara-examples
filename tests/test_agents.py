"""Both agents driven by a scripted model against a local store.

The model is scripted, so these do not test Claude. They test everything else: that
recall reaches the prompt, that a tool call the model makes lands in memory as the
right kind of write, and that the answer the model is handed back is read out of the
store rather than out of the conversation.
"""

import json

from memvara_examples.agents import AssistantAgent, EngineerAgent
from memvara_examples.model import ScriptedModel


def calls(model, index):
    """The tool results the model was handed on request `index`."""
    msgs = model.requests[index]["messages"]
    return [r["content"] for r in msgs[-1]["content"]]


# -- assistant ---------------------------------------------------------------------

def test_assistant_stores_a_preference_and_a_fact(memory):
    model = ScriptedModel([
        [("memory_remember", {"predicate": "prefers", "object": "replies under 50 words",
                              "memory_type": "procedural"}),
         ("memory_remember", {"predicate": "lives_in", "object": "Lisbon"})],
        "Noted: you prefer short replies and you live in Lisbon.",
    ])
    agent = AssistantAgent(memory, model)
    reply = agent.turn("Keep replies under 50 words. I live in Lisbon.")
    assert reply.startswith("Noted")
    added = json.loads(calls(model, 1)[0])["added"]
    assert added[0].startswith("user prefers replies under 50 words")
    assert [c.object for c in memory.standing()] == ["replies under 50 words"]
    prov = memory.why(memory.search("Lisbon")[0]["id"])
    assert prov.episodes[0].content == "Keep replies under 50 words. I live in Lisbon."


def test_assistant_prompt_carries_standing_preferences_and_recall(memory):
    memory.remember("user", "prefers", "metric units", memory_type="procedural")
    memory.remember("user", "lives_in", "Lisbon")
    model = ScriptedModel(["It is about 15 km."])
    AssistantAgent(memory, model).turn("How far is the airport from my flat?")
    system = model.requests[0]["system"]
    assert "prefers: metric units" in system
    assert "Lisbon" in system


def test_assistant_correction_ends_the_old_value_and_can_explain(memory):
    memory.remember("user", "lives_in", "Berlin", true_since="2026-01-10",
                    source_text="I live in Berlin")
    model = ScriptedModel([
        [("memory_remember", {"predicate": "lives_in", "object": "London",
                              "true_since": "2026-03-15"})],
        "Updated: London since 15 March. Berlin is kept as history.",
        [("memory_search", {"query": "where do they live"})],
        [("memory_history", {"predicate": "lives_in"})],
        "Before London you lived in Berlin, from 10 January to 15 March.",
    ])
    agent = AssistantAgent(memory, model)
    agent.turn("I moved to London on 15 March.")
    receipt = json.loads(calls(model, 1)[0])
    assert receipt["ended"][0].startswith("user lives_in Berlin")
    reply = agent.turn("Where did I live before?")
    history = json.loads(calls(model, 4)[0])
    assert [(r["object"], r["state"]) for r in history] == [("Berlin", "ended"), ("London", "live")]
    assert "Berlin" in reply


def test_assistant_forget_after_why(memory):
    receipt = memory.remember("user", "allergic_to", "peanuts",
                              source_text="my brother is allergic to peanuts")
    claim_id = receipt.added[0].id
    model = ScriptedModel([
        [("memory_search", {"query": "allergic"})],
        [("memory_why", {"claim_id": claim_id})],
        "That came from 'my brother is allergic to peanuts'. Shall I retire it?",
        [("memory_forget", {"claim_id": claim_id})],
        "Retired. It was recorded as never having been about you.",
    ])
    agent = AssistantAgent(memory, model)
    agent.turn("Why do you think I'm allergic to peanuts?")
    why = json.loads(calls(model, 2)[0])
    assert why["sources"][0]["text"] == "my brother is allergic to peanuts"
    agent.turn("Yes, that was about my brother.")
    assert memory.history("user", "allergic_to")[0]["state"] == "retired"
    assert "peanut" not in memory.recall("allergies")


def test_tool_errors_go_back_to_the_model_as_errors(memory):
    model = ScriptedModel([
        [("memory_end", {"claim_id": "cl_nope"}), ("no_such_tool", {})],
        "Nothing to end.",
    ])
    AssistantAgent(memory, model).turn("end it")
    results = model.requests[1]["messages"][-1]["content"]
    assert results[0]["is_error"] is False and "nothing changed" in results[0]["content"]
    assert results[1]["is_error"] is True and "Unknown tool" in results[1]["content"]


# -- engineer ----------------------------------------------------------------------

def test_engineer_records_a_decision_and_answers_the_four_questions(memory):
    model = ScriptedModel([
        [("record_decision", {"component": "checkout-service", "predicate": "auth_strategy",
                              "value": "API keys", "decision": "authenticate with per-consumer API keys",
                              "decided_on": "2026-02-03"})],
        "Recorded.",
        [("record_decision", {"component": "checkout-service", "predicate": "auth_strategy",
                              "value": "OAuth 2.0 client credentials",
                              "decision": "migrate service-to-service auth from API keys to OAuth 2.0",
                              "rationale": "manual key rotation does not work for the Q3 integrators",
                              "decided_on": "2026-06-12"})],
        "Recorded: auth_strategy is now OAuth 2.0 client credentials; API keys ended on 12 June.",
        [("memory_recall", {"query": "checkout-service auth strategy"})],
        "OAuth 2.0 client credentials.",
        [("memory_history", {"subject": "checkout-service", "predicate": "auth_strategy"})],
        "API keys, from 3 February until 12 June, when it became OAuth 2.0.",
    ])
    agent = EngineerAgent(memory, model, project="shop")
    agent.turn("Checkout uses per-consumer API keys, decided 3 Feb 2026.")
    agent.turn("Decision 12 June 2026: migrate checkout auth to OAuth 2.0 client credentials. "
               "Manual key rotation does not work for the Q3 integrators.")
    second = json.loads(calls(model, 3)[0])
    assert second["fact_now"] == "checkout-service auth_strategy OAuth 2.0 client credentials"
    assert second["previous_values_ended"] == ["API keys (held from 2026-02-03)"]

    assert "OAuth" in agent.turn("What is the auth strategy now?")
    assert "OAuth" in calls(model, 5)[0]

    agent.turn("What was it before and when did it change?")
    rows = json.loads(calls(model, 7)[0])
    assert [(r["object"], r["state"]) for r in rows] == \
        [("API keys", "ended"), ("OAuth 2.0 client credentials", "live")]
    assert rows[0]["true_until"].startswith("2026-06-12")

    # Why: the decision cites the message it came from.
    decided = memory.history("checkout-service", "decided")
    assert len(decided) == 2
    prov = memory.why(decided[1]["id"])
    assert prov.episodes[0].content.startswith("Decision 12 June 2026")
    assert "Because: manual key rotation" in decided[1]["object"]


def test_engineer_state_at_answers_a_past_date(memory):
    memory.set_fact("checkout-service", "auth_strategy", "API keys", true_since="2026-02-03")
    memory.set_fact("checkout-service", "auth_strategy", "OAuth 2.0", true_since="2026-06-12")
    model = ScriptedModel([
        [("project_state_at", {"question": "how does checkout-service authenticate?",
                               "at": "2026-04-01"})],
        "On 1 April it was API keys.",
    ])
    EngineerAgent(memory, model, project="shop").turn("What would we have said on 1 April?")
    result = json.loads(calls(model, 1)[0])
    assert "API keys" in result["narrative"]
    # The facts were backfilled today, so on 1 April this store held nothing yet. The
    # tool reports that difference rather than hiding it.
    corrected = result["corrected_after_the_fact"]
    assert corrected[0]["true_then_by_todays_record"] == ["API keys"]
    assert corrected[0]["what_we_would_have_said_then"] == []


def test_engineer_prompt_names_the_project(memory):
    model = ScriptedModel(["Hello."])
    EngineerAgent(memory, model, project="atlas").turn("hi")
    assert '"atlas"' in model.requests[0]["system"]
    names = [t["name"] for t in model.requests[0]["tools"]]
    assert "record_decision" in names and "project_state_at" in names and "memory_why" in names


def test_a_failed_model_call_leaves_no_half_turn_behind(memory):
    class Broken:
        def complete(self, **kwargs):
            raise RuntimeError("rate limited")

    agent = AssistantAgent(memory, Broken())
    try:
        agent.turn("hello")
    except RuntimeError:
        pass
    assert agent.transcript == []


def test_confidence_is_clamped_to_the_unit_interval(memory):
    model = ScriptedModel([
        [("memory_remember", {"predicate": "likes", "object": "jazz", "confidence": 7})],
        "ok",
    ])
    AssistantAgent(memory, model).turn("I might like jazz")
    assert memory.search("jazz")[0]["confidence"] == 1.0
