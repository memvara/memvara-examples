"""The memory layer against a local store: every call an agent makes, checked on the
values it reads back, not on whether it returned."""

from datetime import datetime, timezone

from memvara_examples.memory import parse_when


def test_remember_then_recall(memory):
    memory.remember("user", "lives_in", "Lisbon", source_text="I live in Lisbon now")
    notes = memory.recall("where does the user live?")
    assert "Lisbon" in notes


def test_remember_into_a_single_valued_slot_ends_the_old_value(memory):
    memory.remember("user", "lives_in", "Berlin", true_since="2026-01-10")
    receipt = memory.remember("user", "lives_in", "London", true_since="2026-03-15")
    assert [c.object for c in receipt.closed] == ["Berlin"]
    rows = memory.history("user", "lives_in")
    assert [(r["object"], r["state"]) for r in rows] == [("Berlin", "ended"), ("London", "live")]
    assert rows[0]["true_until"].startswith("2026-03-15")


def test_set_fact_replaces_an_undeclared_predicate(memory):
    # `auth_strategy` is not in the built-in vocabulary, so a plain remember would
    # accumulate a second value beside the first. set_fact ends the old one explicitly.
    memory.set_fact("checkout", "auth_strategy", "API keys", true_since="2026-02-03")
    ended, receipt = memory.set_fact("checkout", "auth_strategy", "OAuth 2.0",
                                     true_since="2026-06-12",
                                     source_text="Decision: move checkout to OAuth 2.0")
    assert [c.object for c in ended] == ["API keys"]
    assert [c.object for c in receipt.added] == ["OAuth 2.0"]
    rows = memory.history("checkout", "auth_strategy")
    assert [(r["object"], r["state"]) for r in rows] == [("API keys", "ended"), ("OAuth 2.0", "live")]
    assert "OAuth" in memory.recall("how does checkout authenticate?")


def test_why_returns_the_source_sentence(memory):
    receipt = memory.remember("user", "works_at", "Acme",
                              source_text="I just started at Acme as a data engineer")
    prov = memory.why(receipt.added[0].id)
    assert prov is not None
    assert [e.content for e in prov.episodes] == ["I just started at Acme as a data engineer"]
    assert prov.episodes[0].role == "system"  # filed as a cited transcript turn


def test_end_and_forget_record_different_closures(memory):
    a = memory.remember("user", "subscribed_to", "Gym A").added[0]
    b = memory.remember("user", "allergic_to", "peanuts").added[0]
    assert memory.end(a.id, at="2026-05-01")
    assert memory.forget(b.id)
    states = {r["object"]: r["state"] for r in
              memory.history("user", "subscribed_to") + memory.history("user", "allergic_to")}
    assert states == {"Gym A": "ended", "peanuts": "retired"}
    assert not memory.end("cl_does_not_exist")


def test_standing_returns_only_procedural_facts_about_the_user(memory):
    memory.remember("user", "prefers", "short replies", memory_type="procedural")
    memory.remember("user", "lives_in", "Lisbon")
    memory.remember("memvara", "uses_tool", "pytest", memory_type="procedural")
    assert [(c.predicate, c.object) for c in memory.standing()] == [("prefers", "short replies")]


def test_search_returns_ids_and_a_past_view(memory):
    memory.remember("user", "lives_in", "Berlin", true_since="2026-01-10")
    memory.remember("user", "lives_in", "London", true_since="2026-03-15")
    now = memory.search("where do they live", k=3)
    assert now and now[0]["object"] == "London" and now[0]["id"].startswith("cl")
    then = memory.search("where do they live", k=3,
                         valid_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    assert [r["object"] for r in then] == ["Berlin"]


def test_ask_reports_what_was_true_then(memory):
    memory.remember("user", "lives_in", "Berlin", true_since="2026-01-10")
    memory.remember("user", "lives_in", "London", true_since="2026-03-15")
    answer = memory.ask("where do they live?", at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    assert answer.readings
    assert [c.object for c in answer.readings[0].then] == ["Berlin"]
    assert [c.object for c in answer.readings[0].now] == ["London"]


def test_describe_counts_facts(memory):
    assert memory.describe()["claims_visible"] == 0
    memory.remember("user", "lives_in", "Lisbon")
    assert memory.describe()["claims_visible"] == 1


def test_parse_when_accepts_dates_and_instants():
    assert parse_when("2026-03-01") == datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert parse_when("2026-03-01T09:00:00Z") == datetime(2026, 3, 1, 9, tzinfo=timezone.utc)
    assert parse_when("2026-03-01T09:00:00z") == datetime(2026, 3, 1, 9, tzinfo=timezone.utc)
    assert parse_when(None) is None
