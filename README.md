# memvara-examples

Two working AI agents that keep their long-term memory in [Memvara](https://memvara.dev),
and a command that starts either one.

| Agent | What it does |
|---|---|
| `assistant` | A personal assistant. It reads what it knows about you before every reply, stores what it learns as structured facts, and corrects a wrong memory the right way: ended if the world moved on, forgotten if the record was never right. |
| `engineer` | Keeps the record of a software project's decisions. It answers what is true now, what was true before, when that changed, why, and what the record would have said on a past date. |

Both agents talk to Claude through the Anthropic SDK and to Memvara through the `memvara`
Python library. The memory is real and shared: a fact one session stores, a session next
week reads.

## Setup

You need Python 3.10 or newer, an Anthropic API key, and a Memvara credential.

```bash
git clone https://github.com/memvara/memvara-examples && cd memvara-examples
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

For the Memvara credential, either of these works:

- **Hosted (default).** Run `memvara-mcp login` once (it comes with `pip install memvara`)
  and approve in the browser. The key lands in `~/.memvara/credentials.json`, and
  `Memvara.connect()` finds it. `MEMVARA_API_KEY` in the environment works too.
- **Local, no account.** Pass `--local ./memory.db` to any command. The store is a SQLite
  file on your machine and nothing leaves it.

Check both before starting an agent:

```bash
memvara-examples check
```

## Start an agent

```bash
memvara-examples                       # menu: pick 1 or 2
memvara-examples assistant             # start the assistant directly
memvara-examples engineer --project checkout
memvara-examples assistant --local ./memory.db
```

Every command takes `--user NAME` (whose memory to open; default your login name),
`--model ID` (default `claude-opus-5`), `--local PATH` and `--quiet` (hide the tool calls
the agent makes). Inside a chat, `/standing` shows the preferences memory holds for you and
`/quit` leaves.

## What a session looks like

The assistant, with the tool calls it makes shown in brackets:

```
you> Keep replies under 50 words. I moved to London on 15 March.
    [memory_remember(predicate='prefers', object='replies under 50 words', memory_type='procedural') -> {"added": [...]}]
    [memory_remember(predicate='lives_in', object='London', true_since='2026-03-15') -> {"added": [...], "ended": ["user lives_in Berlin ..."]}]

assistant> Noted. Short replies from now on, and you live in London since 15 March. Berlin is kept as history.

you> Why do you think I'm allergic to peanuts?
    [memory_search(query='allergic to peanuts') -> [{"id": "cl_...", ...}]]
    [memory_why(claim_id='cl_...') -> {"sources": [{"text": "my brother is allergic to peanuts"}]}]

assistant> It came from a message that said "my brother is allergic to peanuts". That looks like it was about your brother, not you. Should I retire it?

you> Yes.
    [memory_forget(claim_id='cl_...') -> Retired. It was recorded as never having been right.]
```

The engineer, two weeks after a decision:

```
you> Decision 12 June: migrate checkout auth from API keys to OAuth 2.0 client credentials. Manual key rotation does not work for the Q3 integrators.
    [record_decision(component='checkout-service', predicate='auth_strategy', value='OAuth 2.0 client credentials', decision='...', rationale='...', decided_on='2026-06-12') -> {"fact_now": ..., "previous_values_ended": ["API keys (held from 2026-02-03)"]}]

engineer> Recorded. checkout-service auth_strategy is now OAuth 2.0 client credentials; API keys ended on 12 June.

you> What was it before, and when did it change?
    [memory_history(subject='checkout-service', predicate='auth_strategy') -> [...]]

engineer> API keys, from 3 February until 12 June 2026, when it became OAuth 2.0 client credentials.

you> What would we have said on 1 April?
    [project_state_at(question='how does checkout-service authenticate?', at='2026-04-01') -> {...}]

engineer> API keys. The record also shows this was written down on 13 September, so on 1 April the store itself would have said nothing yet.
```

## How it is built

```
memvara_examples/
  memory.py          Memory: one user's store, hosted or local, behind the calls an agent makes
  model.py           ClaudeModel (Anthropic SDK) and ScriptedModel (a replay double for tests)
  tools.py           the memory tools the model can call, and the dispatcher that runs them
  agents/base.py     the loop: standing preferences + recall into the prompt, tool calls until an answer
  agents/assistant.py
  agents/engineer.py adds record_decision and project_state_at
  cli.py             the memvara-examples command
```

Three rules from Memvara's own documentation are applied in `memory.py`, once, so neither
agent can get them wrong:

1. **Facts are written as triples**, never as prose. A hosted deployment may have no
   extraction model, and a paragraph it does not recognise is accepted and stored as
   nothing. `memory_remember` takes a subject, a predicate and an object.
2. **A new value ends the old one, it does not overwrite it.** `set_fact` closes every
   current value in the slot on the world clock and then writes the new one, so
   `memory_history` shows both with the interval each held. This is done explicitly rather
   than relying on the predicate's declared cardinality, because a project vocabulary the
   store has never seen (`auth_strategy`, `deploy_target`) is multi-valued by default.
3. **Ending and forgetting mean different things.** `memory_end` says the fact was true and
   has stopped being true. `memory_forget` says the record was never right. Neither deletes
   anything; both stay visible in the history with the reason recorded.

Every write cites the user's message it came from, so `memory_why` can show the sentence a
fact was read from. The cited turn is stored with the `system` role so that the store's own
rule-based extractor does not write a second reading of the same sentence beside the
agent's; the agent is the only writer.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests run both agents with a scripted model against a local store, so they need no API
key and no network. They check what lands in memory and what the model is handed back, not
whether a call returned. The Claude path itself is one method, `ClaudeModel.complete`.

## License

Apache-2.0. See [LICENSE](LICENSE).
