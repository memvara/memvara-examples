"""The memory tools the model can call, and the code that runs them.

Each tool is a JSON schema the model sees plus a Python handler. The handlers return
plain text, which is what goes back to the model as the tool result. Every handler is
bound to one :class:`~memvara_examples.memory.Memory`, so a tool call can only ever read
or write the current user's memory.

The descriptions matter more than the code. They are what the model reads to decide
which tool to call, and they carry the one distinction that is easy to get wrong:
``memory_end`` is for a fact that was true and has stopped being true, and
``memory_forget`` is for a record that was never right.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .memory import Memory, parse_when

Handler = Callable[[dict[str, Any]], str]


def _schema(name: str, description: str, properties: dict[str, Any],
            required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def _json(value: Any) -> str:
    return json.dumps(value, indent=1, default=str)


class Toolbox:
    """The tools one agent exposes, and the dispatcher that runs them.

    ``source_text`` is the user's current message. A write made while handling that
    message cites it, so a later ``memory_why`` can show the sentence the fact came from.
    The agent sets it at the start of every turn.
    """

    def __init__(self, memory: Memory) -> None:
        self.memory = memory
        self.source_text: str | None = None
        self.schemas: list[dict[str, Any]] = []
        self._handlers: dict[str, Handler] = {}

    def register(self, schema: dict[str, Any], handler: Handler) -> None:
        self.schemas.append(schema)
        self._handlers[schema["name"]] = handler

    def run(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Run one tool. Returns the result text and whether it is an error."""
        handler = self._handlers.get(name)
        if handler is None:
            return f"Unknown tool: {name}", True
        if "_unparseable_arguments" in arguments:
            return ("The arguments for this call were not valid JSON, so it was not run: "
                    f"{arguments['_unparseable_arguments'][:200]}"), True
        try:
            return handler(arguments), False
        except Exception as exc:  # the model gets the error text and can recover
            return f"{type(exc).__name__}: {exc}", True


def memory_tools(memory: Memory) -> Toolbox:
    """The tools both agents share: read, write, correct, and explain memory."""
    box = Toolbox(memory)

    def recall(args: dict[str, Any]) -> str:
        return memory.recall(args["query"], k=int(args.get("k", 8))) or "Nothing stored matches."

    box.register(_schema(
        "memory_recall",
        "Look up what is already known that bears on a question. Call it before "
        "answering anything that could depend on something the user said earlier. "
        "Returns plain notes; an empty result means nothing is stored, not that you "
        "should retry.",
        {"query": {"type": "string", "description": "The question or topic, in natural language."},
         "k": {"type": "integer", "description": "Most notes to return. Default 8."}},
        ["query"]), recall)

    def search(args: dict[str, Any]) -> str:
        rows = memory.search(args["query"], k=int(args.get("k", 10)),
                             valid_at=parse_when(args.get("valid_at")))
        return _json(rows) if rows else "No matching facts."

    box.register(_schema(
        "memory_search",
        "Find stored facts with their ids. Use it when you need an id to pass to "
        "memory_why, memory_end or memory_forget, or to see what was true at a past "
        "date with valid_at. To simply answer the user, use memory_recall instead.",
        {"query": {"type": "string"},
         "k": {"type": "integer", "description": "Most rows to return. Default 10."},
         "valid_at": {"type": "string", "description": "ISO-8601 date or instant. What was true in the world then."}},
        ["query"]), search)

    def remember(args: dict[str, Any]) -> str:
        receipt = memory.remember(
            args.get("subject", "user"), args["predicate"], args["object"],
            memory_type=args.get("memory_type"), true_since=args.get("true_since"),
            confidence=min(1.0, max(0.0, float(args.get("confidence", 1.0)))),
            source_text=box.source_text)
        out = {"added": [f"{c.subject} {c.predicate} {c.object} (id {c.id})" for c in receipt.added],
               "ended": [f"{c.subject} {c.predicate} {c.object} (id {c.id})" for c in receipt.closed],
               "reinforced": len(receipt.reinforced)}
        return _json(out)

    box.register(_schema(
        "memory_remember",
        "Store one exact fact as a subject, predicate, object triple. Use it when the "
        "user states something worth knowing next week: a preference, a constraint, a "
        "decision, where they live or work. Reuse a predicate you have seen in "
        "memory_search results rather than inventing a synonym. If the fact started "
        "being true before now, pass true_since. Use memory_type 'procedural' only for "
        "how the user wants work done, 'episodic' for something that happened at a "
        "time, and leave it unset for a durable fact.",
        {"subject": {"type": "string", "description": "Who or what the fact is about. Default 'user'."},
         "predicate": {"type": "string", "description": "snake_case relation: lives_in, works_at, prefers, allergic_to, uses_tool."},
         "object": {"type": "string", "description": "The value, as short as it can be."},
         "memory_type": {"type": "string", "enum": ["semantic", "episodic", "procedural"]},
         "true_since": {"type": "string", "description": "ISO-8601 date the fact became true, if before now."},
         "confidence": {"type": "number", "minimum": 0, "maximum": 1,
                        "description": "0 to 1. Lower it for something you inferred rather than were told."}},
        ["predicate", "object"]), remember)

    def end(args: dict[str, Any]) -> str:
        ok = memory.end(args["claim_id"], at=args.get("at"))
        return "Ended." if ok else "No such fact is visible here; nothing changed."

    box.register(_schema(
        "memory_end",
        "Close a fact that was true and has stopped being true, with nothing replacing "
        "it: they left the job, the subscription lapsed. Pass at as the date it stopped "
        "if that is not now. The fact keeps answering questions about the period it "
        "held. If a new value replaces the old one, call memory_remember instead; it "
        "ends the old value itself. If the record was never right, use memory_forget.",
        {"claim_id": {"type": "string"},
         "at": {"type": "string", "description": "ISO-8601 date or instant it stopped being true. Default now."}},
        ["claim_id"]), end)

    def forget(args: dict[str, Any]) -> str:
        ok = memory.forget(args["claim_id"])
        return "Retired. It was recorded as never having been right." if ok else \
            "No such fact is visible here; nothing changed."

    box.register(_schema(
        "memory_forget",
        "Retire a fact that was never right: a misheard sentence, a note about someone "
        "else filed under this user. Before calling it, show the user the evidence from "
        "memory_why and let them confirm. This is not erasure; the record stays in the "
        "history marked as wrong. Do not use it for a value that has merely changed.",
        {"claim_id": {"type": "string"}},
        ["claim_id"]), forget)

    def why(args: dict[str, Any]) -> str:
        prov = memory.why(args["claim_id"])
        if prov is None:
            return "No fact with that id is visible here."
        return _json({
            "fact": f"{prov.claim.subject} {prov.claim.predicate} {prov.claim.object}",
            "state": prov.claim.state,
            "recorded_by": prov.extractor,
            "derivation": prov.derivation.value,
            "sources": [{"role": e.role, "said_at": e.ts.isoformat(), "text": e.content}
                        for e in prov.episodes],
            "replaced": [f"{c.subject} {c.predicate} {c.object}" for c in prov.superseded],
        })

    box.register(_schema(
        "memory_why",
        "Explain why a fact is believed: the turns it came from, what recorded it, and "
        "what it replaced. Call it when the user asks why you think something, or "
        "before correcting a fact, so the correction is made from the evidence rather "
        "than from the complaint.",
        {"claim_id": {"type": "string"}},
        ["claim_id"]), why)

    def history(args: dict[str, Any]) -> str:
        rows = memory.history(args.get("subject", "user"), args["predicate"])
        return _json(rows) if rows else "That fact has never held a value here."

    box.register(_schema(
        "memory_history",
        "Every value one fact has held, oldest first, with the interval each was true "
        "for and whether it is live, ended or retired. Use it for 'what was it before' "
        "and 'when did it change'. Do not quote an ended or retired row as current.",
        {"subject": {"type": "string", "description": "Default 'user'."},
         "predicate": {"type": "string"}},
        ["predicate"]), history)

    return box
