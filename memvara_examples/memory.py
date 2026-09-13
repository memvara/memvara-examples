"""One user's memory, behind the calls the agents make.

The agents never touch ``memvara`` directly. They hold a :class:`Memory`, which wraps a
scoped Memvara handle and exposes the handful of operations an agent needs. The same
class works against a hosted deployment (``Memvara.connect()``, the default) and against
a local SQLite file (``--local``), so an agent written once runs either way.

Three rules from Memvara's own documentation are applied here, once, so that neither
agent can get them wrong:

* A fact is written as a triple with :meth:`Memory.remember`, never as prose. A hosted
  deployment may have no extraction model, and a paragraph it does not recognise is
  accepted and stored as nothing.
* Setting a fact that already has a value ends the old value on the world clock and then
  writes the new one. Nothing is deleted; ``history`` still shows both.
* Closing a fact has two different meanings. :meth:`Memory.end` says the fact was true and
  has stopped being true. :meth:`Memory.forget` says the record was never right. They are
  separate methods so the caller has to choose.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from memvara import Memvara, MemoryType, NullLLM
from memvara.types import Answer, Claim, Provenance, WriteReceipt


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_when(value: str | datetime | None) -> datetime | None:
    """An ISO-8601 date or instant as an aware UTC datetime, or None for None.

    ``2026-03-01`` and ``2026-03-01T09:00:00Z`` are both accepted. A value with no
    timezone is read as UTC, because a model passing ``at`` has no local clock.
    """
    if value is None or isinstance(value, datetime):
        return value
    text = value.strip()
    if text[-1:] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _summary(claim: Claim) -> dict[str, Any]:
    """A claim as the small dictionary the model is shown."""
    return {
        "id": claim.id,
        "subject": claim.subject,
        "predicate": claim.predicate,
        "object": claim.object,
        "type": claim.memory_type.value,
        "state": claim.state,
        "true_since": claim.valid_from.isoformat(),
        "true_until": claim.valid_to.isoformat() if claim.valid_to else None,
        "confidence": claim.confidence,
    }


class Memory:
    """One user's memory, local or hosted.

    Construct it with :meth:`Memory.open`. Every method is bound to one user; there is
    no argument that reaches another user's facts.
    """

    def __init__(self, root: Any, scoped: Any, *, hosted: bool, user: str,
                 label: str) -> None:
        self._root = root
        self._m = scoped
        self.hosted = hosted
        self.user = user
        self.label = label

    @classmethod
    def open(cls, *, user: str, local: str | None = None) -> "Memory":
        """Open the memory for ``user``.

        With ``local`` set, the store is a SQLite file at that path and needs no network
        and no account. Without it, the hosted deployment is used, with the credential
        ``memvara-mcp login`` wrote to ``~/.memvara/credentials.json`` or the one in
        ``MEMVARA_API_KEY``. A missing credential raises ``MissingCredential`` with the
        fix in the message.
        """
        if local:
            root = Memvara(local, user=user, llm=NullLLM())
            return cls(root, root.scope(user=user), hosted=False, user=user,
                       label=f"local store {local}")
        root = Memvara.connect()
        return cls(root, root.scope(user=user), hosted=True, user=user,
                   label="hosted Memvara (app.memvara.dev)")

    def close(self) -> None:
        self._root.close()

    # -- what this store is ----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """A few facts about the store, for the CLI's preflight check.

        Against a hosted deployment this is one request that also proves the credential
        works. Against a local file it is a row count.
        """
        if self.hosted:
            service = self._root.service()
            visible = service.get("visible")
            if isinstance(visible, dict):
                visible = visible.get("claims")
            extractor = service.get("extractor")
            read_only = bool(service.get("read_only", False))
        else:
            stats = self._root.stats()
            visible = stats.get("live_claims", stats.get("claims"))
            extractor = "none (facts are written as triples)"
            read_only = False
        return {"store": self.label, "user": self.user, "claims_visible": visible,
                "extractor": extractor, "read_only": read_only}

    # -- reading ----------------------------------------------------------------

    def recall(self, query: str, *, k: int = 8) -> str:
        """Notes relevant to ``query``, already formatted for a prompt."""
        return str(self._m.recall(query, k=k))

    def standing(self) -> list[Claim]:
        """The user's standing preferences: how they want work done.

        A hosted deployment answers this server-side. The local engine has no such call,
        so the scope is paged and filtered here.
        """
        if self.hosted:
            return list(self._m.standing())
        return [c for c in self._m.get_all()
                if c.memory_type == MemoryType.PROCEDURAL and c.subject == "user"]

    def search(self, query: str, *, k: int = 10,
               valid_at: datetime | None = None) -> list[dict[str, Any]]:
        """Matching facts with their ids, for the correction flow."""
        rows = self._m.search(query, k=k, valid_at=valid_at)
        return [{**_summary(r.claim), "relevance": round(r.score, 3)} for r in rows]

    def history(self, subject: str, predicate: str) -> list[dict[str, Any]]:
        """Every value one fact has held, oldest first, with the interval each held."""
        return [_summary(c) for c in self._m.history(subject, predicate)]

    def why(self, claim_id: str) -> Provenance | None:
        return self._m.why(claim_id)

    def ask(self, question: str, *, at: datetime | None = None) -> Answer:
        """What was true at ``at``, and what the store would have said then."""
        return self._m.ask(question, at=at)

    # -- writing ----------------------------------------------------------------

    def add(self, text: str, *, role: str = "system") -> list[str]:
        """Store one conversation turn and return its episode ids.

        Used to give a fact a source: the turn the user said it in. The default role is
        ``system`` on purpose. A turn stored with ``role="user"`` is also read by the
        rule-based extractor, which would write its own reading of the sentence beside
        the fact the agent is about to write, so two writers would disagree about one
        turn. With ``system`` the turn is stored and cited and nothing is extracted from
        it; the agent is the only writer.
        """
        receipt = self._m.add(text, role=role)
        return list(receipt.episode_ids)

    def remember(self, subject: str, predicate: str, obj: str, *,
                 memory_type: str | None = None,
                 true_since: str | datetime | None = None,
                 confidence: float = 1.0,
                 source_text: str | None = None) -> WriteReceipt:
        """Write one fact as a triple, citing ``source_text`` if given.

        Against a local store the source turn is stored first and cited by id. Against a
        hosted deployment the turn is sent inside the same request, so the fact and its
        source land in one transaction.
        """
        since = parse_when(true_since)
        kind = MemoryType(memory_type) if memory_type else None
        sources: list[Any] | None = None
        if source_text:
            if self.hosted:
                sources = [{"role": "system", "content": source_text}]
            else:
                sources = self.add(source_text)
        return self._m.remember(subject, predicate, obj, memory_type=kind,
                                valid_from=since, confidence=confidence,
                                sources=sources)

    def set_fact(self, subject: str, predicate: str, obj: str, *,
                 memory_type: str | None = None,
                 true_since: str | datetime | None = None,
                 source_text: str | None = None) -> tuple[list[Claim], WriteReceipt]:
        """Replace whatever this fact currently says with ``obj``.

        The new value is written first, then every other live value in the slot is
        ended on the world clock at ``true_since`` (or now). Writing first means a
        failure between the two steps leaves two live values, which history shows and
        the next call repairs, rather than a slot with no value at all. This is
        deterministic whatever the predicate's declared cardinality, which matters for a
        project vocabulary the store has never seen: an undeclared predicate is
        multi-valued by default, so a plain ``remember`` would add a second value beside
        the first instead of replacing it. Returns the values that were ended and the
        receipt for the new one.
        """
        at = parse_when(true_since) or utcnow()
        before = [c for c in self._m.history(subject, predicate) if c.state == "live"]
        receipt = self.remember(subject, predicate, obj, memory_type=memory_type,
                                true_since=at, source_text=source_text)
        # Re-asserting the value the slot already holds is a reinforcement: the store
        # returns nothing added and nothing closed, and the live claim must stay live.
        keep = ({c.id for c in receipt.added} | {c.id for c in receipt.closed}
                | {c.id for c in receipt.reinforced})
        ended = list(receipt.closed)
        for claim in before:
            if claim.id in keep or claim.object == obj:
                continue
            if self._m.delete(claim.id, at=at, close="ended"):
                ended.append(claim)
        return ended, receipt

    def end(self, claim_id: str, *, at: str | datetime | None = None) -> bool:
        """The fact was true and has stopped being true, at ``at`` (default now)."""
        return bool(self._m.delete(claim_id, at=parse_when(at), close="ended"))

    def forget(self, claim_id: str) -> bool:
        """The record was never right. It is retired, not erased."""
        return bool(self._m.delete(claim_id, close="retired"))
