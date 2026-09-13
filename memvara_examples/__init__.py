"""Working AI agents that keep their long-term memory in Memvara.

Two agents ship here, and a CLI that starts either one:

* ``assistant`` is a personal assistant. It reads what it knows about you before every
  reply and writes down what it learns as structured facts.
* ``engineer`` keeps the record of a software project's decisions. It answers what is
  true now, what was true before, when that changed and why.

Both talk to Claude through the Anthropic SDK and to Memvara through the ``memvara``
library. The memory layer is in :mod:`memvara_examples.memory`, the model layer in
:mod:`memvara_examples.model`, and the tools the model can call in
:mod:`memvara_examples.tools`.
"""

__version__ = "0.1.0"
