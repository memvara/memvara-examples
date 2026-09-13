import pytest

from memvara_examples.memory import Memory


@pytest.fixture
def memory(tmp_path):
    """A local store in a temporary file, for one test."""
    mem = Memory.open(user="alice", local=str(tmp_path / "memory.db"))
    yield mem
    mem.close()
