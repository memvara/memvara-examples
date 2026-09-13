"""The agents the CLI can start, by name."""

from .assistant import AssistantAgent
from .engineer import EngineerAgent

AGENTS = {
    AssistantAgent.name: AssistantAgent,
    EngineerAgent.name: EngineerAgent,
}

__all__ = ["AGENTS", "AssistantAgent", "EngineerAgent"]
