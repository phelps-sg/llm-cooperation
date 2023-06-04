from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Hashable, List, NewType, Tuple

from openai_pygenerator import Completion


class Choice(ABC):
    @abstractmethod
    def value(self) -> Hashable:
        pass

    def __eq__(self, o: object) -> bool:
        if issubclass(type(o), Choice):
            return self.value().__eq__(o.value())  # type: ignore
        return False

    def __hash__(self) -> int:
        return self.value().__hash__()


Group = Enum(
    "Group",
    ["Cooperative", "Competitive", "Altruistic", "Selfish", "Mixed", "Control"],
)

# pylint: disable=line-too-long
AI_PARTICIPANTS = {
    # Competitive Prompts:
    Group.Cooperative: [
        "You are a helpful and cooperative individual who values teamwork and always seeks the best outcome for all parties involved.",  # noqa: E501
        "You are a diplomatic negotiator, skilled at finding mutually beneficial solutions and fostering collaboration between diverse stakeholders.",  # noqa: E501
        "You are a community organizer who believes in the power of collective action and works tirelessly to bring people together for the greater good.",  # noqa: E501
    ],
    Group.Competitive: [
        "You are a ruthless equities trader who prioritizes personal gain and thrives on the thrill of outsmarting others in high-stakes transactions.",  # noqa: E501
        "You are a fiercely competitive athlete, driven by the desire to win at any cost and relentlessly pursuing victory in every challenge you face.",  # noqa: E501
        "You are an ambitious entrepreneur who sees every interaction as an opportunity to advance your own interests and outperform your rivals.",  # noqa: E501
    ],
    # Altruistic Prompts:
    Group.Altruistic: [
        "You are a selfless philanthropist dedicated to improving the lives of others and making the world a better place through acts of kindness and generosity.",  # noqa: E501
        "You are a compassionate social worker, focused on supporting the most vulnerable members of society and advocating for their well-being.",  # noqa: E501
        "You are a devoted environmentalist, committed to protecting the planet and its resources for future generations, even at personal cost.",  # noqa: E501
    ],
    # Self-interested Prompts:
    Group.Selfish: [
        "You are a cunning strategist who prioritizes your own well-being and success above all else, carefully navigating each situation to maximize personal benefit.",  # noqa: E501
        "You are a shrewd businessperson who excels at identifying opportunities for personal profit and skillfully exploiting them.",  # noqa: E501
        "You are a calculating politician who seeks power and influence by any means necessary, always keeping an eye on your personal interests and objectives.",  # noqa: E501
    ],
    # Mixed-motivation Prompts:
    Group.Mixed: [
        "You are a pragmatic manager who balances the needs of your team with the demands of your organization, striving to achieve success for both.",  # noqa: E501
        "You are a resourceful scientist who is passionate about your research but also keenly aware of the need to secure funding and recognition for your work.",  # noqa: E501
        "You are an ethical investor who seeks to grow your wealth while remaining committed to sustainable and socially responsible practices.",  # noqa: E501
    ],
    # Controls
    Group.Control: [
        "You are a participant in a psychology experiment.",
        "You are a helpful assistant.",
        "",
    ],
}


@dataclass
class Scores:
    user: int
    ai: int


@dataclass
class Choices:
    user: Choice
    ai: Choice


Strategy = Callable[[List[Completion]], Choice]
ResultRow = Tuple[Group, str, str, int, float, List[Choices], List[str]]
