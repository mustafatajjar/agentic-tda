from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AgentInput:
    data: Any
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentOutput:
    result: Any
    metadata: Optional[Dict[str, Any]] = None


class Agent(ABC):
    @abstractmethod
    def run(self, input: AgentInput) -> AgentOutput:
        pass