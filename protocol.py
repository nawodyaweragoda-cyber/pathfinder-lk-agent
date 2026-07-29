"""A2A-inspired structured message protocol for agent-to-agent communication.

Every agent in PathFinder LK communicates by exchanging `AgentMessage` objects
instead of raw strings. This gives us:
  - a `trace_id` so one user query can be followed across all agents
  - a `performative` (speech-act type) so receivers know how to interpret content
  - a typed `content` payload validated by pydantic
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Performative = Literal[
    "request",    # asking another agent to do work
    "inform",     # returning results / data
    "propose",    # a draft answer submitted for critique
    "critique",   # feedback on a proposal (reflection pattern)
    "final",      # the finished answer for the user
    "reject",     # query out of scope
]


class AgentMessage(BaseModel):
    trace_id: str
    sender: str
    receiver: str
    performative: Performative
    intent: Optional[str] = None
    content: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def new_trace(cls, **kwargs: Any) -> "AgentMessage":
        """Create the first message of a new conversation trace."""
        return cls(trace_id=str(uuid.uuid4()), **kwargs)

    def reply(
        self,
        sender: str,
        performative: Performative,
        content: dict[str, Any],
        intent: Optional[str] = None,
    ) -> "AgentMessage":
        """Build a reply that stays on the same trace, addressed back to sender."""
        return AgentMessage(
            trace_id=self.trace_id,
            sender=sender,
            receiver=self.sender,
            performative=performative,
            intent=intent or self.intent,
            content=content,
        )

    def pretty(self) -> str:
        return (
            f"[{self.sender} -> {self.receiver}] "
            f"({self.performative}/{self.intent}) {self.content}"
        )
