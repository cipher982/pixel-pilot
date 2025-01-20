from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class Action(BaseModel):
    """Represents an action to be taken"""

    command: str
    args: Optional[dict] = None


class ActionResponse(BaseModel):
    """Response from the LLM for next action decision"""

    action: Action
    next_path: str = Field(..., description="Next path to take: 'terminal', 'visual', or 'end'")
    is_task_complete: bool = False
    reasoning: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score between 0 and 1")
