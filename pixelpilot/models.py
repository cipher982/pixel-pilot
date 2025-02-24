from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class Action(BaseModel):
    """Base model for all actions"""

    type: Literal["terminal", "visual"]
    command: str
    args: Optional[dict] = None  # For terminal: subprocess.run kwargs, for visual: operation parameters


class ActionResponse(BaseModel):
    """Response from the LLM for next action decision"""

    action: Action
    next_path: str = Field(..., description="Next path to take: 'terminal', 'visual', or 'end'")
    is_task_complete: bool = False
    reasoning: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score between 0 and 1")
