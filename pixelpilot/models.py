from typing import Dict
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import Field


class Action(BaseModel):
    """Base model for all actions"""

    type: str = Field(description="Type of action to perform")
    reason: str = Field(description="Explanation of why this action was chosen")


class TerminalAction(Action):
    """Action for terminal operations"""

    type: Literal["terminal"]
    command: str = Field(description="Command to execute in terminal")


class VisualAction(Action):
    """Action for visual operations"""

    type: Literal["visual"]
    operation: Literal["click", "scroll", "read"] = Field(description="Type of visual operation")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Target coordinates for the operation")
    element_id: Optional[str] = Field(None, description="Target element identifier")


class ActionResponse(BaseModel):
    """LLM response format for next action decision"""

    action: Union[TerminalAction, VisualAction] = Field(description="Next action to take")
    next_path: Literal["terminal", "visual", "end"] = Field(description="Which path to take next")
    reasoning: str = Field(description="Explanation of the decision")
    is_task_complete: bool = Field(default=False, description="Whether the task has been completed")
