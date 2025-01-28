"""Common types for evaluation system."""

from dataclasses import dataclass
from typing import Any
from typing import Dict


@dataclass
class VerificationRule:
    """A rule for verifying test results."""

    type: str
    description: str
    condition: Dict[str, Any]
    required: bool = True

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON dictionary."""
        return {
            "type": self.type,
            "condition": self.condition,
            "description": self.description,
            "required": self.required,
        }
