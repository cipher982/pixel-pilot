"""Data structures for test cases and results."""

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

from eval.verification import VerificationResult


@dataclass
class VerificationRule:
    """A rule for verifying a specific outcome"""

    type: Literal["file_exists", "file_content", "visual_element", "terminal_output"]
    condition: Dict[str, Any]  # Specific conditions for this rule type
    description: str  # Human readable description of what we're verifying
    required: bool = True  # Whether this rule must pass for task success

    def to_json(self) -> Dict:
        """Convert to JSON dictionary."""
        return {
            "type": self.type,
            "condition": self.condition,
            "description": self.description,
            "required": self.required,
        }


@dataclass
class EvalCase:
    """Represents a single test case with verifiable outcomes."""

    task: str
    verification_rules: List[VerificationRule]
    test_type: Literal["terminal", "gui", "mixed"] = "terminal"
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Dict) -> "EvalCase":
        """Create an EvalCase from a JSON dictionary."""
        rules = []
        for rule in data.get("verification_rules", []):
            rules.append(VerificationRule(**rule))

        return cls(
            task=data["task"],
            verification_rules=rules,
            test_type=data.get("test_type", "terminal"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> Dict:
        """Convert to JSON dictionary."""
        return {
            "task": self.task,
            "verification_rules": [
                {
                    "type": rule.type,
                    "condition": rule.condition,
                    "description": rule.description,
                    "required": rule.required,
                }
                for rule in self.verification_rules
            ],
            "test_type": self.test_type,
            "metadata": self.metadata,
        }


@dataclass
class ActionRecord:
    """Record of an action taken by the agent"""

    type: Literal["terminal_command", "visual_interaction", "state_change"]
    description: str
    timestamp: float
    details: Dict[str, Any]
    success: bool
    error: Optional[str] = None


@dataclass
class EvalResult:
    """Result of a test execution with detailed action history."""

    test_case: EvalCase
    success: bool
    verification_results: List[VerificationResult]  # Results of each verification rule
    actions: List[ActionRecord]  # What the agent actually did
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> Dict:
        """Convert to JSON dictionary."""
        return {
            "test": self.test_case.to_json(),
            "success": self.success,
            "verification_results": [
                {"rule": v.rule.to_json(), "passed": v.passed, "details": v.details} for v in self.verification_results
            ],
            "actions": [
                {
                    "type": a.type,
                    "description": a.description,
                    "timestamp": a.timestamp,
                    "details": a.details,
                    "success": a.success,
                    "error": a.error,
                }
                for a in self.actions
            ],
            "error": self.error,
            "metrics": self.metrics,
        }
