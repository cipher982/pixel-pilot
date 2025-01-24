"""Data structures for test cases and results."""

from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional


@dataclass
class EvalCase:
    """Represents a single test case."""

    task: str
    expected_result: Dict
    test_type: Literal["terminal", "gui", "mixed"] = "terminal"
    expected_trajectory: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Dict) -> "EvalCase":
        """Create a EvalCase from a JSON dictionary."""
        # Default to terminal type for backward compatibility
        test_type = data.get("test_type", "terminal")
        return cls(
            task=data["task"],
            expected_result=data["expected_result"],
            test_type=test_type,
            expected_trajectory=data.get("expected_trajectory"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> Dict:
        """Convert to JSON dictionary."""
        return {
            "task": self.task,
            "expected_result": self.expected_result,
            "test_type": self.test_type,
            "expected_trajectory": self.expected_trajectory,
            "metadata": self.metadata,
        }


@dataclass
class EvalResult:
    """Represents the result of a test execution."""

    test_case: EvalCase
    success: bool
    actual_result: Dict
    trajectory: List[str]
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> Dict:
        """Convert to JSON dictionary."""
        return {
            "test": self.test_case.to_json(),
            "success": self.success,
            "actual_result": self.actual_result,
            "trajectory": self.trajectory,
            "error": self.error,
            "metrics": self.metrics,
        }
