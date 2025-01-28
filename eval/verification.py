"""Verification system for evaluating test results."""

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from eval.datasets.types import VerificationRule
from pixelpilot.logger import setup_logger
from pixelpilot.system_control import SystemController

logger = setup_logger(__name__)


class VerificationResult:
    """Result of a verification check."""

    def __init__(self, rule: "VerificationRule", passed: bool, details: Dict):
        self.rule = rule
        self.passed = passed
        self.details = details

    def to_dict(self) -> Dict:
        return {"rule": self.rule, "passed": self.passed, "details": self.details}


class Verifier(ABC):
    """Base class for verification implementations."""

    @abstractmethod
    def verify(self, rule: "VerificationRule", state: Dict) -> VerificationResult:
        """Verify if the rule passes given the current state."""
        raise NotImplementedError("Subclasses must implement verify()")


class FileVerifier(Verifier):
    """Verifies file-related conditions using the system controller."""

    def verify(self, rule: "VerificationRule", state: Dict) -> VerificationResult:
        if rule.type not in ["file", "file_exists", "file_content"]:
            raise ValueError(f"FileVerifier cannot handle rule type: {rule.type}")

        # Get controller from state
        controller: Optional[SystemController] = state.get("controller")
        if not controller:
            return VerificationResult(
                rule=rule, passed=False, details={"error": "No system controller available for verification"}
            )

        path = rule.condition.get("path")
        if not path:
            return VerificationResult(rule=rule, passed=False, details={"error": "No path specified in condition"})

        try:
            # Check if file exists using ls command
            result = controller.run_command(f"ls {path}")
            exists = result.success

            if not exists:
                return VerificationResult(rule=rule, passed=False, details={"error": f"File not found: {path}"})

            # If we need to check content
            if "content" in rule.condition or "matches" in rule.condition:
                result = controller.run_command(f"cat {path}")
                if not result.success:
                    return VerificationResult(
                        rule=rule, passed=False, details={"error": f"Failed to read file: {result.message}"}
                    )

                content = result.message
                expected = rule.condition.get("content") or rule.condition.get("matches")
                passed = expected in content
                return VerificationResult(rule=rule, passed=passed, details={"matches": passed, "path": path})

            return VerificationResult(rule=rule, passed=True, details={"exists": True, "path": path})

        except Exception as e:
            return VerificationResult(rule=rule, passed=False, details={"error": f"Failed to verify file: {str(e)}"})


class VisionVerifier(Verifier):
    """Uses AI to verify screenshots match expected state."""

    def verify(self, rule: "VerificationRule", state: Dict) -> VerificationResult:
        if rule.type != "vision":
            raise ValueError(f"VisionVerifier cannot handle rule type: {rule.type}")

        # Get final screenshot
        screen_state = state.get("screen_state", {})
        if not screen_state or "final_screenshot" not in screen_state:
            return VerificationResult(rule=rule, passed=False, details={"error": "No final screenshot available"})

        # Get the description of what we expect to see
        expected_description = rule.condition.get("description")
        if not expected_description:
            return VerificationResult(
                rule=rule, passed=False, details={"error": "No description provided of expected visual state"}
            )

        # TODO: Call AI to verify screenshot matches description
        # For now just print what we would ask
        print(f"Would ask AI: Does this screenshot show: {expected_description}?")

        # Temporary placeholder - always fail until AI integration
        return VerificationResult(
            rule=rule,
            passed=False,
            details={"error": "AI vision verification not yet implemented", "expected": expected_description},
        )


class VerificationEngine:
    """Main engine for running verifications."""

    def __init__(self):
        self.verifiers = {
            "file": FileVerifier(),
            "file_exists": FileVerifier(),
            "file_content": FileVerifier(),
            "vision": VisionVerifier(),
        }

    def verify_all(self, rules: List["VerificationRule"], state: Dict) -> Tuple[bool, List[VerificationResult]]:
        """Verify all rules and determine overall success."""
        results = []
        success = True

        for rule in rules:
            if rule.type not in self.verifiers:
                result = VerificationResult(
                    rule=rule, passed=False, details={"error": f"No verifier for rule type: {rule.type}"}
                )
                if rule.required:
                    success = False
            else:
                verifier = self.verifiers[rule.type]
                result = verifier.verify(rule, state)
                if rule.required and not result.passed:
                    success = False

            results.append(result)

        return success, results
