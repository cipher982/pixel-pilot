"""Verification system for evaluating test outcomes."""

import os
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Tuple

from eval.datasets.test_case import VerificationRule


class VerificationResult:
    """Result of a verification check."""

    def __init__(self, rule: VerificationRule, passed: bool, details: Dict):
        self.rule = rule
        self.passed = passed
        self.details = details

    def to_dict(self) -> Dict:
        return {"rule": self.rule, "passed": self.passed, "details": self.details}


class Verifier(ABC):
    """Base class for verification implementations."""

    @abstractmethod
    def verify(self, rule: VerificationRule, state: Dict) -> VerificationResult:
        """Verify if the rule passes given the current state."""
        raise NotImplementedError("Subclasses must implement verify()")  # type: ignore


class FileVerifier(Verifier):
    """Verifies file-related conditions."""

    def verify(self, rule: VerificationRule, state: Dict) -> VerificationResult:
        if rule.type not in ["file_exists", "file_content"]:
            raise ValueError(f"FileVerifier cannot handle rule type: {rule.type}")

        if rule.type == "file_exists":
            path = rule.condition.get("path")
            if not path:
                return VerificationResult(rule=rule, passed=False, details={"error": "No path specified in condition"})

            exists = os.path.exists(path)
            return VerificationResult(rule=rule, passed=exists, details={"exists": exists, "path": path})

        elif rule.type == "file_content":
            path = rule.condition.get("path")
            if not path or not os.path.exists(path):
                return VerificationResult(rule=rule, passed=False, details={"error": f"File not found: {path}"})

            try:
                with open(path, "r") as f:
                    content = f.read()

                # Check different content conditions
                if "contains" in rule.condition:
                    contains = rule.condition["contains"]
                    passed = contains in content
                    details = {"contains_match": passed, "searched_for": contains}

                elif "matches" in rule.condition:
                    matches = rule.condition["matches"]
                    passed = content.strip() == matches.strip()
                    details = {"exact_match": passed, "expected": matches}

                else:
                    passed = False
                    details = {"error": "No content matching rule specified"}

                return VerificationResult(rule=rule, passed=passed, details=details)

            except Exception as e:
                return VerificationResult(rule=rule, passed=False, details={"error": f"Failed to read file: {str(e)}"})


class VisualVerifier(Verifier):
    """Verifies visual/GUI-related conditions."""

    def verify(self, rule: VerificationRule, state: Dict) -> VerificationResult:
        if rule.type != "visual_element":
            raise ValueError(f"VisualVerifier cannot handle rule type: {rule.type}")

        # Get the latest screen state
        screen_state = state.get("screen_state", {})
        if not screen_state:
            return VerificationResult(rule=rule, passed=False, details={"error": "No screen state available"})

        # Check different visual conditions
        if "text_contains" in rule.condition:
            text_list = rule.condition["text_contains"]
            if isinstance(text_list, str):
                text_list = [text_list]

            # Check if any of the texts are present
            found_texts = []
            for text in text_list:
                if text.lower() in screen_state.get("text", "").lower():
                    found_texts.append(text)

            passed = len(found_texts) > 0
            return VerificationResult(
                rule=rule, passed=passed, details={"found_texts": found_texts, "searched_for": text_list}
            )

        elif "element_visible" in rule.condition:
            element_id = rule.condition.get("element_id")
            expected_visible = rule.condition["element_visible"]

            elements = screen_state.get("elements", {})
            actual_visible = elements.get(element_id, {}).get("visible", False)

            passed = actual_visible == expected_visible
            return VerificationResult(
                rule=rule,
                passed=passed,
                details={
                    "element_id": element_id,
                    "expected_visible": expected_visible,
                    "actual_visible": actual_visible,
                },
            )

        return VerificationResult(rule=rule, passed=False, details={"error": "Unknown visual condition type"})


class VerificationEngine:
    """Main engine for running verifications."""

    def __init__(self):
        self.verifiers = {
            "file_exists": FileVerifier(),
            "file_content": FileVerifier(),
            "visual_element": VisualVerifier(),
        }

    def verify_all(self, rules: List[VerificationRule], state: Dict) -> Tuple[bool, List[VerificationResult]]:
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
