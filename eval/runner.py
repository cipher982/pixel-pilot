import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict


@dataclass
class TestCase:
    task: str
    expected_result: Dict

    @classmethod
    def from_file(cls, path: str) -> "TestCase":
        with open(path) as f:
            data = json.load(f)
            return cls(task=data["task"], expected_result=data["expected_result"])


def run_eval(test_case: TestCase) -> Dict:
    """Run a single evaluation"""
    result = subprocess.run(
        ["python", "-m", "pixelpilot.main", "--output-format", "json", "--instructions", test_case.task],
        capture_output=True,
        text=True,
    )

    try:
        output = json.loads(result.stdout)
        return {
            "success": output["task_result"]["success"],
            "actual_result": output["task_result"],
            "matches_expected": validate_result(output["task_result"], test_case.expected_result),
        }
    except json.JSONDecodeError:
        return {"success": False, "error": "Failed to parse output", "raw_output": result.stdout}


def validate_result(actual: Dict, expected: Dict) -> bool:
    """Validate actual result against expected result"""
    # Basic file existence checks
    if "files" in expected:
        for fname, fspec in expected["files"].items():
            if fspec.get("exists", True) != os.path.exists(fname):
                return False

    # Success flag check
    if "success" in expected:
        if actual["success"] != expected["success"]:
            return False

    return True


def main():
    """Run all test cases in test_cases directory"""
    test_dir = "eval/test_cases"
    results = []

    for filename in os.listdir(test_dir):
        if filename.endswith(".json"):
            test_case = TestCase.from_file(os.path.join(test_dir, filename))
            result = run_eval(test_case)
            results.append({"test": filename, **result})

    # Print results
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
