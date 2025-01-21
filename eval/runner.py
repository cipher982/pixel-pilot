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
    try:
        # Run the command and ignore stdout/stderr
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "pixelpilot.main",
                "--output-format",
                "json",
                "--instructions",
                test_case.task,
            ],
            capture_output=True,
            text=True,
            check=True,
            env=os.environ.copy(),
        )

        # Read result from file
        try:
            with open("eval_result.json") as f:
                output = json.load(f)
                return {
                    "success": output["task_result"]["success"],
                    "actual_result": output["task_result"],
                    "matches_expected": validate_result(output["task_result"], test_case.expected_result),
                }
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return {
                "success": False,
                "error": f"Failed to read results: {str(e)}",
            }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"Process failed with exit code {e.returncode}",
            "raw_output": e.stdout,
            "raw_error": e.stderr,
        }
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}", "exception_type": type(e).__name__}


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
