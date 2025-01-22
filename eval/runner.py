import json
import os
import subprocess
import sys
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
        # Run the command and allow output to flow through
        _ = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-u",  # Unbuffered output
                "-m",
                "pixelpilot.main",
                "--output-format",
                "json",
                "--instructions",
                test_case.task,
            ],
            text=True,
            check=True,
            env=os.environ.copy(),
        )

        # Read result from file
        try:
            with open("eval/artifacts/eval_result.json") as f:
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
    print("Direct stdout test", flush=True)  # Direct to stdout
    sys.stdout.write("Direct stdout write test\n")  # Another direct test
    sys.stdout.flush()

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
