"""Test runner for evaluation."""

import json
import os
import subprocess
from typing import List

from eval.datasets import EvalCase
from eval.datasets import EvalResult


def run_terminal_test(test_case: EvalCase) -> EvalResult:
    """Run a terminal-based test case."""
    try:
        print(f"Running command in directory: {os.getcwd()}")
        # Run the command through your agent
        result = subprocess.run(
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
            capture_output=True,  # Capture output for debugging
        )

        print(f"Command output: {result.stdout}")
        if result.stderr:
            print(f"Command errors: {result.stderr}")

        # Read result from file
        try:
            with open("eval/artifacts/eval_result.json") as f:
                output = json.load(f)
                return EvalResult(
                    test_case=test_case,
                    success=output["task_result"]["success"],
                    actual_result=output["task_result"],
                    trajectory=output.get("trajectory", []),
                )
        except FileNotFoundError:
            print("eval_result.json not found - checking current directory")
            files = os.listdir(".")
            print(f"Current directory contents: {files}")
            raise

    except Exception as e:
        print(f"Error details: {str(e)}")
        return EvalResult(
            test_case=test_case,
            success=False,
            actual_result={"error": str(e)},
            trajectory=[],
            error=str(e),
        )


def run_gui_test(test_case: EvalCase) -> EvalResult:
    """Run a GUI-based test case."""
    try:
        print(f"Running command in directory: {os.getcwd()}")
        # Similar to terminal test but with GUI flags
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-u",
                "-m",
                "pixelpilot.main",
                "--output-format",
                "json",
                "--gui-mode",
                "--instructions",
                test_case.task,
                "--window-info",
                json.dumps(test_case.metadata.get("window_info", {})),
            ],
            text=True,
            check=True,
            env=os.environ.copy(),
            capture_output=True,  # Capture output for debugging
        )

        print(f"Command output: {result.stdout}")
        if result.stderr:
            print(f"Command errors: {result.stderr}")

        # Read result
        try:
            with open("eval/artifacts/eval_result.json") as f:
                output = json.load(f)
                return EvalResult(
                    test_case=test_case,
                    success=output["task_result"]["success"],
                    actual_result=output["task_result"],
                    trajectory=output.get("trajectory", []),
                )
        except FileNotFoundError:
            print("eval_result.json not found - checking current directory")
            files = os.listdir(".")
            print(f"Current directory contents: {files}")
            raise

    except Exception as e:
        print(f"Error details: {str(e)}")
        return EvalResult(
            test_case=test_case,
            success=False,
            actual_result={"error": str(e)},
            trajectory=[],
            error=str(e),
        )


def run_eval(test_case: EvalCase) -> EvalResult:
    """Run a single test case."""
    return run_terminal_test(test_case) if test_case.test_type == "terminal" else run_gui_test(test_case)


def save_results(results: List[EvalResult]) -> None:
    """Save test results to a JSON file."""
    output = {
        "results": [result.to_json() for result in results],
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
        },
    }

    os.makedirs("eval/artifacts", exist_ok=True)
    with open("eval/artifacts/results.json", "w") as f:
        json.dump(output, f, indent=2)
