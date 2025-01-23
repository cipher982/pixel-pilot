"""Runner for PixelPilot evaluation tests."""

import json
import os
import subprocess
from typing import List
from typing import Optional

from langsmith import Client
from langsmith import RunEvaluator
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example
from langsmith.schemas import Run

from eval.datasets import DatasetManager
from eval.datasets import TestCase
from eval.datasets import TestResult


def run_terminal_test(test_case: TestCase) -> TestResult:
    """Run a terminal-based test."""
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
                return TestResult(
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
        return TestResult(
            test_case=test_case,
            success=False,
            actual_result={"error": str(e)},
            trajectory=[],
            error=str(e),
        )


def run_gui_test(test_case: TestCase) -> TestResult:
    """Run a GUI-based test."""
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
                return TestResult(
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
        return TestResult(
            test_case=test_case,
            success=False,
            actual_result={"error": str(e)},
            trajectory=[],
            error=str(e),
        )


def run_eval(test_case: TestCase, client: Optional[Client] = None) -> TestResult:
    """Run a single evaluation."""
    # Run based on test type
    return run_terminal_test(test_case) if test_case.test_type == "terminal" else run_gui_test(test_case)


def save_results(results: List[TestResult]) -> None:
    """Save test results to file."""
    output = {
        "results": [result.to_json() for result in results],
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
        },
    }

    with open("eval/artifacts/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)


def main():
    """Run all test cases."""
    # Load test cases
    manager = DatasetManager()
    test_cases = manager.load_test_cases()
    print(f"\n📋 Found {len(test_cases)} test cases")

    # Run tests
    results = []
    for test_case in test_cases:
        print(f"\n🧪 Running test: {test_case.task}")
        result = run_eval(test_case)
        results.append(result)
        print(f"{'✅' if result.success else '❌'} Test completed")

    # Save results
    save_results(results)
    print("\n📊 Results saved to eval/artifacts/eval_results.json")


class TerminalCommandEvaluator(RunEvaluator):
    """Evaluates terminal command execution quality using LLM."""

    def __init__(self, llm):
        """Initialize with an LLM instance."""
        self.llm = llm
        super().__init__()

    def evaluate_run(self, run: Run, example: Optional[Example] = None) -> EvaluationResult:
        if not run.outputs:
            return EvaluationResult(key="terminal_quality", score=0.0, feedback="No outputs available for evaluation")

        criteria = """Evaluate the terminal command execution based on:
        1. Correctness: Did it achieve the desired outcome?
        2. Efficiency: Was it the most appropriate method?
        3. Safety: Were proper precautions taken?
        4. Robustness: Would it work in edge cases?
        
        Task: {task}
        Actual Result: {actual}
        """

        # Get the LLM's evaluation
        evaluation = self.llm.predict(criteria.format(task=run.inputs.get("task", ""), actual=str(run.outputs)))

        # Parse evaluation into structured feedback
        return EvaluationResult(key="terminal_quality", score=self._parse_score(evaluation), feedback=evaluation)

    def _parse_score(self, evaluation: str) -> float:
        # Extract numerical score from LLM evaluation
        # This is a simple implementation - could be more sophisticated
        if "excellent" in evaluation.lower():
            return 1.0
        elif "good" in evaluation.lower():
            return 0.8
        elif "acceptable" in evaluation.lower():
            return 0.6
        else:
            return 0.4


if __name__ == "__main__":
    main()
