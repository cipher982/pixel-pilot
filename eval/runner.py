"""Runner for PixelPilot evaluation tests."""

import json
import os
import subprocess
from contextlib import nullcontext
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
        # Run the command through your agent
        subprocess.run(
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
        with open("eval/artifacts/eval_result.json") as f:
            output = json.load(f)
            return TestResult(
                test_case=test_case,
                success=output["task_result"]["success"],
                actual_result=output["task_result"],
                trajectory=output.get("trajectory", []),
            )
    except Exception as e:
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
        # Similar to terminal test but with GUI flags
        subprocess.run(
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
        )

        # Read result
        with open("eval/artifacts/eval_result.json") as f:
            output = json.load(f)
            return TestResult(
                test_case=test_case,
                success=output["task_result"]["success"],
                actual_result=output["task_result"],
                trajectory=output.get("trajectory", []),
            )
    except Exception as e:
        return TestResult(
            test_case=test_case,
            success=False,
            actual_result={"error": str(e)},
            trajectory=[],
            error=str(e),
        )


def run_eval(test_case: TestCase, client: Optional[Client] = None) -> TestResult:
    """Run a single evaluation."""
    try:
        # Use context manager for run tracking if client available
        run_context = (
            client.run_tracker(
                project_name=os.getenv("LANGSMITH_PROJECT", "default"),
                name="PixelPilot Task Execution",
                inputs={"task": test_case.task},
            )
            if client
            else nullcontext()
        )

        with run_context as run:
            # Run based on test type
            result = run_terminal_test(test_case) if test_case.test_type == "terminal" else run_gui_test(test_case)

            # Update run if available
            if run and not result.error:
                run.end(outputs=result.actual_result)
            elif run:
                run.end(error=result.error)

            return result

    except Exception as e:
        return TestResult(
            test_case=test_case,
            success=False,
            actual_result={"error": str(e)},
            trajectory=[],
            error=str(e),
        )


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
    # Initialize LangSmith client if environment variables are set
    client = Client() if os.getenv("LANGCHAIN_API_KEY") else None

    if client:
        print("🔗 Connected to LangSmith")
    else:
        print("❌ Not connected to LangSmith")

    # Load test cases
    manager = DatasetManager()
    test_cases = manager.load_test_cases()
    print(f"\n📋 Found {len(test_cases)} test cases")

    # Run tests
    results = []
    for test_case in test_cases:
        print(f"\n🧪 Running test: {test_case.task}")
        result = run_eval(test_case, client)
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
