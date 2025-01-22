import json
import os
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict
from typing import Optional

from langsmith import Client
from langsmith import RunEvaluator
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example
from langsmith.schemas import Run


@dataclass
class TestCase:
    task: str
    expected_result: Dict

    @classmethod
    def from_file(cls, path: str) -> "TestCase":
        with open(path) as f:
            data = json.load(f)
            return cls(task=data["task"], expected_result=data["expected_result"])

    def to_langsmith_example(self) -> Example:
        """Convert test case to LangSmith example format"""
        return Example(inputs={"task": self.task}, outputs=self.expected_result)


def run_eval(test_case: TestCase, client: Optional[Client] = None) -> Dict:
    """Run a single evaluation"""
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
                    result = {
                        "success": output["task_result"]["success"],
                        "actual_result": output["task_result"],
                        "matches_expected": validate_result(output["task_result"], test_case.expected_result),
                    }

                    # Update run if available
                    if run:
                        run.end(outputs=output["task_result"])

                    return result
            except (FileNotFoundError, json.JSONDecodeError) as e:
                error_result = {
                    "success": False,
                    "error": f"Failed to read results: {str(e)}",
                }
                if run:
                    run.end(error=str(e))
                return error_result
    except subprocess.CalledProcessError as e:
        error_result = {
            "success": False,
            "error": f"Process failed with exit code {e.returncode}",
        }
        if run:
            run.end(error=str(e))
        return error_result
    except Exception as e:
        error_result = {"success": False, "error": f"Unexpected error: {str(e)}", "exception_type": type(e).__name__}
        if run:
            run.end(error=str(e))
        return error_result


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
    # Initialize LangSmith client if environment variables are set
    client = Client() if os.getenv("LANGCHAIN_API_KEY") else None

    if client:
        print("🔗 Connected to LangSmith")
        # Create or get dataset
        dataset_name = "PixelPilot Evaluation"
        try:
            dataset = client.create_dataset(dataset_name)
        except Exception as _:
            dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        print("❌ Not connected to LangSmith")
        dataset = None

    test_dir = "eval/test_cases"
    results = []

    for filename in os.listdir(test_dir):
        if filename.endswith(".json"):
            test_case = TestCase.from_file(os.path.join(test_dir, filename))

            # Add example to dataset if using LangSmith
            if client and dataset:
                client.create_example(
                    inputs={"task": test_case.task}, outputs=test_case.expected_result, dataset_id=dataset.id
                )

            # Run evaluation
            result = run_eval(test_case, client)
            results.append({"test": filename, **result})

    # Print JSON results as before
    print(json.dumps({"results": results}, indent=2))

    # Run LangSmith evaluators if available
    if client and dataset:
        from langchain.chat_models import ChatOpenAI

        llm = ChatOpenAI(temperature=0)
        evaluator = TerminalCommandEvaluator(llm)

        runs = client.list_runs(project_name=os.getenv("LANGSMITH_PROJECT", "default"), dataset_name=dataset_name)

        for run in runs:
            eval_result = evaluator.evaluate_run(run, None)
            client.create_feedback(run.id, eval_result.key, score=eval_result.score, comment=eval_result.feedback)

        print("\n🔍 LangSmith Evaluation Complete")


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
