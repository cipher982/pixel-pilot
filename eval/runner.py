"""Test runner for evaluation."""

import json
import os
import subprocess
import time
from typing import Dict
from typing import List
from typing import Optional

from PIL import Image

from eval.datasets import EvalCase
from eval.datasets import EvalResult
from eval.datasets.manager import DatasetManager
from eval.verification import VerificationEngine
from pixelpilot.system_control import SystemController
from pixelpilot.system_control_factory import SystemControllerFactory


def collect_state(
    output: Dict,
    result: subprocess.CompletedProcess,
    final_screenshot: Optional[Image.Image] = None,
    controller: Optional[SystemController] = None,
) -> Dict:
    """Collect state information from test execution."""
    state = {
        "terminal": {"output": result.stdout, "error": result.stderr, "return_code": result.returncode},
        "files": {},  # Will be populated by verifier
        "screen_state": {**output.get("screen_state", {}), "final_screenshot": final_screenshot},
        "controller": controller,  # Add controller to state for verification
    }
    return state


def run_terminal_test(test_case: EvalCase) -> EvalResult:
    """Run a terminal-based test case."""
    try:
        print(f"Starting terminal test: {test_case.task}")
        print(f"Test type: {test_case.test_type}")
        print(f"Test metadata: {test_case.metadata}")

        print(f"Running command in directory: {os.getcwd()}")
        start_time = time.time()

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
            check=False,  # Don't raise on non-zero exit
            env=os.environ.copy(),
            capture_output=True,  # Capture output for debugging
            timeout=300,  # 5 min timeout
        )

        elapsed = time.time() - start_time
        print(f"Command completed in {elapsed:.2f}s")
        print(f"Return code: {result.returncode}")
        print("Command output:")
        print(result.stdout)
        if result.stderr:
            print("Command errors:")
            print(result.stderr)

        if result.returncode != 0:
            print("Command failed with non-zero exit code")
            return EvalResult(
                test_case=test_case,
                success=False,
                verification_results=[],
                actions=[],
                error=f"Exit {result.returncode}",
            )

        # Create artifacts dir with proper permissions
        print("Creating artifacts directory")
        os.makedirs("eval/artifacts", exist_ok=True)
        os.chmod("eval/artifacts", 0o777)  # Ensure writable by all users

        # Read result and verify
        try:
            print("Reading eval_result.json")
            with open("eval/artifacts/eval_result.json") as f:
                output = json.load(f)

            # Create controller for verification
            controller = SystemControllerFactory.create()
            controller.setup()

            try:
                # Collect state for verification
                state = collect_state(output, result, controller=controller)

                # Run verifications
                engine = VerificationEngine()
                success, verification_results = engine.verify_all(test_case.verification_rules, state)

                # Create result with verification details
                return EvalResult(
                    test_case=test_case,
                    success=success,
                    verification_results=verification_results,
                    actions=output.get("actions", []),
                )
            finally:
                controller.cleanup()

        except FileNotFoundError:
            print("eval_result.json not found - checking current directory")
            files = os.listdir(".")
            print(f"Current directory contents: {files}")
            return EvalResult(
                test_case=test_case,
                success=False,
                verification_results=[],
                actions=[],
                error="Failed to create eval_result.json",
            )

    except subprocess.TimeoutExpired as e:
        print(f"Command timed out after {e.timeout} seconds")
        return EvalResult(
            test_case=test_case,
            success=False,
            verification_results=[],
            actions=[],
            error=f"Timeout after {e.timeout}s",
        )
    except Exception as e:
        print(f"Error details: {str(e)}")
        return EvalResult(
            test_case=test_case,
            success=False,
            verification_results=[],
            actions=[],
            error=str(e),
        )


def run_gui_test(test_case: EvalCase) -> EvalResult:
    """Run a GUI-based test case."""
    try:
        print(f"Starting GUI test: {test_case.task}")
        print(f"Test type: {test_case.test_type}")
        print(f"Test metadata: {test_case.metadata}")

        # Add window info to task if available
        window_info = test_case.metadata.get("window_info")
        window_info_arg = f"--window-info '{json.dumps(window_info)}'" if window_info else ""

        print(f"Running command in directory: {os.getcwd()}")
        start_time = time.time()

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
                "--gui-mode",
                window_info_arg,
                "--instructions",
                test_case.task,
            ],
            text=True,
            check=False,  # Don't raise on non-zero exit
            env=os.environ.copy(),
            capture_output=True,  # Capture output for debugging
            timeout=300,  # 5 min timeout
        )

        elapsed = time.time() - start_time
        print(f"Command completed in {elapsed:.2f}s")
        print(f"Return code: {result.returncode}")
        print("Command output:")
        print(result.stdout)
        if result.stderr:
            print("Command errors:")
            print(result.stderr)

        if result.returncode != 0:
            print("Command failed with non-zero exit code")
            return EvalResult(
                test_case=test_case,
                success=False,
                verification_results=[],
                actions=[],
                error=f"Exit {result.returncode}",
            )

        # Create artifacts dir with proper permissions
        print("Creating artifacts directory")
        os.makedirs("eval/artifacts", exist_ok=True)
        os.chmod("eval/artifacts", 0o777)  # Ensure writable by all users

        # Read result and verify
        try:
            print("Reading eval_result.json")
            with open("eval/artifacts/eval_result.json") as f:
                output = json.load(f)

            # Create controller for verification
            controller = SystemControllerFactory.create()
            controller.setup()

            try:
                # Take final screenshot
                print("Capturing final screenshot")
                screenshot, capture_result = controller.capture_screen()
                if not capture_result.success:
                    print(f"Screenshot capture failed: {capture_result.message}")
                    screenshot = None
                elif screenshot is not None:  # Only save if we have a screenshot
                    # Save screenshot to artifacts
                    screenshot_path = "eval/artifacts/final_screenshot.png"
                    screenshot.save(screenshot_path)
                    print(f"Screenshot saved to {screenshot_path}")

                # Collect state for verification
                state = collect_state(output, result, screenshot, controller)

                # Run verifications
                engine = VerificationEngine()
                success, verification_results = engine.verify_all(test_case.verification_rules, state)

                # Create result with verification details
                return EvalResult(
                    test_case=test_case,
                    success=success,
                    verification_results=verification_results,
                    actions=output.get("actions", []),
                )
            finally:
                controller.cleanup()

        except FileNotFoundError:
            print("eval_result.json not found - checking current directory")
            files = os.listdir(".")
            print(f"Current directory contents: {files}")
            return EvalResult(
                test_case=test_case,
                success=False,
                verification_results=[],
                actions=[],
                error="Failed to create eval_result.json",
            )

    except subprocess.TimeoutExpired as e:
        print(f"Command timed out after {e.timeout} seconds")
        return EvalResult(
            test_case=test_case,
            success=False,
            verification_results=[],
            actions=[],
            error=f"Timeout after {e.timeout}s",
        )
    except Exception as e:
        print(f"Error details: {str(e)}")
        return EvalResult(
            test_case=test_case,
            success=False,
            verification_results=[],
            actions=[],
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


def main():
    """Main entry point for eval runner."""
    print("\n🔍 Loading test cases...")
    manager = DatasetManager("eval/test_cases")
    test_cases = manager.load_test_cases()

    if not test_cases:
        print("❌ No test cases found!")
        print("\nSearched directories:")
        for root, dirs, files in os.walk("eval/test_cases"):
            print(f"  {root}/")
            for f in files:
                print(f"    - {f}")
        return

    print(f"\n📋 Found {len(test_cases)} test cases:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"  {i}. [{test_case.test_type}] {test_case.task}")

    print("\nRunning tests...")
    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n▶️  Test case {i}/{len(test_cases)}: {test_case.task}")
        result = run_eval(test_case)
        results.append(result)

        # Display result summary
        verification_summary = [f"{'✓' if v.passed else '✗'} {v.rule.description}" for v in result.verification_results]
        print(f"{'✅' if result.success else '❌'} Result:")
        if verification_summary:
            print("\n".join(f"  {line}" for line in verification_summary))
        if result.error:
            print(f"  Error: {result.error}")

    save_results(results)
    print("\n📊 Results Summary:")
    passed = sum(1 for r in results if r.success)
    print(f"Passed: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
