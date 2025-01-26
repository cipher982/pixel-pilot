"""Tests for the evaluation runner."""

import json
from unittest.mock import mock_open
from unittest.mock import patch

import pytest

from eval.datasets import EvalCase
from eval.runner import run_eval
from eval.runner import run_gui_test
from eval.runner import run_terminal_test


@pytest.fixture
def terminal_test_case():
    """Create a sample terminal test case."""
    return EvalCase.from_json(
        {
            "task": "create a file called test.txt",
            "expected_result": {"success": True, "files": {"test.txt": {"exists": True}}},
            "test_type": "terminal",
            "expected_trajectory": ["check_path", "create_file"],
        }
    )


@pytest.fixture
def gui_test_case():
    """Create a sample GUI test case."""
    return EvalCase.from_json(
        {
            "task": "click the submit button",
            "expected_result": {
                "success": True,
                "interactions": {"click": {"element_id": "submit-button", "performed": True}},
            },
            "test_type": "gui",
            "expected_trajectory": ["locate_window", "find_element", "perform_click"],
            "metadata": {
                "window_info": {"title": "Quiz Application", "element": {"id": "submit-button", "type": "button"}}
            },
        }
    )


@pytest.fixture
def mock_successful_result():
    """Mock a successful test result."""
    return {"task_result": {"success": True, "output": "Operation completed"}, "trajectory": ["step1", "step2"]}


def test_run_terminal_test(terminal_test_case, mock_successful_result):
    """Test running a terminal test."""
    # Mock subprocess.run and file reading
    with (
        patch("subprocess.run") as mock_run,
        patch("builtins.open", mock_open(read_data=json.dumps(mock_successful_result))),
    ):
        # Configure mock subprocess return value
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Command output"
        mock_run.return_value.stderr = ""

        result = run_terminal_test(terminal_test_case)

        # Verify subprocess was called correctly
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "--instructions" in args
        assert terminal_test_case.task in args

        # Verify result
        assert result.success
        assert result.test_case == terminal_test_case
        assert len(result.trajectory) > 0


def test_run_gui_test(gui_test_case, mock_successful_result):
    """Test running a GUI test."""
    # Mock subprocess.run and file reading
    with (
        patch("subprocess.run") as mock_run,
        patch("builtins.open", mock_open(read_data=json.dumps(mock_successful_result))),
    ):
        # Configure mock subprocess return value
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Command output"
        mock_run.return_value.stderr = ""

        result = run_gui_test(gui_test_case)

        # Verify subprocess was called with GUI flags
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "--gui-mode" in args
        assert "--window-info" in args

        # Verify result
        assert result.success
        assert result.test_case == gui_test_case
        assert len(result.trajectory) > 0


def test_run_eval_handles_errors(terminal_test_case):
    """Test error handling in run_eval."""
    # Mock subprocess to raise an error
    with patch("subprocess.run", side_effect=Exception("Test error")):
        result = run_eval(terminal_test_case)

        # Verify error handling
        assert not result.success
        assert result.error == "Test error"
        assert isinstance(result.actual_result, dict)
        assert "error" in result.actual_result
