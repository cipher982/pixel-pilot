"""Quick test to verify new test case format."""

from eval.datasets import DatasetManager


def test_load_new_format():
    """Verify we can load the new test case format."""
    # Now use root directory since we support recursive loading
    manager = DatasetManager("eval/test_cases")
    test_cases = manager.load_test_cases()

    assert len(test_cases) >= 2, "Should find at least two test cases"

    # Check terminal test
    create_file_test = next((t for t in test_cases if t.task == "create a file called test.txt"), None)
    assert create_file_test is not None, "Should find create_file test"
    assert create_file_test.test_type == "terminal"
    assert "test.txt" in create_file_test.expected_result["files"]

    # Check GUI test
    quiz_test = next((t for t in test_cases if "click the submit button" in t.task), None)
    assert quiz_test is not None, "Should find quiz button test"
    assert quiz_test.test_type == "gui"
    assert "submit-button" in quiz_test.metadata["window_info"]["element"]["id"]
