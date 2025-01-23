"""Quick test to verify new test case format."""

from eval.datasets import DatasetManager


def test_load_new_format():
    """Verify we can load the new test case format."""
    manager = DatasetManager("eval/test_cases/terminal")
    test_cases = manager.load_test_cases()

    assert len(test_cases) > 0, "Should find at least one test case"

    # Check our create_file test specifically
    create_file_test = next((t for t in test_cases if t.task == "create a file called test.txt"), None)
    assert create_file_test is not None, "Should find create_file test"
    assert create_file_test.test_type == "terminal"
    assert "test.txt" in create_file_test.expected_result["files"]
