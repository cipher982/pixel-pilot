"""Tests for dataset management functionality."""

import tempfile
from pathlib import Path

import pytest

from eval.datasets import DatasetManager
from eval.datasets import TestCase
from eval.datasets import TestResult


@pytest.fixture
def sample_test_case_data():
    """Sample test case data for testing."""
    return {
        "task": "create a file called test.txt",
        "expected_result": {"success": True, "files": {"test.txt": {"exists": True}}},
        "test_type": "terminal",
        "expected_trajectory": ["check_path", "create_file"],
        "metadata": {"priority": "high"},
    }


@pytest.fixture
def sample_test_case(sample_test_case_data):
    """Create a sample TestCase instance."""
    return TestCase.from_json(sample_test_case_data)


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestTestCase:
    """Tests for TestCase class."""

    def test_create_test_case(self, sample_test_case_data):
        """Test creating a TestCase from dictionary."""
        test_case = TestCase.from_json(sample_test_case_data)
        assert test_case.task == sample_test_case_data["task"]
        assert test_case.expected_result == sample_test_case_data["expected_result"]
        assert test_case.test_type == sample_test_case_data["test_type"]
        assert test_case.expected_trajectory == sample_test_case_data["expected_trajectory"]
        assert test_case.metadata == sample_test_case_data["metadata"]

    def test_test_case_to_json(self, sample_test_case, sample_test_case_data):
        """Test converting TestCase to JSON."""
        json_data = sample_test_case.to_json()
        assert json_data == sample_test_case_data

    def test_default_values(self):
        """Test TestCase default values."""
        minimal_data = {"task": "simple task", "expected_result": {"success": True}}
        test_case = TestCase.from_json(minimal_data)
        assert test_case.test_type == "terminal"
        assert test_case.expected_trajectory is None
        assert test_case.metadata == {}


class TestTestResult:
    """Tests for TestResult class."""

    def test_create_test_result(self, sample_test_case):
        """Test creating a TestResult."""
        result = TestResult(
            test_case=sample_test_case,
            success=True,
            actual_result={"output": "file created"},
            trajectory=["check_path", "create_file"],
            metrics={"efficiency": 0.95},
        )
        assert result.success
        assert result.actual_result == {"output": "file created"}
        assert result.trajectory == ["check_path", "create_file"]
        assert result.metrics == {"efficiency": 0.95}

    def test_test_result_to_json(self, sample_test_case):
        """Test converting TestResult to JSON."""
        result = TestResult(
            test_case=sample_test_case,
            success=True,
            actual_result={"output": "file created"},
            trajectory=["check_path", "create_file"],
        )
        json_data = result.to_json()
        assert json_data["test"] == sample_test_case.to_json()
        assert json_data["success"]
        assert json_data["actual_result"] == {"output": "file created"}
        assert json_data["trajectory"] == ["check_path", "create_file"]


class TestDatasetManager:
    """Tests for DatasetManager class."""

    def test_save_and_load_test_case(self, sample_test_case, temp_test_dir):
        """Test saving and loading a test case."""
        manager = DatasetManager(temp_test_dir)

        # Save test case
        assert manager.save_test_case(sample_test_case, "test1.json")

        # Verify file exists
        test_file = Path(temp_test_dir) / "test1.json"
        assert test_file.exists()

        # Load and verify
        loaded_case = manager.load_test_case(str(test_file))
        assert loaded_case.task == sample_test_case.task
        assert loaded_case.expected_result == sample_test_case.expected_result

    def test_create_dataset(self, sample_test_case, temp_test_dir):
        """Test creating a dataset with multiple test cases."""
        manager = DatasetManager(temp_test_dir)
        test_cases = [sample_test_case] * 3  # Create 3 identical test cases

        # Create dataset
        assert manager.create_dataset("test_set", test_cases)

        # Verify dataset directory exists
        dataset_dir = Path(temp_test_dir) / "test_set"
        assert dataset_dir.exists()
        assert dataset_dir.is_dir()

        # Verify files were created
        files = list(dataset_dir.glob("*.json"))
        assert len(files) == 3

    def test_load_test_cases(self, sample_test_case, temp_test_dir):
        """Test loading all test cases from a directory."""
        manager = DatasetManager(temp_test_dir)

        # Create multiple test cases
        for i in range(3):
            manager.save_test_case(sample_test_case, f"test{i}.json")

        # Load all test cases
        loaded_cases = manager.load_test_cases()
        assert len(loaded_cases) == 3
        assert all(isinstance(case, TestCase) for case in loaded_cases)

    def test_handle_invalid_json(self, temp_test_dir):
        """Test handling invalid JSON files."""
        manager = DatasetManager(temp_test_dir)

        # Create invalid JSON file
        invalid_file = Path(temp_test_dir) / "invalid.json"
        invalid_file.write_text("invalid json content")

        # Should return None for invalid file
        assert manager.load_test_case(str(invalid_file)) is None

        # Should skip invalid files when loading all
        loaded_cases = manager.load_test_cases()
        assert len(loaded_cases) == 0
