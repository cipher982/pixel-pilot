"""Tests for dataset management."""

import os
import tempfile
from typing import Dict

import pytest

from eval.datasets import DatasetManager
from eval.datasets import EvalCase
from eval.datasets import EvalResult


@pytest.fixture
def sample_test_case_data() -> Dict:
    """Sample test case data for testing."""
    return {
        "task": "List files in current directory",
        "expected_result": {"files": ["file1.txt", "file2.txt"]},
        "test_type": "terminal",
        "expected_trajectory": ["command executed", "output received"],
        "metadata": {"key": "value"},
    }


def create_test_case(data: Dict) -> EvalCase:
    """Create a sample EvalCase instance."""
    return EvalCase.from_json(data)


@pytest.fixture
def minimal_data() -> Dict:
    """Minimal test case data with only required fields."""
    return {
        "task": "Simple task",
        "expected_result": {"status": "success"},
    }


class TestEvalCase:
    """Tests for EvalCase class."""

    def test_create_from_json(self, sample_test_case_data):
        """Test creating an EvalCase from dictionary."""
        test_case = EvalCase.from_json(sample_test_case_data)
        assert test_case.task == sample_test_case_data["task"]
        assert test_case.expected_result == sample_test_case_data["expected_result"]
        assert test_case.test_type == sample_test_case_data["test_type"]
        assert test_case.expected_trajectory == sample_test_case_data["expected_trajectory"]
        assert test_case.metadata == sample_test_case_data["metadata"]

    def test_to_json(self, sample_test_case_data):
        """Test converting EvalCase to JSON."""
        test_case = create_test_case(sample_test_case_data)
        json_data = test_case.to_json()
        assert json_data == sample_test_case_data

    def test_default_values(self, minimal_data):
        """Test EvalCase default values."""
        test_case = EvalCase.from_json(minimal_data)
        assert test_case.test_type == "terminal"
        assert test_case.expected_trajectory is None
        assert test_case.metadata == {}


class TestEvalResult:
    """Tests for EvalResult class."""

    def test_create_result(self, sample_test_case_data):
        """Test creating an EvalResult."""
        result = EvalResult(
            test_case=create_test_case(sample_test_case_data),
            success=True,
            actual_result={"files": ["file1.txt", "file2.txt"]},
            trajectory=["step1", "step2"],
            error=None,
            metrics={"time": 1.0},
        )
        assert result.success
        assert result.actual_result == {"files": ["file1.txt", "file2.txt"]}
        assert result.trajectory == ["step1", "step2"]
        assert result.error is None
        assert result.metrics == {"time": 1.0}

    def test_to_json(self, sample_test_case_data):
        """Test converting EvalResult to JSON."""
        result = EvalResult(
            test_case=create_test_case(sample_test_case_data),
            success=True,
            actual_result={"files": ["file1.txt", "file2.txt"]},
            trajectory=["step1", "step2"],
        )
        json_data = result.to_json()
        assert json_data["test"] == sample_test_case_data
        assert json_data["success"]
        assert json_data["actual_result"] == {"files": ["file1.txt", "file2.txt"]}
        assert json_data["trajectory"] == ["step1", "step2"]


class TestDatasetManager:
    """Tests for DatasetManager class."""

    @pytest.fixture
    def test_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_save_and_load_test_case(self, test_dir, sample_test_case_data):
        """Test saving and loading a test case."""
        manager = DatasetManager(test_dir)
        test_case = create_test_case(sample_test_case_data)

        # Save test case
        assert manager.save_test_case(test_case, "test1.json")

        # Load test case
        loaded_case = manager.load_test_case(os.path.join(test_dir, "test1.json"))
        assert loaded_case is not None
        assert loaded_case.task == test_case.task
        assert loaded_case.expected_result == test_case.expected_result

    def test_create_dataset(self, test_dir, sample_test_case_data):
        """Test creating a dataset with multiple test cases."""
        manager = DatasetManager(test_dir)
        test_cases = [create_test_case(sample_test_case_data) for _ in range(3)]

        # Create dataset
        assert manager.create_dataset("test_dataset", test_cases)

        # Verify files were created
        dataset_dir = os.path.join(test_dir, "test_dataset")
        assert os.path.exists(dataset_dir)
        assert len(os.listdir(dataset_dir)) == 3

    def test_load_test_cases(self, test_dir, sample_test_case_data):
        """Test loading all test cases from a directory."""
        manager = DatasetManager(test_dir)

        # Create multiple test cases
        test_cases = [create_test_case(sample_test_case_data) for _ in range(3)]
        manager.create_dataset("dataset1", test_cases)

        # Load all test cases
        loaded_cases = manager.load_test_cases()
        assert len(loaded_cases) == 3
        assert all(isinstance(case, EvalCase) for case in loaded_cases)

    def test_handle_invalid_json(self, test_dir):
        """Test handling invalid JSON files."""
        _ = DatasetManager(test_dir)

        # Create invalid JSON file
        invalid_path = os.path.join(test_dir, "invalid.json")
        with open(invalid_path, "w") as f:
            f.write("invalid json content")
