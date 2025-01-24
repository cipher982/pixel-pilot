"""Dataset management utilities."""

import json
import os
from typing import List
from typing import Optional

from .test_case import EvalCase


class DatasetManager:
    """Manages test case datasets."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def load_test_cases(self) -> List[EvalCase]:
        """Load all test cases from the dataset directory."""
        test_cases = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".json"):
                    path = os.path.join(root, file)
                    if test_case := self.load_test_case(path):
                        test_cases.append(test_case)
        return test_cases

    def load_test_case(self, path: str) -> Optional[EvalCase]:
        """Load a single test case from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EvalCase.from_json(data)
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return None

    def save_test_case(self, test_case: EvalCase, filename: str) -> bool:
        """Save a test case to a JSON file."""
        try:
            path = os.path.join(self.base_dir, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)  # Create directories if needed
            with open(path, "w") as f:
                json.dump(test_case.to_json(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving test case {filename}: {str(e)}")
            return False

    def create_dataset(self, name: str, test_cases: List[EvalCase]) -> bool:
        """Create a new dataset directory with test cases."""
        dataset_dir = os.path.join(self.base_dir, name)
        try:
            os.makedirs(dataset_dir, exist_ok=True)
            for i, test_case in enumerate(test_cases):
                filename = f"test_{i+1}.json"
                self.save_test_case(test_case, os.path.join(name, filename))
            return True
        except Exception as e:
            print(f"Error creating dataset {name}: {str(e)}")
            return False
