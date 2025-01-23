"""Manages test case datasets and their loading/saving."""

import json
import os
from typing import List
from typing import Optional

from .test_case import TestCase


class DatasetManager:
    """Manages loading and saving of test cases."""

    def __init__(self, test_dir: str = "eval/test_cases"):
        self.test_dir = test_dir

    def load_test_cases(self) -> List[TestCase]:
        """Load all test cases from the test directory."""
        test_cases = []
        for filename in os.listdir(self.test_dir):
            if filename.endswith(".json"):
                test_case = self.load_test_case(os.path.join(self.test_dir, filename))
                if test_case:
                    test_cases.append(test_case)
        return test_cases

    def load_test_case(self, path: str) -> Optional[TestCase]:
        """Load a single test case from a JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
                return TestCase.from_json(data)
        except Exception as e:
            print(f"Error loading test case {path}: {str(e)}")
            return None

    def save_test_case(self, test_case: TestCase, filename: str) -> bool:
        """Save a test case to a JSON file."""
        try:
            path = os.path.join(self.test_dir, filename)
            with open(path, "w") as f:
                json.dump(test_case.to_json(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving test case {filename}: {str(e)}")
            return False

    def create_dataset(self, name: str, test_cases: List[TestCase]) -> bool:
        """Create a new dataset directory with test cases."""
        dataset_dir = os.path.join(self.test_dir, name)
        try:
            os.makedirs(dataset_dir, exist_ok=True)
            for i, test_case in enumerate(test_cases):
                filename = f"test_{i+1}.json"
                self.save_test_case(test_case, os.path.join(name, filename))
            return True
        except Exception as e:
            print(f"Error creating dataset {name}: {str(e)}")
            return False
