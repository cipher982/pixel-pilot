import unittest

import torch


class TestTorchDevice(unittest.TestCase):
    def test_get_device(self):
        """Test if we can get a valid PyTorch device."""
        device = self.get_device()
        self.assertIn(device, ["cuda", "mps", "cpu"])

        # Test that the device can be used with torch
        torch_device = torch.device(device)
        self.assertIsInstance(torch_device, torch.device)

    def test_device_availability(self):
        """Test the device availability checks."""
        if torch.cuda.is_available():
            self.assertEqual(self.get_device(), "cuda")
        elif torch.backends.mps.is_available():
            self.assertEqual(self.get_device(), "mps")
        else:
            self.assertEqual(self.get_device(), "cpu")

    @staticmethod
    def get_device():
        """Get the best available device for PyTorch."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"


if __name__ == "__main__":
    unittest.main()
