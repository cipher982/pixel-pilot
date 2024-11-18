import logging

from pixelpilot.action_system import ActionSystem

logging.basicConfig(level=logging.INFO)


def main():
    print("Testing parser...")

    # Initialize with debug mode
    action_system = ActionSystem()

    # Test with a sample screenshot
    test_image = "./examples/quiz.png"

    print("\nRunning parser test...")
    success = action_system.test_parser(test_image)

    if success:
        print("\nTest Results:")
        print("-" * 50)
        print("✓ Models loaded successfully")
        print("✓ Screenshot parsed")
        print("✓ OCR and icon detection completed")
    else:
        print("\n❌ Parser test failed!")


if __name__ == "__main__":
    main()
