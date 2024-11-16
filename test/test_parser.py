from autocomply.action_system import ActionSystem


def main():
    # Initialize with debug mode
    action_system = ActionSystem(debug=True)

    # Test with a sample image
    test_image = "examples/quiz.png"
    success = action_system.test_parser(test_image)

    if success:
        print("OmniParser integration test passed!")
    else:
        print("OmniParser integration test failed!")


if __name__ == "__main__":
    main()
