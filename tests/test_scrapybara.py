"""Simple test script for Scrapybara functionality."""

import os
import time

from dotenv import load_dotenv
from scrapybara import Scrapybara

# Load environment variables
load_dotenv()


def main():
    """Test basic Scrapybara functionality."""
    print("Starting Scrapybara test...")

    # Initialize client
    api_key = os.getenv("SCRAPYBARA_API_KEY")
    if not api_key:
        raise ValueError("SCRAPYBARA_API_KEY environment variable not set")
    print(f"Using API key: {api_key[:8]}...")  # Only show first 8 chars

    client = Scrapybara(api_key=api_key)
    print("Client initialized")
    instance = None

    try:
        # Start VM
        print("\nStarting Ubuntu VM...")
        instance = client.start_ubuntu()
        print("VM started successfully")
        time.sleep(2)  # Give VM time to fully initialize

        # Test basic commands
        print("\nTesting basic commands:")

        print("\n1. pwd")
        result = instance.bash(command="pwd")
        print(f"Result: {result}")
        print(f"Result type: {type(result)}")
        time.sleep(1)  # Wait between commands

        print("\n2. ls")
        result = instance.bash(command="ls")
        print(f"Result: {result}")
        print(f"Result type: {type(result)}")
        time.sleep(1)  # Wait between commands

        print("\n3. echo 'hello world'")
        result = instance.bash(command="echo 'hello world'")
        print(f"Result: {result}")
        print(f"Result type: {type(result)}")
        time.sleep(1)  # Wait between commands

        # Test if VM is still responsive
        print("\nChecking VM responsiveness...")
        result = instance.bash(command="date")
        print(f"Current VM time: {result.get('output') if isinstance(result, dict) else result}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error attributes: {dir(e)}")
        raise
    finally:
        if instance:
            try:
                # Stop VM
                print("\nStopping VM...")
                instance.stop()
                print("VM stopped successfully")
                time.sleep(2)  # Give time for cleanup
            except Exception as e:
                print(f"Error stopping VM: {e}")


if __name__ == "__main__":
    main()
