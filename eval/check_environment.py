#!/usr/bin/env python3

import os
import sys


def check_display():
    print("Checking X11 display...")
    display = os.environ.get("DISPLAY")
    if not display:
        print("❌ DISPLAY environment variable not set")
        return False
    print(f"✓ DISPLAY is set to {display}")

    # Check if X server is running
    try:
        import Xlib.display

        display = Xlib.display.Display()
        screen = display.screen()
        print(f"✓ Connected to X server, screen dimensions: {screen.width_in_pixels}x{screen.height_in_pixels}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to X server: {str(e)}")
        return False


def check_gui_tools():
    print("\nChecking GUI automation tools...")
    try:
        import pyautogui

        print("✓ pyautogui imported successfully")

        # Test basic screen info
        try:
            size = pyautogui.size()
            print(f"✓ Screen size detected: {size.width}x{size.height}")
            return True
        except Exception as e:
            print(f"❌ Failed to get screen info: {str(e)}")
            return False

    except Exception as e:
        print(f"❌ Failed to import pyautogui: {str(e)}")
        return False


def check_dependencies():
    print("\nChecking required packages...")
    required = ["Xlib", "pyautogui", "mouseinfo"]

    all_good = True
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError as e:
            print(f"❌ {package} is missing: {str(e)}")
            all_good = False
    return all_good


def main():
    print("🔍 Running environment checks...")

    checks = [check_display(), check_dependencies(), check_gui_tools()]

    if all(checks):
        print("\n✅ All checks passed!")
        return 0
    else:
        print("\n❌ Some checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
