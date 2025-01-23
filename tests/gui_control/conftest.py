"""Test fixtures for GUI control testing."""

import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Generator
from typing import Optional

import pytest


def find_x11_tool(cmd: str) -> Optional[str]:
    """Find X11 tool, checking standard locations."""
    # Standard PATH check
    path = shutil.which(cmd)
    if path:
        return path

    # macOS XQuartz location
    if platform.system() == "Darwin":
        xquartz_path = f"/opt/X11/bin/{cmd}"
        if os.path.exists(xquartz_path):
            return xquartz_path

    # Linux X11 locations
    linux_paths = [f"/usr/bin/{cmd}", f"/usr/X11R6/bin/{cmd}", f"/usr/local/bin/{cmd}"]
    for path in linux_paths:
        if os.path.exists(path):
            return path

    return None


def check_x11_deps():
    """Check if X11 dependencies are installed."""
    missing = []
    paths = {}

    for cmd in ["xauth", "Xvfb", "xdpyinfo"]:
        path = find_x11_tool(cmd)
        if not path:
            missing.append(cmd)
        else:
            paths[cmd] = path

    if missing:
        if platform.system() == "Darwin":
            pytest.skip(
                f"Missing X11 dependencies: {', '.join(missing)}. "
                "Please install with: brew install xquartz xorg-server"
            )
        else:
            pytest.skip(f"Missing X11 dependencies: {', '.join(missing)}. " "Please install xauth, xvfb, and x11-utils")

    # Add XQuartz bin to PATH if needed
    if platform.system() == "Darwin" and "/opt/X11/bin" not in os.environ["PATH"]:
        os.environ["PATH"] = "/opt/X11/bin:" + os.environ["PATH"]

    return paths


@pytest.fixture(scope="session")
def x11_server() -> Generator[None, None, None]:
    """Start Xvfb server for tests."""
    # Check dependencies first
    x11_paths = check_x11_deps()

    # Set up X11 auth
    auth_file = "/tmp/.Xauthority"
    display_num = "99"
    display = f":{display_num}"

    # Clean up any existing files
    if os.path.exists(auth_file):
        os.remove(auth_file)

    # Create fresh auth file
    os.environ["XAUTHORITY"] = auth_file
    os.environ["DISPLAY"] = display

    # Start Xvfb first
    xvfb_proc = subprocess.Popen(
        [x11_paths.get("Xvfb", "Xvfb"), display, "-screen", "0", "1024x768x24", "-ac"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for X server to start
    time.sleep(1)

    try:
        # Now set up auth
        subprocess.run(["touch", auth_file], check=True)

        # Create auth with proper permissions
        os.chmod(auth_file, 0o600)  # Only owner can read/write
        subprocess.run(
            [x11_paths.get("xauth", "xauth"), "add", display, "MIT-MAGIC-COOKIE-1", "dead00beef00"],
            check=True,
            env={"XAUTHORITY": auth_file},
        )

        # Verify X server is running
        try:
            subprocess.run([x11_paths.get("xdpyinfo", "xdpyinfo")], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"xdpyinfo output: {e.stdout}\nerror: {e.stderr}", file=sys.stderr)
            raise

        yield
    finally:
        xvfb_proc.terminate()
        xvfb_proc.wait()
        # Cleanup
        if os.path.exists(auth_file):
            os.remove(auth_file)


@pytest.fixture
def gui_controller():
    """Create GUI controller for testing."""
    from pixelpilot.gui_control import GUIControllerFactory

    # Force eval mode for testing
    os.environ["PIXELPILOT_EVAL"] = "1"

    controller = GUIControllerFactory.create(mode="eval")
    try:
        result = controller.initialize()
        assert result.success, f"Controller initialization failed: {result.message}"
        yield controller
    finally:
        controller.cleanup()
