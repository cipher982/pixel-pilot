"""Test UI-TARS integration using remote vLLM deployment."""

import base64
import logging

import pytest
from openai import OpenAI
from PIL import Image

from tests.test_uitars import convert_model_coords_to_absolute

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestRemoteUITARS:
    """Test UI-TARS functionality using remote vLLM deployment."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create OpenAI client configured for vLLM."""
        return OpenAI(
            base_url="http://jelly:8000/v1",  # Update with your actual server
            api_key="empty",  # vLLM doesn't need a real key
        )

    def test_coordinate_prediction(self, client, page_screenshot, caplog):
        """Test basic coordinate prediction from screenshot."""
        caplog.set_level(logging.INFO)

        # Get screenshot and actual coordinates
        screenshot_path = page_screenshot["screenshot_path"]
        actual_coords = page_screenshot["button"]
        logger.info(f"Actual button coordinates: {actual_coords}")

        if not screenshot_path.exists():
            pytest.skip(f"Test screenshot not found at {screenshot_path}")

        try:
            # Log original image details
            with Image.open(screenshot_path) as img:
                img_width, img_height = img.size
                logger.info(f"Original image size: {img.size}, mode: {img.mode}")
                # Calculate button center in original space
                button_center_x = actual_coords["x"] + actual_coords["width"] / 2
                button_center_y = actual_coords["y"] + actual_coords["height"] / 2
                logger.info(f"Button center in original space: ({button_center_x}, {button_center_y})")

            # Create chat completion request
            with open(screenshot_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

            response = client.chat.completions.create(
                model="bytedance-research/UI-TARS-7B-DPO",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Output only the coordinate of one point in your response. "
                                    "What element matches the following task: click the submit button."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                        ],
                    }
                ],
                max_tokens=50,
            )

            # Get response
            model_response = response.choices[0].message.content
            logger.info(f"Model response: {model_response}")

            # Validate response format
            assert model_response is not None, "Should get a response from the model"
            assert model_response.startswith("(") and model_response.endswith(
                ")"
            ), f"Response should be in coordinate format, got: {model_response}"

            # Parse coordinates
            try:
                coords = model_response.strip("()").split(",")
                model_x, model_y = map(int, coords)
                logger.info(f"Model output (normalized coords): ({model_x}, {model_y})")

                # Convert normalized coordinates to absolute pixels
                abs_x, abs_y = convert_model_coords_to_absolute(model_x, model_y, img_width, img_height)
                logger.info(f"Converted to absolute coords: ({abs_x}, {abs_y})")

                # Calculate distance from predicted point to button center
                center_distance = ((abs_x - button_center_x) ** 2 + (abs_y - button_center_y) ** 2) ** 0.5
                logger.info(f"Distance from button center: {center_distance:.2f} pixels")

                # Check if predicted coordinates are within reasonable distance of button center
                MAX_CENTER_DISTANCE = 20  # pixels
                assert center_distance <= MAX_CENTER_DISTANCE, (
                    f"Predicted point ({abs_x}, {abs_y}) too far from button center "
                    f"({button_center_x}, {button_center_y}). Distance: {center_distance:.2f} pixels"
                )

                logger.info("✓ Coordinates successfully validated!")

            except (ValueError, AssertionError) as e:
                pytest.fail(f"Failed to validate coordinates from response '{model_response}': {e}")

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            pytest.fail(f"Test failed: {e}")
