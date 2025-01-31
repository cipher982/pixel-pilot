"""Test UI-TARS integration for coordinate prediction."""

import logging

import numpy as np
import pytest
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor

# Configure logging to be more visible in pytest
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def convert_model_coords_to_absolute(
    model_x: int, model_y: int, image_width: int, image_height: int
) -> tuple[int, int]:
    """Convert model's normalized coordinates (0-1000 range) to absolute pixel coordinates"""
    abs_x = round(image_width * model_x / 1000)
    abs_y = round(image_height * model_y / 1000)
    return abs_x, abs_y


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestUITARS:
    """Test UI-TARS functionality."""

    @pytest.fixture(scope="class")
    def model_and_processor(self):
        """Load UI-TARS model and processor."""
        try:
            model_name = "bytedance-research/UI-TARS-7B-DPO"
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            logger.info(f"Using device: {device}")

            logger.info(f"Loading model: {model_name}")
            model = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map=device
            )

            logger.info("Loading processor")
            processor = AutoProcessor.from_pretrained(model_name)

            return model, processor
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def test_coordinate_prediction(self, model_and_processor, page_screenshot, caplog):
        """Test basic coordinate prediction from screenshot."""
        caplog.set_level(logging.INFO)
        model, processor = model_and_processor

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

            # Create messages format with their recommended prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": str(screenshot_path),
                        },
                        {
                            "type": "text",
                            "text": (
                                "Output only the coordinate of one point in your response. "
                                "What element matches the following task: click the submit button"
                            ),
                        },
                    ],
                }
            ]

            # Prepare inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)[:2]  # Only take first two elements

            # Log image processing details
            if isinstance(image_inputs, list) and len(image_inputs) > 0:
                if isinstance(image_inputs[0], np.ndarray):
                    logger.info(f"Processed image shape: {image_inputs[0].shape}")
                    logger.info(f"Processed image dtype: {image_inputs[0].dtype}")
                    logger.info(f"Processed image range: [{image_inputs[0].min()}, {image_inputs[0].max()}]")

            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            # Log model input details
            logger.info("Input keys available: %s", inputs.keys())
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    logger.info(f"{k} shape: {v.shape}")
                    logger.info(f"{k} dtype: {v.dtype}")
                    logger.info(f"{k} range: [{v.min().item()}, {v.max().item()}]")

            # Generate prediction with optimizations
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,  # Deterministic
                    num_beams=1,  # No beam search
                    use_cache=True,  # Use KV cache
                )

            # Get input length for trimming
            input_length = inputs["input_ids"].shape[1]
            generated_ids_trimmed = outputs[:, input_length:]

            # Process response
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            logger.info(f"Model response: {response}")

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            pytest.fail(f"Test failed: {e}")

        # Validate response format
        assert response is not None, "Should get a response from the model"
        assert response.startswith("(") and response.endswith(
            ")"
        ), f"Response should be in coordinate format, got: {response}"

        # Parse coordinates
        try:
            coords = response.strip("()").split(",")
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
            pytest.fail(f"Failed to validate coordinates from response '{response}': {e}")

    def test_coordinate_conversion(self):
        """Verify coordinate conversion math"""
        test_cases = [
            # model_x, model_y, width, height, expected_x, expected_y
            (56, 301, 1280, 720, 72, 217),  # Our current case
            (500, 500, 1000, 1000, 500, 500),  # Middle of screen
            (0, 0, 1280, 720, 0, 0),  # Origin
        ]
        for model_x, model_y, w, h, exp_x, exp_y in test_cases:
            x, y = convert_model_coords_to_absolute(model_x, model_y, w, h)
            assert abs(x - exp_x) <= 1, f"X conversion failed for {(model_x, model_y)}"  # Allow 1px rounding difference
            assert abs(y - exp_y) <= 1, f"Y conversion failed for {(model_x, model_y)}"
