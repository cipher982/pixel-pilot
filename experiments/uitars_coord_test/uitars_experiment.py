"""Experiment with UI-TARS coordinate predictions."""

import asyncio
import base64
import logging
from pathlib import Path
from typing import List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from PIL import Image
from playwright.sync_api import sync_playwright

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Setup paths
EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def convert_coords_to_absolute(norm_x: int, norm_y: int, img_width: int, img_height: int) -> tuple[int, int]:
    """Convert normalized coordinates to absolute pixel coordinates."""
    return int(norm_x * img_width / 1000), int(norm_y * img_height / 1000)


async def process_batch(
    client: AsyncOpenAI,
    batch_prompts: List[str],
    img_base64: str,
    temperature: float,
) -> List[ChatCompletion]:
    """Process a batch of prompts concurrently with small random delays."""
    tasks = []
    for prompt in batch_prompts:
        # Add a small random delay between 0-100ms
        await asyncio.sleep(np.random.random() * 0.1)
        tasks.append(
            client.chat.completions.create(
                model="bytedance-research/UI-TARS-7B-DPO",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                        ],
                    }
                ],
                max_tokens=50,
                temperature=temperature,
            )
        )
    return await asyncio.gather(*tasks)


async def run_experiment(
    image_path: str | Path,
    button_coords: dict,
    n_samples: int = 50,
    temperature: float = 0.7,
    outlier_threshold: int = 100,
    batch_size: int = 5,
) -> None:
    """Run coordinate prediction experiment with batched requests."""
    # Setup client
    client = AsyncOpenAI(base_url="http://jelly:8000/v1", api_key="empty")

    # Load and encode image
    img = Image.open(image_path)
    img_width, img_height = img.size
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Store predictions and outliers
    predictions = []
    outliers = []

    base_prompt = """Output only the coordinate of one point in your response.
    What element matches the following task: click the submit button.
    """

    # Create batches
    batches = []
    for i in range(0, n_samples, batch_size):
        batch = [base_prompt] * min(batch_size, n_samples - i)
        batches.append(batch)

    # Process batches
    for batch_idx, batch_prompts in enumerate(batches):
        try:
            responses = await process_batch(client, batch_prompts, img_base64, temperature)

            for i, response in enumerate(responses):
                try:
                    model_response = response.choices[0].message.content
                    if not model_response:
                        logger.error(f"Empty response from model on prediction {batch_idx * batch_size + i + 1}")
                        continue

                    coords = model_response.strip("()").split(",")
                    model_x, model_y = map(int, coords)
                    abs_x, abs_y = convert_coords_to_absolute(model_x, model_y, img_width, img_height)
                    predictions.append((abs_x, abs_y))

                    # Check for outliers
                    button_center_x = button_coords["x"] + button_coords["width"] / 2
                    button_center_y = button_coords["y"] + button_coords["height"] / 2
                    distance = ((abs_x - button_center_x) ** 2 + (abs_y - button_center_y) ** 2) ** 0.5

                    logger.info(
                        f"Prediction {batch_idx * batch_size + i + 1}: ({abs_x}, {abs_y}), distance: {distance:.2f}px"
                    )

                    if distance > outlier_threshold:
                        outliers.append(
                            {
                                "coords": (abs_x, abs_y),
                                "distance": distance,
                                "prompt": batch_prompts[i],
                                "response": model_response,
                            }
                        )
                        logger.warning(f"Found outlier! Distance: {distance:.2f}px")

                except Exception as e:
                    logger.error(f"Error processing response {batch_idx * batch_size + i + 1}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}: {e}")
            continue

    # Create visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    # Plot button
    plt.gca().add_patch(
        patches.Rectangle(
            (button_coords["x"], button_coords["y"]),
            button_coords["width"],
            button_coords["height"],
            fill=False,
            color="green",
            linewidth=2,
            label="Actual Button",
        )
    )

    if predictions:
        # Plot predictions
        x_coords, y_coords = zip(*predictions)
        plt.scatter(x_coords, y_coords, c="red", marker="x", s=100, label="Predictions")

        # Plot mean
        mean_x, mean_y = np.mean(x_coords), np.mean(y_coords)
        plt.scatter([mean_x], [mean_y], c="blue", marker="o", s=100, label="Mean Prediction")

    plt.title(f"UI-TARS Predictions (n={len(predictions)})")
    plt.legend()

    # Save plot
    plt.savefig(RESULTS_DIR / "predictions.png")
    plt.close()

    # Log statistics
    if predictions:
        distances = [
            (
                (x - (button_coords["x"] + button_coords["width"] / 2)) ** 2
                + (y - (button_coords["y"] + button_coords["height"] / 2)) ** 2
            )
            ** 0.5
            for x, y in predictions
        ]

        logger.info("\nExperiment Results:")
        logger.info(f"Total predictions: {len(predictions)}")
        logger.info(f"Outliers: {len(outliers)}")
        logger.info(f"Mean distance: {np.mean(distances):.2f}px")
        logger.info(f"Std dev: {np.std(distances):.2f}px")
        logger.info(f"Min/Max distance: {min(distances):.2f}px / {max(distances):.2f}px")
    else:
        logger.warning("No valid predictions to analyze")


def setup_test_page() -> tuple[Path, dict]:
    """Create test page and get screenshot with button coordinates."""
    screenshot_path = RESULTS_DIR / "test_screenshot.png"

    # Setup Playwright and take screenshot
    with sync_playwright() as p:
        # Launch browser with fixed viewport
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})

        # Load the test page
        page.goto(f"file://{RESULTS_DIR.absolute()}/realistic_test_page.html")

        # Get button coordinates
        button = page.locator("#submit-btn")
        box = button.bounding_box()

        if not box:
            raise ValueError("Failed to get submit button bounding box")

        # Take screenshot
        page.screenshot(path=str(screenshot_path))

        # Clean up
        browser.close()

        button_coords = {
            "x": int(box["x"]),
            "y": int(box["y"]),
            "width": int(box["width"]),
            "height": int(box["height"]),
        }

        logger.info(f"Created test page screenshot at {screenshot_path}")
        logger.info(f"Submit button coordinates: {button_coords}")

        return screenshot_path, button_coords


def main():
    """Run the experiment with test page screenshot."""
    # First run the sync setup
    image_path, button_coords = setup_test_page()

    # Then run the async experiment
    asyncio.run(
        run_experiment(
            image_path=image_path,
            button_coords=button_coords,
            n_samples=50,
            temperature=0.7,
            outlier_threshold=100,
            batch_size=5,  # Process 5 requests concurrently
        )
    )


if __name__ == "__main__":
    main()
