import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables from .env
load_dotenv()

# Configuration
BASE_URL = "https://api.fireworks.ai/inference/v1"
TEXT_MODEL = "accounts/fireworks/models/deepseek-v3"
VISION_MODEL = "accounts/fireworks/models/phi-3-vision-128k-instruct"
API_KEY = os.getenv("FIREWORKS_API_KEY")

if not API_KEY:
    raise ValueError("FIREWORKS_API_KEY environment variable not set")


class ActionResponse(BaseModel):
    action_type: str
    reason: str


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def test_text_only():
    """Test simple text completion"""
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    print("\nTesting text-only completion...")
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}],
    )
    print(f"Response: {response.choices[0].message.content}")


def test_with_image(image_path):
    """Test completion with image"""
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    # Encode image
    base64_image = encode_image(image_path)

    print(f"\nTesting completion with image from {image_path}...")
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ],
            }
        ],
    )
    print(f"Response: {response.choices[0].message.content}")


if __name__ == "__main__":
    # Test text-only first
    test_text_only()

    # Test with image if path provided
    image_path = input("\nEnter path to test image (or press Enter to skip): ").strip()
    if image_path and Path(image_path).exists():
        test_with_image(image_path)
    else:
        print("No valid image path provided, skipping image test")
