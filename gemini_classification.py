import os
from dotenv import load_dotenv
from google import genai


load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=GEMINI_API_KEY)

from google import genai
from PIL import Image
import io
import json
import re
import base64
import os

# -----------------------------
# 1. Setup client and labels
# -----------------------------
# client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

def load_imagenet_labels():
    """Load the canonical 1k ImageNet labels from torchvision metadata."""
    try:
        from torchvision.models import ResNet50_Weights
        labels = ResNet50_Weights.DEFAULT.meta.get("categories", [])
    except Exception as exc:
        raise RuntimeError("Install torchvision to load ImageNet labels dynamically.") from exc
    if len(labels) != 1000:
        raise RuntimeError(f"Expected 1000 ImageNet labels, got {len(labels)}")
    return labels

IMAGENET_LABELS = load_imagenet_labels()

def _strip_code_fences(text: str) -> str:
    """Remove ``` fences that Gemini sometimes wraps around JSON."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _sanitize_json_like(text: str) -> str:
    """Best-effort cleanup for minor JSON mistakes (dangling commas, lonely comma lines)."""
    cleaned = text
    cleaned = re.sub(r"\n(\s*),\s*(?=\n)", lambda m: f"\n{m.group(1)}}},", cleaned)
    cleaned = re.sub(r",(\s*[\]}])", r"\1", cleaned)
    return cleaned.strip()

def load_image_bytes(path: str) -> bytes:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

# -----------------------------
# 2. Classification function
# -----------------------------
def classify_imagenet(image_path: str, top_k: int = 1):
    img_bytes = load_image_bytes(image_path)

    # We’ll give Gemini the label list and force JSON output
    labels_text = ", ".join(IMAGENET_LABELS)

    system_prompt = f"""
You are an ImageNet image classifier.

You are given an image and the list of possible ImageNet classes.
You must choose the most likely classes.

Return a JSON object with:
    - "predictions": a list of objects, each with:
    - "label": one label from the list
    - "confidence": a float between 0 and 1

Use only labels from this list:
[{labels_text}]
"""

    user_prompt = f"""
Classify this image into ImageNet classes.
Return exactly {top_k} predictions ordered from most to least likely.
Respond with **only** valid JSON, no extra text.
"""
    print("Sent the prompt to model")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": system_prompt + "\n\n" + user_prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_bytes,
                        }
                    },
                ],
            }
        ],
    )
    print("Got the response")
    

    # Gemini response text should be pure JSON; strip code fences and clean common mistakes.
    text = _strip_code_fences(response.text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        sanitized = _sanitize_json_like(text)
        try:
            data = json.loads(sanitized)
        except json.JSONDecodeError:
            raise ValueError(f"Model did not return valid JSON: {text}")

    # return data["predictions"]
    return response.text

# -----------------------------
# 3. Example usage
# -----------------------------
if __name__ == "__main__":
    preds = classify_imagenet("tiny-imagenet-200/val/images/val_5.JPEG", top_k=5)
    print(preds)
    # for p in preds:
    #     print(f"{p['label']}: {p['confidence']:.3f}")
