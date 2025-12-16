import re
import base64
from io import BytesIO
from typing import Optional, Tuple, Dict, Any
import litellm
import numpy as np
from PIL import Image


class VLMEvaluator:
    """Evaluates VLMs using LiteLLM."""

    def __init__(self, model: str):
        self.model = model
        litellm.set_verbose = False

    def query_model(
        self, image: np.ndarray, prompt: str
    ) -> Tuple[Optional[str], Optional[Exception], Optional[Dict[str, Any]]]:
        """Query the VLM with image and prompt.
        
        Returns:
            Tuple of (response, error, metadata)
        """
        metadata = None
        try:
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            width, height = pil_image.size
            pixels = width * height
            metadata = {
                "size": (width, height),
                "estimated_tokens": int(pixels / 750),
            }

            prompt = (
                f"{prompt}\n\n"
                f"Image dimensions: {width} pixels wide × {height} pixels tall.\n"
                f"Respond with only the coordinates in the format: (x, y)."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self._encode_image(pil_image)}"
                            },
                        },
                    ],
                }
            ]

            base_url = None
            if "openrouter" in self.model.lower():
                base_url = "https://openrouter.ai/api/v1"
            
            response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=100,
                base_url=base_url,
            )

            content = response.choices[0].message.content
            return content, None, metadata

        except Exception as e:
            error_msg = str(e)
            
            if "gemini" in self.model.lower():
                if "API key" in error_msg or "INVALID_ARGUMENT" in error_msg:
                    return None, Exception(f"Gemini API Error: Invalid or missing API key. Please check your GEMINI_API_KEY environment variable."), metadata
                elif "QUOTA_EXCEEDED" in error_msg or "quota" in error_msg.lower():
                    return None, Exception(f"Gemini API Error: Quota exceeded. Please check your API usage limits."), metadata
                elif "SAFETY" in error_msg or "safety" in error_msg.lower():
                    return None, Exception(f"Gemini API Error: Content was blocked by safety filters."), metadata
                else:
                    return None, Exception(f"Gemini API Error: {error_msg}"), metadata
            
            return None, e, metadata

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    def extract_coordinates(self, response: str) -> Optional[Tuple[int, int]]:
        """Extract coordinates from model response."""
        if not response:
            return None

        patterns = [
            r"\((\d+),\s*(\d+)\)",
            r"\[(\d+),\s*(\d+)\]",
            r"(\d+),\s*(\d+)",
            r"x:\s*(\d+),\s*y:\s*(\d+)",
            r"x=(\d+),\s*y=(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    x = int(match.group(1))
                    y = int(match.group(2))
                    return (x, y)
                except (ValueError, IndexError):
                    continue

        return None

