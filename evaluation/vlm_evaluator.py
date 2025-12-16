import re
import base64
import os
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
                f"Coordinate system: The top-left corner is (0, 0) and ({width}, {height}) is the bottom-right. \n"
                f"Respond with only the coordinates in the format: (x, y)"
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
            completion_kwargs = {"max_tokens": 100}
            
            if "openrouter" in self.model.lower():
                # Get OpenRouter configuration (matching test_glm_comparison.py)
                openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
                openrouter_base_url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
                
                if not openrouter_api_key:
                    return None, Exception("OPENROUTER_API_KEY not set"), metadata
                
                # Explicitly set environment variables for OpenRouter (required by LiteLLM)
                os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
                os.environ["OPENROUTER_API_BASE"] = openrouter_base_url
                base_url = openrouter_base_url
                # Don't set max_tokens for OpenRouter (let model decide, matching test file)
                completion_kwargs = {}
            
            response = litellm.completion(
                model=self.model,
                messages=messages,
                base_url=base_url,
                **completion_kwargs
            )

            if not response or not response.choices:
                return None, Exception("Empty response from model"), metadata
            
            content = response.choices[0].message.content
            
            if content is None or content.strip() == "":
                # Debug: Check response structure
                debug_info = f"Response object type: {type(response)}"
                if hasattr(response, 'model'):
                    debug_info += f", model: {response.model}"
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    choice = response.choices[0]
                    debug_info += f", choice type: {type(choice)}"
                    if hasattr(choice, 'message'):
                        msg = choice.message
                        debug_info += f", message type: {type(msg)}"
                        if hasattr(msg, 'content'):
                            debug_info += f", content type: {type(msg.content)}, content: {repr(msg.content)}"
                
                error_msg = f"Model returned empty response (model: {self.model}). {debug_info}"
                if "openrouter" in self.model.lower():
                    error_msg += " Check OpenRouter API key and model availability."
                return None, Exception(error_msg), metadata
            
            return content, None, metadata

        except Exception as e:
            error_msg = str(e)
            
            if "openrouter" in self.model.lower():
                if "API key" in error_msg or "authentication" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                    return None, Exception(f"OpenRouter API Error: Invalid or missing API key. Please check your OPENROUTER_API_KEY environment variable."), metadata
                elif "quota" in error_msg.lower() or "429" in error_msg:
                    return None, Exception(f"OpenRouter API Error: Quota exceeded. Please check your API usage limits."), metadata
                elif "not found" in error_msg.lower() or "404" in error_msg or "invalid model" in error_msg.lower():
                    return None, Exception(f"OpenRouter API Error: Model not found or invalid. Model: {self.model}"), metadata
                else:
                    return None, Exception(f"OpenRouter API Error: {error_msg}"), metadata
            
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

