#!/usr/bin/env python3
"""Standalone script to evaluate VLM pixel pointing accuracy.

This script:
1. Generates synthetic test images with shapes/buttons
2. Creates metadata with prompts and ground truth coordinates
3. Uses LiteLLM to evaluate different VLMs
4. Calculates accuracy metrics (distance between predicted and actual)
"""

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# Load environment variables from .env file
load_dotenv()
# Model configurations

# Model configurations
MODELS = {
    "sonnet": "anthropic/claude-sonnet-4-20250514",
    "opus": "anthropic/claude-opus-4-5-20251101",
    "gemini3": "gemini/gemini-3-pro-preview",
    "chatgpt": "openai/gpt-5.1",
    "haiku": "anthropic/claude-haiku-4-5-20251001",
}
# Screen size configurations for testing
# Optimal sizes for Anthropic: max 1.15 MP, within 1568px on both dimensions
# These sizes avoid resizing and improve latency
SCREEN_SIZES = [
    {"name": "optimal_9_16", "width": 819, "height": 1456},  # 9:16 aspect (1.19 MP, optimal for portrait)
    {"name": "optimal_1_2", "width": 784, "height": 1568},  # 1:2 aspect (1.23 MP, max height)
    {"name": "optimal_2_3", "width": 896, "height": 1344},  # 2:3 aspect (1.20 MP)
    {"name": "optimal_3_4", "width": 951, "height": 1268},  # 3:4 aspect (1.21 MP)
    {"name": "optimal_1_1", "width": 1092, "height": 1092},  # 1:1 aspect (1.19 MP, square)
    {"name": "optimal_16_9", "width": 1456, "height": 819},  # 16:9 aspect (1.19 MP, landscape)
]

# Test image configurations
TEST_CONFIGS = [
    {
        "name": "simple_circle",
        "prompt": "Point to the center of the purple circle",
        "shape": "circle",
        "color": "purple",
        "position": "center",
    },
    {
        "name": "top_right_square",
        "prompt": "Point to the top right corner of the red square",
        "shape": "square",
        "color": "red",
        "position": "top_right",
    },
    {
        "name": "middle_x",
        "prompt": "Point to the middle of the X in the top right",
        "shape": "x",
        "color": "blue",
        "position": "top_right",
    },
    {
        "name": "transparent_button",
        "prompt": "Point to the center of the transparent button",
        "shape": "button",
        "color": "transparent",
        "position": "center",
    },
    {
        "name": "small_circle",
        "prompt": "Point to the center of the small green circle",
        "shape": "circle",
        "color": "green",
        "position": "center",
        "size": "small",
    },
    {
        "name": "bottom_left_triangle",
        "prompt": "Point to the center of the yellow triangle in the bottom left",
        "shape": "triangle",
        "color": "yellow",
        "position": "bottom_left",
    },
    {
        "name": "overlapping_shapes",
        "prompt": "Point to the center of the orange circle that is behind the blue square",
        "shape": "circle",
        "color": "orange",
        "position": "center",
        "overlap": True,
    },
    {
        "name": "low_contrast_button",
        "prompt": "Point to the center of the gray button on gray background",
        "shape": "button",
        "color": "lightgray",
        "position": "center",
        "background": "gray",
    },
]


@dataclass
class TestCase:
    """Represents a single test case with image and metadata."""

    name: str
    prompt: str
    image: np.ndarray
    expected_coords: Tuple[int, int]
    image_size: Tuple[int, int]
    metadata: Dict[str, Any]


class ImageGenerator:
    """Generates synthetic test images with shapes and buttons.
    
    Default dimensions are optimal for Anthropic: 819 x 1456 pixels (9:16 aspect, ~1.19 MP).
    This avoids resizing and improves latency for all models.
    """

    COLOR_MAP = {
        "purple": (128, 0, 128),
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "green": (0, 128, 0),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "gray": (128, 128, 128),
        "lightgray": (200, 200, 200),
        "transparent": (255, 255, 255, 128),  # Semi-transparent white
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }

    def __init__(
        self,
        width: int = 819,  # Optimal width (9:16 aspect)
        height: int = 1456,  # Optimal height (9:16 aspect)
        background_color: str = "white",
    ):
        self.width = width
        self.height = height
        self.bg_color = self.COLOR_MAP.get(background_color, (255, 255, 255))

    def generate_image(
        self, config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Generate an image based on test configuration.
        
        If 'expected_coords' is provided in config, uses those exact coordinates.
        Otherwise, calculates coordinates from 'position' description.
        """
        img = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img, "RGBA")

        shape = config.get("shape", "circle")
        color_name = config.get("color", "blue")
        position = config.get("position", "center")
        size = config.get("size", "medium")
        overlap = config.get("overlap", False)
        bg_color_name = config.get("background", "white")
        
        # Check if exact coordinates are provided
        expected_coords = config.get("expected_coords")
        if expected_coords:
            # Use provided coordinates directly
            if isinstance(expected_coords, list) and len(expected_coords) == 2:
                center_x, center_y = expected_coords[0], expected_coords[1]
            elif isinstance(expected_coords, tuple) and len(expected_coords) == 2:
                center_x, center_y = expected_coords[0], expected_coords[1]
            else:
                # Invalid format, fall back to position calculation
                expected_coords = None

        if bg_color_name != "white":
            img = Image.new("RGB", (self.width, self.height), self.COLOR_MAP[bg_color_name])
            draw = ImageDraw.Draw(img, "RGBA")

        # Determine size
        if size == "small":
            shape_size = min(self.width, self.height) // 8
        elif size == "large":
            shape_size = min(self.width, self.height) // 3
        else:  # medium
            shape_size = min(self.width, self.height) // 5

        # Determine position (only if not provided in expected_coords)
        if not expected_coords:
            if position == "center":
                center_x, center_y = self.width // 2, self.height // 2
            elif position == "top_right":
                center_x = int(self.width * 0.75)
                center_y = int(self.height * 0.25)
            elif position == "bottom_left":
                center_x = int(self.width * 0.25)
                center_y = int(self.height * 0.75)
            elif position == "top_left":
                center_x = int(self.width * 0.25)
                center_y = int(self.height * 0.25)
            elif position == "bottom_right":
                center_x = int(self.width * 0.75)
                center_y = int(self.height * 0.75)
            else:
                center_x, center_y = self.width // 2, self.height // 2

        # Handle overlapping shapes
        if overlap:
            # Draw a blue square first
            square_size = shape_size
            square_x = center_x - square_size // 2
            square_y = center_y - square_size // 2
            draw.rectangle(
                [square_x, square_y, square_x + square_size, square_y + square_size],
                fill=(0, 0, 255, 200),
            )
            # Draw orange circle behind (slightly offset)
            circle_offset = shape_size // 4
            circle_center_x = center_x - circle_offset
            circle_center_y = center_y - circle_offset

        # Draw the target shape
        color = self.COLOR_MAP.get(color_name, (0, 0, 255))
        if color_name == "transparent":
            fill_color = (*color[:3], 128)  # Semi-transparent
        else:
            fill_color = (*color, 255) if len(color) == 3 else color

        if shape == "circle":
            if overlap:
                target_center = (circle_center_x, circle_center_y)
            else:
                target_center = (center_x, center_y)
            bbox = [
                target_center[0] - shape_size // 2,
                target_center[1] - shape_size // 2,
                target_center[0] + shape_size // 2,
                target_center[1] + shape_size // 2,
            ]
            draw.ellipse(bbox, fill=fill_color)
            expected_coords = target_center

        elif shape == "square":
            bbox = [
                center_x - shape_size // 2,
                center_y - shape_size // 2,
                center_x + shape_size // 2,
                center_y + shape_size // 2,
            ]
            draw.rectangle(bbox, fill=fill_color)
            if position == "top_right":
                expected_coords = (bbox[2], bbox[1])  # Top right corner
            else:
                expected_coords = (center_x, center_y)

        elif shape == "triangle":
            points = [
                (center_x, center_y - shape_size // 2),  # Top
                (center_x - shape_size // 2, center_y + shape_size // 2),  # Bottom left
                (center_x + shape_size // 2, center_y + shape_size // 2),  # Bottom right
            ]
            draw.polygon(points, fill=fill_color)
            expected_coords = (center_x, center_y)

        elif shape == "x":
            line_width = max(3, shape_size // 10)
            half_size = shape_size // 2
            # Draw X
            draw.line(
                [
                    center_x - half_size,
                    center_y - half_size,
                    center_x + half_size,
                    center_y + half_size,
                ],
                fill=fill_color,
                width=line_width,
            )
            draw.line(
                [
                    center_x - half_size,
                    center_y + half_size,
                    center_x + half_size,
                    center_y - half_size,
                ],
                fill=fill_color,
                width=line_width,
            )
            expected_coords = (center_x, center_y)

        elif shape == "button":
            # Draw button with rounded rectangle
            button_width = shape_size * 2
            button_height = shape_size
            bbox = [
                center_x - button_width // 2,
                center_y - button_height // 2,
                center_x + button_width // 2,
                center_y + button_height // 2,
            ]
            if color_name == "transparent":
                # Draw with border only for transparent
                draw.rectangle(bbox, outline=(128, 128, 128, 200), width=3)
            else:
                draw.rectangle(bbox, fill=fill_color)
            expected_coords = (center_x, center_y)

        else:
            expected_coords = (center_x, center_y)

        # Ensure expected_coords is a tuple (convert from list if needed)
        if isinstance(expected_coords, list):
            expected_coords = tuple(expected_coords)
        elif not isinstance(expected_coords, tuple):
            # Fallback: use calculated center
            expected_coords = (center_x, center_y)

        # Convert to numpy array
        img_array = np.array(img)
        return img_array, expected_coords


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
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Calculate metadata (no resizing - images are already optimal size)
            width, height = pil_image.size
            pixels = width * height
            metadata = {
                "size": (width, height),
                "estimated_tokens": int(pixels / 750),
            }

            # Prepare message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}\n\nRespond with only the coordinates in the format: (x, y)"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self._encode_image(pil_image)}"
                            },
                        },
                    ],
                }
            ]

            response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=100,
            )

            content = response.choices[0].message.content
            return content, None, metadata

        except Exception as e:
            return None, e, metadata

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64."""
        import base64
        from io import BytesIO

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    def extract_coordinates(self, response: str) -> Optional[Tuple[int, int]]:
        """Extract coordinates from model response."""
        if not response:
            return None

        # Try various patterns to extract coordinates
        patterns = [
            r"\((\d+),\s*(\d+)\)",  # (x, y)
            r"\[(\d+),\s*(\d+)\]",  # [x, y]
            r"(\d+),\s*(\d+)",  # x, y
            r"x:\s*(\d+),\s*y:\s*(\d+)",  # x: 123, y: 456
            r"x=(\d+),\s*y=(\d+)",  # x=123, y=456
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


def calculate_distance(
    predicted: Tuple[int, int], expected: Tuple[int, int]
) -> float:
    """Calculate Euclidean distance between predicted and expected coordinates."""
    return math.sqrt(
        (predicted[0] - expected[0]) ** 2 + (predicted[1] - expected[1]) ** 2
    )


def calculate_accuracy_metrics(
    results: List[Dict[str, Any]], image_size: Tuple[int, int]
) -> Dict[str, Any]:
    """Calculate accuracy metrics from results."""
    distances = []
    successful_extractions = 0
    total_tests = len(results)

    for result in results:
        if result["predicted_coords"] is not None:
            successful_extractions += 1
            distance = result["distance"]
            distances.append(distance)

    if not distances:
        return {
            "total_tests": total_tests,
            "successful_extractions": 0,
            "extraction_rate": 0.0,
            "mean_distance": None,
            "median_distance": None,
            "max_distance": None,
            "min_distance": None,
            "accuracy_within_10px": 0.0,
            "accuracy_within_5_percent": 0.0,
        }

    # Calculate metrics
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    max_distance = np.max(distances)
    min_distance = np.min(distances)

    # Accuracy thresholds
    screen_diagonal = math.sqrt(image_size[0] ** 2 + image_size[1] ** 2)
    threshold_5_percent = screen_diagonal * 0.05

    accuracy_10px = sum(1 for d in distances if d <= 10) / len(distances) * 100
    accuracy_5_percent = (
        sum(1 for d in distances if d <= threshold_5_percent) / len(distances) * 100
    )

    return {
        "total_tests": total_tests,
        "successful_extractions": successful_extractions,
        "extraction_rate": (successful_extractions / total_tests) * 100,
        "mean_distance": mean_distance,
        "median_distance": median_distance,
        "max_distance": max_distance,
        "min_distance": min_distance,
        "accuracy_within_10px": accuracy_10px,
        "accuracy_within_5_percent": accuracy_5_percent,
    }


def run_evaluation(
    models: List[str],
    test_configs: List[Dict[str, Any]],
    output_dir: str,
    image_width: int = 1080,
    image_height: int = 2400,
    save_images: bool = True,
    test_multiple_sizes: bool = False,
) -> None:
    """Run evaluation for all models and test cases.
    
    Args:
        models: List of model names to evaluate
        test_configs: List of test configurations
        output_dir: Output directory for results
        image_width: Single test width (used if test_multiple_sizes=False)
        image_height: Single test height (used if test_multiple_sizes=False)
        save_images: Whether to save generated images
        test_multiple_sizes: If True, test all screen sizes in SCREEN_SIZES
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine which screen sizes to test
    if test_multiple_sizes:
        screen_sizes_to_test = SCREEN_SIZES
        print(f"\nTesting {len(screen_sizes_to_test)} different screen sizes:")
        for size in screen_sizes_to_test:
            print(f"  - {size['name']}: {size['width']}x{size['height']}")
    else:
        screen_sizes_to_test = [{"name": "custom", "width": image_width, "height": image_height}]

    all_size_results = {}

    for screen_config in screen_sizes_to_test:
        screen_name = screen_config["name"]
        screen_width = screen_config["width"]
        screen_height = screen_config["height"]
        
        print(f"\n{'='*60}")
        print(f"Testing screen size: {screen_name} ({screen_width}x{screen_height})")
        print(f"{'='*60}")
        
        # Create subdirectory for this screen size
        size_output_path = output_path / screen_name
        size_output_path.mkdir(parents=True, exist_ok=True)
        images_dir = size_output_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        generator = ImageGenerator(screen_width, screen_height)

        # Generate all images first and save them once
        test_images = {}
        if save_images:
            for config in test_configs:
                image, expected_coords = generator.generate_image(config)
                img_path = images_dir / f"{config['name']}.png"
                Image.fromarray(image).save(img_path)
                test_images[config['name']] = (image, expected_coords)

        all_results = {}
        consolidated_results = []

        for model_name in models:
            print(f"\n{'='*60}")
            print(f"Evaluating model: {model_name}")
            print(f"{'='*60}")

            if model_name not in MODELS:
                print(f"Warning: Unknown model {model_name}, skipping...")
                continue

            model_id = MODELS[model_name]
            evaluator = VLMEvaluator(model_id)

            model_results = []

            for i, config in enumerate(test_configs):
                print(f"\nTest {i+1}/{len(test_configs)}: {config['name']}")
                print(f"Prompt: {config['prompt']}")

                # Get pre-generated image
                if config['name'] in test_images:
                    image, expected_coords = test_images[config['name']]
                else:
                    image, expected_coords = generator.generate_image(config)

                # Query model
                print("Querying model...")
                response, error, metadata = evaluator.query_model(image, config["prompt"])
                
                if metadata:
                    print(f"Image size: {metadata['size'][0]}x{metadata['size'][1]} "
                          f"(~{metadata['estimated_tokens']} tokens)")

                if error:
                    print(f"Error: {error}")
                    result = {
                        "test_name": config["name"],
                        "prompt": config["prompt"],
                        "expected_coords": expected_coords,
                        "response": None,
                        "predicted_coords": None,
                        "distance": None,
                        "error": str(error),
                        "image_metadata": metadata,
                    }
                else:
                    print(f"Response: {response}")
                    predicted_coords = evaluator.extract_coordinates(response)

                    if predicted_coords:
                        distance = calculate_distance(predicted_coords, expected_coords)
                        print(
                            f"Expected: {expected_coords}, Predicted: {predicted_coords}, Distance: {distance:.2f}px"
                        )
                        result = {
                            "test_name": config["name"],
                            "prompt": config["prompt"],
                            "expected_coords": expected_coords,
                            "response": response,
                            "predicted_coords": predicted_coords,
                            "distance": distance,
                            "error": None,
                            "image_metadata": metadata,
                        }
                    else:
                        print("Could not extract coordinates from response")
                        result = {
                            "test_name": config["name"],
                            "prompt": config["prompt"],
                            "expected_coords": expected_coords,
                            "response": response,
                            "predicted_coords": None,
                            "distance": None,
                            "error": "Could not extract coordinates",
                            "image_metadata": metadata,
                        }

                model_results.append(result)

            # Calculate metrics
            metrics = calculate_accuracy_metrics(
                model_results, (screen_width, screen_height)
            )

            # Add to consolidated results
            for i, result in enumerate(model_results):
                test_name = result["test_name"]
                if i >= len(consolidated_results):
                    consolidated_results.append({
                        "test_name": test_name,
                        "prompt": result["prompt"],
                        "expected_coords": result["expected_coords"],
                        "image_file": f"images/{test_name}.png",
                        "models": {}
                    })
                consolidated_results[i]["models"][model_name] = {
                    "predicted_coords": result["predicted_coords"],
                    "distance": result["distance"],
                    "response": result["response"],
                    "error": result["error"]
                }

            print(f"\n{'='*60}")
            print(f"Results for {model_name}:")
            print(f"{'='*60}")
            print(f"Total tests: {metrics['total_tests']}")
            print(f"Successful extractions: {metrics['successful_extractions']}")
            print(f"Extraction rate: {metrics['extraction_rate']:.2f}%")
            if metrics["mean_distance"] is not None:
                print(f"Mean distance: {metrics['mean_distance']:.2f}px")
                print(f"Median distance: {metrics['median_distance']:.2f}px")
                print(f"Max distance: {metrics['max_distance']:.2f}px")
                print(f"Min distance: {metrics['min_distance']:.2f}px")
                print(f"Accuracy within 10px: {metrics['accuracy_within_10px']:.2f}%")
                print(
                    f"Accuracy within 5% screen: {metrics['accuracy_within_5_percent']:.2f}%"
                )

            all_results[model_name] = {
                "metrics": metrics,
                "results": model_results,
            }

            # Save individual model results
            results_file = size_output_path / f"{model_name}_results.json"
            with open(results_file, "w") as f:
                json.dump(
                    {
                        "model": model_name,
                        "model_id": model_id,
                        "metrics": metrics,
                        "results": model_results,
                    },
                    f,
                    indent=2,
                )

            # Save summary for this screen size
            summary = {
                "screen_size": {"name": screen_name, "width": screen_width, "height": screen_height},
                "models": {name: MODELS[name] for name in models if name in MODELS},
                "test_configs": test_configs,
                "results": {
                    model: {
                        "metrics": all_results[model]["metrics"],
                    }
                    for model in all_results
                },
            }

            summary_file = size_output_path / "summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            # Save consolidated results for web viewer
            consolidated_file = size_output_path / "consolidated_results.json"
            with open(consolidated_file, "w") as f:
                json.dump({
                    "screen_size": {"name": screen_name, "width": screen_width, "height": screen_height},
                    "models": {name: MODELS[name] for name in models if name in MODELS},
                    "tests": consolidated_results
                }, f, indent=2)

            all_size_results[screen_name] = {
                "screen_size": {"width": screen_width, "height": screen_height},
                "results": all_results,
            }

    # Save overall summary across all screen sizes
    overall_summary = {
        "screen_sizes_tested": [
            {"name": name, "width": data["screen_size"]["width"], "height": data["screen_size"]["height"]}
            for name, data in all_size_results.items()
        ],
        "models": {name: MODELS[name] for name in models if name in MODELS},
        "test_configs": test_configs,
        "results_by_screen_size": {
            name: {
                "screen_size": data["screen_size"],
                "model_metrics": {
                    model: data["results"][model]["metrics"]
                    for model in data["results"]
                }
            }
            for name, data in all_size_results.items()
        },
    }

    overall_summary_file = output_path / "overall_summary.json"
    with open(overall_summary_file, "w") as f:
        json.dump(overall_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_path}")
    if test_multiple_sizes:
        print(f"Tested {len(screen_sizes_to_test)} screen sizes")
    print(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate VLM pixel pointing accuracy"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sonnet", "opus", "gemini3", "chatgpt","haiku"],
        choices=list(MODELS.keys()),
        help="Models to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        default="pixel_pointing_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=819,
        help="Image width in pixels (default: 819, optimal for Anthropic)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1456,
        help="Image height in pixels (default: 1456, optimal for Anthropic)",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Don't save generated test images",
    )
    parser.add_argument(
        "--custom-tests",
        type=str,
        help="Path to JSON file with custom test configurations",
    )
    parser.add_argument(
        "--test-multiple-sizes",
        action="store_true",
        help="Test multiple screen sizes (recommended by Anthropic for dimensions below 1920x1080)",
    )

    args = parser.parse_args()

    # Load test configs
    if args.custom_tests:
        with open(args.custom_tests, "r") as f:
            test_configs = json.load(f)
    else:
        test_configs = TEST_CONFIGS

    run_evaluation(
        models=args.models,
        test_configs=test_configs,
        output_dir=args.output_dir,
        image_width=args.width,
        image_height=args.height,
        save_images=not args.no_save_images,
        test_multiple_sizes=args.test_multiple_sizes,
    )


if __name__ == "__main__":
    main()

