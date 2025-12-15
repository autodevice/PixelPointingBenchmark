import numpy as np
from PIL import Image, ImageDraw
from typing import Any, Dict, Tuple


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
        "transparent": (255, 255, 255, 128),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }

    def __init__(
        self,
        width: int = 819,
        height: int = 1456,
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
        
        expected_coords = config.get("expected_coords")
        if expected_coords:
            if isinstance(expected_coords, list) and len(expected_coords) == 2:
                center_x, center_y = expected_coords[0], expected_coords[1]
            elif isinstance(expected_coords, tuple) and len(expected_coords) == 2:
                center_x, center_y = expected_coords[0], expected_coords[1]
            else:
                expected_coords = None

        if bg_color_name != "white":
            img = Image.new("RGB", (self.width, self.height), self.COLOR_MAP[bg_color_name])
            draw = ImageDraw.Draw(img, "RGBA")

        if size == "small":
            shape_size = min(self.width, self.height) // 8
        elif size == "large":
            shape_size = min(self.width, self.height) // 3
        else:
            shape_size = min(self.width, self.height) // 5

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

        if overlap:
            square_size = shape_size
            square_x = center_x - square_size // 2
            square_y = center_y - square_size // 2
            draw.rectangle(
                [square_x, square_y, square_x + square_size, square_y + square_size],
                fill=(0, 0, 255, 200),
            )
            circle_offset = shape_size // 4
            circle_center_x = center_x - circle_offset
            circle_center_y = center_y - circle_offset

        color = self.COLOR_MAP.get(color_name, (0, 0, 255))
        if color_name == "transparent":
            fill_color = (*color[:3], 128)
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
                expected_coords = (bbox[2], bbox[1])
            else:
                expected_coords = (center_x, center_y)

        elif shape == "triangle":
            points = [
                (center_x, center_y - shape_size // 2),
                (center_x - shape_size // 2, center_y + shape_size // 2),
                (center_x + shape_size // 2, center_y + shape_size // 2),
            ]
            draw.polygon(points, fill=fill_color)
            expected_coords = (center_x, center_y)

        elif shape == "x":
            line_width = max(3, shape_size // 10)
            half_size = shape_size // 2
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
            button_width = shape_size * 2
            button_height = shape_size
            bbox = [
                center_x - button_width // 2,
                center_y - button_height // 2,
                center_x + button_width // 2,
                center_y + button_height // 2,
            ]
            if color_name == "transparent":
                draw.rectangle(bbox, outline=(128, 128, 128, 200), width=3)
            else:
                draw.rectangle(bbox, fill=fill_color)
            expected_coords = (center_x, center_y)

        else:
            expected_coords = (center_x, center_y)

        if isinstance(expected_coords, list):
            expected_coords = tuple(expected_coords)
        elif not isinstance(expected_coords, tuple):
            expected_coords = (center_x, center_y)

        img_array = np.array(img)
        return img_array, expected_coords

