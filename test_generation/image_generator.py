import numpy as np
from PIL import Image, ImageDraw
from typing import Any, Dict, Tuple, List
import math


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
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "lime": (0, 255, 0),
        "navy": (0, 0, 128),
        "maroon": (128, 0, 0),
        "olive": (128, 128, 0),
        "teal": (0, 128, 128),
        "silver": (192, 192, 192),
    }
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        elif len(hex_color) == 3:
            return tuple(int(c*2, 16) for c in hex_color)
        return (0, 0, 0)
    
    @staticmethod
    def get_color(color_name: str) -> Tuple[int, int, int]:
        """Get RGB color from name or hexcode."""
        if color_name.startswith('#'):
            return ImageGenerator.hex_to_rgb(color_name)
        return ImageGenerator.COLOR_MAP.get(color_name, (0, 0, 255))

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
        print("debug: expected_coords", expected_coords)
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

        color = self.get_color(color_name)
        if color_name == "transparent":
            fill_color = (*color[:3], 128)
        else:
            fill_color = (*color, 255) if len(color) == 3 else color

        if config.get("multiple_shapes"):
            target_coords = self._draw_multiple_shapes(draw, config, shape, color_name, position, size)
            expected_coords = target_coords if target_coords else (center_x, center_y)
        elif shape == "circle":
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

            prompt_text = (config.get("prompt") or "").lower()
            has_explicit_expected = config.get("expected_coords") is not None
            if (not has_explicit_expected) and position == "top_right" and "corner" in prompt_text:
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
        
        elif shape == "rounded_button":
            button_width = shape_size * 2
            button_height = shape_size
            corner_radius = shape_size // 4
            bbox = [
                center_x - button_width // 2,
                center_y - button_height // 2,
                center_x + button_width // 2,
                center_y + button_height // 2,
            ]
            if color_name == "transparent":
                self._draw_rounded_rectangle(draw, bbox, corner_radius, outline=(128, 128, 128, 200), width=3)
            else:
                self._draw_rounded_rectangle(draw, bbox, corner_radius, fill=fill_color)
            expected_coords = (center_x, center_y)
        
        elif shape == "decagon":
            num_sides = 10
            radius = shape_size // 2
            points = []
            for i in range(num_sides):
                angle = 2 * math.pi * i / num_sides - math.pi / 2
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.append((x, y))
            draw.polygon(points, fill=fill_color)
            expected_coords = (center_x, center_y)
        
        elif shape == "hexagon":
            num_sides = 6
            radius = shape_size // 2
            points = []
            for i in range(num_sides):
                angle = 2 * math.pi * i / num_sides - math.pi / 2
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.append((x, y))
            draw.polygon(points, fill=fill_color)
            expected_coords = (center_x, center_y)
        
        elif shape == "pentagon":
            num_sides = 5
            radius = shape_size // 2
            points = []
            for i in range(num_sides):
                angle = 2 * math.pi * i / num_sides - math.pi / 2
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.append((x, y))
            draw.polygon(points, fill=fill_color)
            expected_coords = (center_x, center_y)
        
        elif shape == "octagon":
            num_sides = 8
            radius = shape_size // 2
            points = []
            for i in range(num_sides):
                angle = 2 * math.pi * i / num_sides - math.pi / 2
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.append((x, y))
            draw.polygon(points, fill=fill_color)
            expected_coords = (center_x, center_y)
        
        else:
            expected_coords = (center_x, center_y)

        if not config.get("multiple_shapes"):
            if isinstance(expected_coords, list):
                expected_coords = tuple(expected_coords)
            elif not isinstance(expected_coords, tuple):
                expected_coords = (center_x, center_y)
        
        img_array = np.array(img)
        return img_array, expected_coords
    
    def _draw_rounded_rectangle(self, draw, bbox, radius, fill=None, outline=None, width=1):
        """Draw a rounded rectangle."""
        x1, y1, x2, y2 = bbox
        if fill:
            draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
            draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
            draw.ellipse([x1, y1, x1 + 2*radius, y1 + 2*radius], fill=fill)
            draw.ellipse([x2 - 2*radius, y1, x2, y1 + 2*radius], fill=fill)
            draw.ellipse([x1, y2 - 2*radius, x1 + 2*radius, y2], fill=fill)
            draw.ellipse([x2 - 2*radius, y2 - 2*radius, x2, y2], fill=fill)
        if outline:
            draw.arc([x1, y1, x1 + 2*radius, y1 + 2*radius], 90, 180, fill=outline, width=width)
            draw.arc([x2 - 2*radius, y1, x2, y1 + 2*radius], 0, 90, fill=outline, width=width)
            draw.arc([x1, y2 - 2*radius, x1 + 2*radius, y2], 180, 270, fill=outline, width=width)
            draw.arc([x2 - 2*radius, y2 - 2*radius, x2, y2], 270, 360, fill=outline, width=width)
            draw.line([x1 + radius, y1, x2 - radius, y1], fill=outline, width=width)
            draw.line([x1 + radius, y2, x2 - radius, y2], fill=outline, width=width)
            draw.line([x1, y1 + radius, x1, y2 - radius], fill=outline, width=width)
            draw.line([x2, y1 + radius, x2, y2 - radius], fill=outline, width=width)
    
    def _draw_multiple_shapes(self, draw, config, target_shape, target_color, target_position, target_size):
        """Draw multiple shapes for comparison tests. Returns target shape coordinates."""
        shapes = config.get("shapes", [])
        target_coords = None
        
        for shape_config in shapes:
            shape_type = shape_config.get("shape", "circle")
            shape_color = shape_config.get("color", "blue")
            shape_pos = shape_config.get("position", "center")
            shape_size_val = shape_config.get("size", "medium")
            
            if shape_size_val == "small":
                s_size = min(self.width, self.height) // 8
            elif shape_size_val == "large":
                s_size = min(self.width, self.height) // 3
            else:
                s_size = min(self.width, self.height) // 5
            
            if shape_pos == "left" or shape_pos == "top_left":
                s_x, s_y = int(self.width * 0.3), self.height // 2
            elif shape_pos == "right" or shape_pos == "top_right":
                s_x, s_y = int(self.width * 0.7), self.height // 2
            elif shape_pos == "top":
                s_x, s_y = self.width // 2, int(self.height * 0.3)
            elif shape_pos == "bottom" or shape_pos == "bottom_left":
                s_x, s_y = self.width // 2, int(self.height * 0.7)
            elif shape_pos == "bottom_right":
                s_x, s_y = int(self.width * 0.7), int(self.height * 0.7)
            elif shape_pos == "center":
                s_x, s_y = self.width // 2, self.height // 2
            else:
                s_x, s_y = self.width // 2, self.height // 2
            
            s_color = self.get_color(shape_color)
            s_fill = (*s_color, 255)
            
            is_target = (shape_type == target_shape and 
                        shape_color == target_color and 
                        shape_pos == target_position)
            
            if shape_type == "circle":
                bbox = [s_x - s_size // 2, s_y - s_size // 2, s_x + s_size // 2, s_y + s_size // 2]
                draw.ellipse(bbox, fill=s_fill)
                if is_target:
                    target_coords = (s_x, s_y)
            elif shape_type == "square":
                bbox = [s_x - s_size // 2, s_y - s_size // 2, s_x + s_size // 2, s_y + s_size // 2]
                draw.rectangle(bbox, fill=s_fill)
                if is_target:
                    target_coords = (s_x, s_y)
            elif shape_type == "decagon":
                num_sides = 10
                radius = s_size // 2
                points = []
                for i in range(num_sides):
                    angle = 2 * math.pi * i / num_sides - math.pi / 2
                    px = s_x + radius * math.cos(angle)
                    py = s_y + radius * math.sin(angle)
                    points.append((px, py))
                draw.polygon(points, fill=s_fill)
                if is_target:
                    target_coords = (s_x, s_y)
            elif shape_type == "hexagon":
                num_sides = 6
                radius = s_size // 2
                points = []
                for i in range(num_sides):
                    angle = 2 * math.pi * i / num_sides - math.pi / 2
                    px = s_x + radius * math.cos(angle)
                    py = s_y + radius * math.sin(angle)
                    points.append((px, py))
                draw.polygon(points, fill=s_fill)
                if is_target:
                    target_coords = (s_x, s_y)
            elif shape_type == "octagon":
                num_sides = 8
                radius = s_size // 2
                points = []
                for i in range(num_sides):
                    angle = 2 * math.pi * i / num_sides - math.pi / 2
                    px = s_x + radius * math.cos(angle)
                    py = s_y + radius * math.sin(angle)
                    points.append((px, py))
                draw.polygon(points, fill=s_fill)
                if is_target:
                    target_coords = (s_x, s_y)
            elif shape_type == "rounded_button":
                button_width = s_size * 2
                button_height = s_size
                corner_radius = s_size // 4
                bbox = [s_x - button_width // 2, s_y - button_height // 2, s_x + button_width // 2, s_y + button_height // 2]
                self._draw_rounded_rectangle(draw, bbox, corner_radius, fill=s_fill)
                if is_target:
                    target_coords = (s_x, s_y)
            elif shape_type == "button":
                button_width = s_size * 2
                button_height = s_size
                bbox = [s_x - button_width // 2, s_y - button_height // 2, s_x + button_width // 2, s_y + button_height // 2]
                draw.rectangle(bbox, fill=s_fill)
                if is_target:
                    target_coords = (s_x, s_y)
        
        return target_coords

