from pathlib import Path
from typing import Dict, List, Optional
from test_suites.base import TestSuite, SyntheticTestSuite, ScreenshotTestSuite


DEFAULT_SYNTHETIC_TESTS = [
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

COLOR_IDENTIFICATION_TESTS = [
    {
        "name": "basic_red",
        "prompt": "Point to the center of the red circle",
        "shape": "circle",
        "color": "red",
        "position": "top_left",
    },
    {
        "name": "basic_blue",
        "prompt": "Point to the center of the blue circle",
        "shape": "circle",
        "color": "blue",
        "position": "top_right",
    },
    {
        "name": "basic_green",
        "prompt": "Point to the center of the green circle",
        "shape": "circle",
        "color": "green",
        "position": "bottom_left",
    },
    {
        "name": "basic_yellow",
        "prompt": "Point to the center of the yellow circle",
        "shape": "circle",
        "color": "yellow",
        "position": "bottom_right",
    },
    {
        "name": "subtle_red_orange",
        "prompt": "Point to the center of the red-orange circle (not the pure orange one)",
        "shape": "circle",
        "color": "#FF4500",
        "position": "left",
        "multiple_shapes": True,
        "shapes": [
            {"shape": "circle", "color": "#FF4500", "position": "left"},
            {"shape": "circle", "color": "orange", "position": "right"},
        ],
    },
    {
        "name": "subtle_blue_cyan",
        "prompt": "Point to the center of the cyan circle (not the blue one)",
        "shape": "circle",
        "color": "cyan",
        "position": "left",
        "multiple_shapes": True,
        "shapes": [
            {"shape": "circle", "color": "cyan", "position": "left"},
            {"shape": "circle", "color": "blue", "position": "right"},
        ],
    },
    {
        "name": "hexcode_1",
        "prompt": "Point to the center of the circle with hex color #FF6B9D",
        "shape": "circle",
        "color": "#FF6B9D",
        "position": "top_left",
    },
    {
        "name": "hexcode_2",
        "prompt": "Point to the center of the circle with hex color #4ECDC4",
        "shape": "circle",
        "color": "#4ECDC4",
        "position": "top_right",
    },
    {
        "name": "hexcode_3",
        "prompt": "Point to the center of the circle with hex color #95E1D3",
        "shape": "circle",
        "color": "#95E1D3",
        "position": "bottom_left",
    },
    {
        "name": "hexcode_similar",
        "prompt": "Point to the center of the circle with hex color #FF6B9D (not #FF6B9E)",
        "shape": "circle",
        "color": "#FF6B9D",
        "position": "left",
        "multiple_shapes": True,
        "shapes": [
            {"shape": "circle", "color": "#FF6B9D", "position": "left"},
            {"shape": "circle", "color": "#FF6B9E", "position": "right"},
        ],
    },
]

SHAPE_IDENTIFICATION_TESTS = [
    {
        "name": "basic_circle",
        "prompt": "Point to the center of the circle",
        "shape": "circle",
        "color": "blue",
        "position": "center",
    },
    {
        "name": "basic_square",
        "prompt": "Point to the center of the square",
        "shape": "square",
        "color": "blue",
        "position": "center",
    },
    {
        "name": "basic_triangle",
        "prompt": "Point to the center of the triangle",
        "shape": "triangle",
        "color": "blue",
        "position": "center",
    },
    {
        "name": "pentagon",
        "prompt": "Point to the center of the pentagon",
        "shape": "pentagon",
        "color": "blue",
        "position": "center",
    },
    {
        "name": "hexagon",
        "prompt": "Point to the center of the hexagon",
        "shape": "hexagon",
        "color": "blue",
        "position": "center",
    },
    {
        "name": "octagon",
        "prompt": "Point to the center of the octagon",
        "shape": "octagon",
        "color": "blue",
        "position": "center",
    },
    {
        "name": "decagon_among_circles",
        "prompt": "Point to the center of the decagon (the 10-sided polygon, not the circles)",
        "shape": "decagon",
        "color": "red",
        "position": "center",
        "multiple_shapes": True,
        "shapes": [
            {"shape": "circle", "color": "blue", "position": "top_left"},
            {"shape": "circle", "color": "blue", "position": "top_right"},
            {"shape": "decagon", "color": "red", "position": "center"},
            {"shape": "circle", "color": "blue", "position": "bottom_left"},
            {"shape": "circle", "color": "blue", "position": "bottom_right"},
        ],
    },
    {
        "name": "rounded_button",
        "prompt": "Point to the center of the button with rounded corners",
        "shape": "rounded_button",
        "color": "green",
        "position": "left",
        "multiple_shapes": True,
        "shapes": [
            {"shape": "rounded_button", "color": "green", "position": "left"},
            {"shape": "button", "color": "green", "position": "right"},
        ],
    },
    {
        "name": "hexagon_vs_octagon",
        "prompt": "Point to the center of the hexagon (6 sides, not 8)",
        "shape": "hexagon",
        "color": "purple",
        "position": "left",
        "multiple_shapes": True,
        "shapes": [
            {"shape": "hexagon", "color": "purple", "position": "left"},
            {"shape": "octagon", "color": "purple", "position": "right"},
        ],
    },
]


RESOLUTION_TEST_TESTS = [
    {
        "name": "simple_rectangle_256",
        "prompt": "Point to the center of the purple rectangle",
        "shape": "square",
        "color": "purple",
        "position": "top_left",
        "size": "medium",
    },
    {
        "name": "simple_rectangle_512",
        "prompt": "Point to the center of the purple rectangle",
        "shape": "square",
        "color": "purple",
        "position": "top_right",
        "size": "medium",
    },
    {
        "name": "simple_rectangle_1024",
        "prompt": "Point to the center of the purple rectangle",
        "shape": "square",
        "color": "purple",
        "position": "bottom_left",
        "size": "medium",
    },
    {
        "name": "simple_rectangle_2048",
        "prompt": "Point to the center of the purple rectangle",
        "shape": "square",
        "color": "purple",
        "position": "bottom_right",
        "size": "medium",
    },
    {
        "name": "simple_rectangle_4096",
        "prompt": "Point to the center of the purple rectangle",
        "shape": "square",
        "color": "purple",
        "position": "center",
        "size": "medium",
    },
]


SIZE_COMPARISON_TESTS = [
    {
        "name": "larger_circle_obvious",
        "prompt": "Point to the center of the larger circle",
        "shape": "circle",
        "color": "blue",
        "position": "left",
        "size": "large",
        "multiple_shapes": True,
        "shapes": [
            {"shape": "circle", "color": "blue", "position": "left", "size": "large"},
            {"shape": "circle", "color": "blue", "position": "right", "size": "small"},
        ],
    },
    {
        "name": "larger_circle_moderate",
        "prompt": "Point to the center of the larger circle",
        "shape": "circle",
        "color": "blue",
        "position": "left",
        "size": "medium",
        "multiple_shapes": True,
        "shapes": [
            {"shape": "circle", "color": "blue", "position": "left", "size": "medium"},
            {"shape": "circle", "color": "blue", "position": "right", "size": "small"},
        ],
    },
    {
        "name": "larger_circle_subtle",
        "prompt": "Point to the center of the larger circle",
        "shape": "circle",
        "color": "blue",
        "position": "left",
        "size": "medium",
        "multiple_shapes": True,
        "shapes": [
            {"shape": "circle", "color": "blue", "position": "left", "size": "medium"},
            {"shape": "circle", "color": "blue", "position": "right", "size": "medium", "size_override": 0.85},
        ],
    },
    {
        "name": "smaller_square_obvious",
        "prompt": "Point to the center of the smaller square",
        "shape": "square",
        "color": "red",
        "position": "right",
        "size": "small",
        "multiple_shapes": True,
        "shapes": [
            {"shape": "square", "color": "red", "position": "left", "size": "large"},
            {"shape": "square", "color": "red", "position": "right", "size": "small"},
        ],
    },
    {
        "name": "smaller_square_moderate",
        "prompt": "Point to the center of the smaller square",
        "shape": "square",
        "color": "red",
        "position": "right",
        "size": "small",
        "multiple_shapes": True,
        "shapes": [
            {"shape": "square", "color": "red", "position": "left", "size": "medium"},
            {"shape": "square", "color": "red", "position": "right", "size": "small"},
        ],
    },
    {
        "name": "smaller_square_subtle",
        "prompt": "Point to the center of the smaller square",
        "shape": "square",
        "color": "red",
        "position": "right",
        "size": "small",
        "multiple_shapes": True,
        "shapes": [
            {"shape": "square", "color": "red", "position": "left", "size": "medium"},
            {"shape": "square", "color": "red", "position": "right", "size": "small"},
        ],
    },
]


class TestSuiteRegistry:
    """Registry for managing test suites."""
    
    def __init__(self):
        self._suites: Dict[str, TestSuite] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default test suites."""
        basic_suite = SyntheticTestSuite(
            name="basic_shapes",
            description="Basic synthetic shapes test suite",
            configs=DEFAULT_SYNTHETIC_TESTS
        )
        self.register(basic_suite)
        
        color_suite = SyntheticTestSuite(
            name="color_identification",
            description="Color identification tests: basic colors → subtle differences → hexcode colors",
            configs=COLOR_IDENTIFICATION_TESTS
        )
        self.register(color_suite)
        
        shape_suite = SyntheticTestSuite(
            name="shape_identification",
            description="Shape identification tests: basic shapes → subtle differences (decagon, rounded corners)",
            configs=SHAPE_IDENTIFICATION_TESTS
        )
        self.register(shape_suite)
        
        resolution_suite_1024x1024 = SyntheticTestSuite(
            name="resolution_test_1024x1024",
            description="Resolution stress test for 1024 x 1024 resolution: same simple test at increasing resolutions",
            configs=RESOLUTION_TEST_TESTS
        )
        self.register(resolution_suite_1024x1024)
        
        resolution_suite_512x512 = SyntheticTestSuite(
            name="resolution_test_512x512",
            description="Resolution stress test for 512 x 512 resolution: same simple test at increasing resolutions",
            configs=RESOLUTION_TEST_TESTS
        )
        self.register(resolution_suite_512x512)
        
        size_suite = SyntheticTestSuite(
            name="size_comparison",
            description="Size comparison tests: find larger/smaller shapes until indistinguishable",
            configs=SIZE_COMPARISON_TESTS
        )
        self.register(size_suite)
    
    def register(self, suite: TestSuite):
        """Register a test suite."""
        self._suites[suite.name] = suite
    
    def get(self, name: str) -> Optional[TestSuite]:
        """Get a test suite by name."""
        return self._suites.get(name)
    
    def list_all(self) -> List[str]:
        """List all registered test suite names."""
        return list(self._suites.keys())
    
    def load_from_directory(self, directory: Path):
        """Load test suites from a directory."""
        directory = Path(directory)
        if not directory.exists():
            return
        
        for item in directory.iterdir():
            if item.is_dir():
                metadata_file = item / "metadata.json"
                if metadata_file.exists():
                    suite = ScreenshotTestSuite(
                        name=item.name,
                        base_path=item
                    )
                    self.register(suite)


# Global registry instance
registry = TestSuiteRegistry()

