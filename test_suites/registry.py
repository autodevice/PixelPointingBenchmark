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

