from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image


class TestSuite(ABC):
    """Base class for test suites."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def get_test_cases(self) -> List[Dict[str, Any]]:
        """Return list of test case configurations."""
        pass
    
    @abstractmethod
    def generate_test_image(
        self, config: Dict[str, Any], width: int, height: int
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Generate test image and return (image_array, expected_coords)."""
        print("debug: generate_test_image", config, width, height)
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this test suite."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
        }


class SyntheticTestSuite(TestSuite):
    """Test suite that generates synthetic images."""
    
    def __init__(self, name: str, description: str = "", configs: List[Dict[str, Any]] = None):
        super().__init__(name, description)
        self.configs = configs or []
        self.generator = None
    
    def get_test_cases(self) -> List[Dict[str, Any]]:
        return self.configs
    
    def generate_test_image(
        self, config: Dict[str, Any], width: int, height: int
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        if self.generator is None or self.generator.width != width or self.generator.height != height:
            try:
                from test_generation.image_generator import ImageGenerator
            except ImportError:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from test_generation.image_generator import ImageGenerator
            self.generator = ImageGenerator(width, height)
        print("debug: config", config)
        return self.generator.generate_image(config)


class ScreenshotTestSuite(TestSuite):
    """Test suite that uses static screenshot images."""
    
    def __init__(self, name: str, description: str = "", base_path: Path = None):
        super().__init__(name, description)
        self.base_path = Path(base_path) if base_path else Path("test_suites/screenshots") / name
        self.test_cases = []
        self._load_test_cases()
    
    def _load_test_cases(self):
        """Load test case metadata from JSON file."""
        metadata_file = self.base_path / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file, "r") as f:
                data = json.load(f)
                self.test_cases = data.get("test_cases", [])
        else:
            image_files = sorted(self.base_path.glob("*.png"))
            self.test_cases = [
                {
                    "name": img.stem,
                    "image_file": str(img.relative_to(self.base_path)),
                    "prompt": f"Point to the target in {img.stem}",
                }
                for img in image_files
            ]
    
    def get_test_cases(self) -> List[Dict[str, Any]]:
        return self.test_cases
    
    def generate_test_image(
        self, config: Dict[str, Any], width: int, height: int
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        image_file = self.base_path / config.get("image_file", f"{config['name']}.png")
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_file}")
        
        img = Image.open(image_file)
        img_array = np.array(img)
        
        expected_coords = config.get("expected_coords")
        if expected_coords:
            if isinstance(expected_coords, list):
                expected_coords = tuple(expected_coords)
        else:
            expected_coords = (img.width // 2, img.height // 2)
        
        return img_array, expected_coords

