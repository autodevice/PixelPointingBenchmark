# Pixel Pointing Benchmark

A comprehensive evaluation framework for testing Vision Language Model (VLM) accuracy in pixel-level pointing tasks. This tool generates synthetic test images, evaluates multiple VLMs, and provides a visual web interface to compare model performance.

> For codebase structure and module organization, see [STRUCTURE.md](STRUCTURE.md).

## Overview

This benchmark evaluates how accurately different VLMs can identify and point to specific locations in images when given natural language prompts. It's particularly useful for:

- Comparing VLM performance on pixel-accurate pointing tasks
- Testing model accuracy across different screen sizes and aspect ratios
- Visualizing model predictions with an interactive web interface
- Evaluating models for UI automation and device control applications

## Features

- **Test Suite System**: Modular test suite architecture supporting both synthetic and screenshot-based tests
- **Synthetic Test Image Generation**: Creates test images with various shapes (circles, squares, triangles, buttons, X marks) in different positions and colors
- **Multi-Model Evaluation**: Supports multiple VLMs including:
  - Claude Sonnet 4
  - Claude Opus 4
  - Gemini 3 Pro
  - GPT-5.1
  - Claude Haiku 4
- **Multiple Passes**: Run evaluations multiple times to calculate statistics and standard deviation
- **Non-Overwriting Results**: Results are stored with timestamps, allowing multiple runs without overwriting
- **Multiple Screen Sizes**: Test models across different screen dimensions and aspect ratios
- **Comprehensive Metrics**: Calculates distance errors, extraction rates, accuracy thresholds, and standard deviation across passes
- **Enhanced Visual Web Viewer**: Interactive HTML interface with:
  - Test suite selection
  - Model and pass filtering
  - Multiple pass visualization
  - Improved color scheme
  - Statistical summaries (mean, std dev, min/max)
- **Custom Test Cases**: Define your own test scenarios via JSON configuration or create custom test suites

## Installation

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file in the project root with your API keys:
   ```
   ANTHROPIC_API_KEY=your_anthropic_key
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   ```

## Usage

### New Structure (Recommended)

The codebase has been refactored with a modular architecture. Use the new `evaluate.py`:

```bash
# List available test suites
python evaluate.py --list-suites

# Run a specific test suite with specific models
python evaluate.py --test-suite basic_shapes --models sonnet opus gemini3

# Run with multiple passes for statistics
python evaluate.py --test-suite basic_shapes --models sonnet --num-passes 3

# Custom screen size
python evaluate.py --test-suite basic_shapes --width 1080 --height 2400
```

### Custom Options

**Select specific models:**
```bash
python evaluate.py --test-suite basic_shapes --models sonnet opus gemini3
```

**Custom screen size:**
```bash
python evaluate.py --test-suite basic_shapes --width 1080 --height 2400
```

**Run multiple passes for statistics:**
```bash
python evaluate.py --test-suite basic_shapes --models sonnet --num-passes 3
```

**Don't save images (faster, smaller output):**
```bash
python evaluate.py --test-suite basic_shapes --no-save-images
```

### Utility Commands

**Fix consolidated results (if models list is missing):**
```bash
python -m evaluation.utils fix --test-suite basic_shapes --screen-size custom
```

**Update test suites index:**
```bash
python -m evaluation.utils index
```

### Viewing Results

**Option 1: Using the Python server (Recommended)**

Run the included server script:
```bash
python serve_viewer.py
```

This will:
- Start a local web server on port 8000
- Automatically open the viewer in your browser
- Allow the viewer to load JSON files and images

**Option 2: Using Python's built-in server**

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/viewer_v2.html` in your browser (or `viewer.html` for the legacy viewer).

**Using the Enhanced Viewer (viewer_v2.html):**

1. Select the results directory (default: `results`)
2. Select a test suite from the dropdown
3. Click "Load Results"
4. Use the filters to:
   - Show/hide specific models
   - Show/hide specific passes (for multi-pass runs)
5. Browse through test images with visual overlays showing:
   - **Black circle with white center**: Ground truth location
   - **Colored dots**: Each model's prediction (multiple passes shown with reduced opacity)
   - **Legend**: Click legend items to toggle model visibility
   - **Statistics**: Mean distance, standard deviation, and min/max across passes

## Output Structure

Results are organized in the output directory as follows:

```
results/
├── test_suites.json              # Index of all test suites
├── basic_shapes/                 # Test suite name
│   └── custom/                   # Screen size name
│       ├── images/               # Test images (one per test case)
│       │   ├── simple_circle.png
│       │   ├── top_right_square.png
│       │   └── ...
│       ├── consolidated_results.json  # All models' predictions per test
│       ├── runs_index.json       # Index of all runs
│       ├── sonnet_pass1_*.json   # Individual run results
│       ├── opus_pass1_*.json
│       └── ...
└── ...
```

## Test Configuration

### Default Test Cases

The benchmark includes 8 default test cases:
- Simple circle (center)
- Top-right square corner
- Middle of X mark
- Transparent button
- Small circle
- Bottom-left triangle
- Overlapping shapes
- Low contrast button

### Custom Test Cases

Create a JSON file with custom test configurations:

```json
[
  {
    "name": "custom_circle",
    "prompt": "Point to the center of the purple circle",
    "shape": "circle",
    "color": "purple",
    "position": "center",
    "expected_coords": [540, 1200]
  },
  {
    "name": "custom_button",
    "prompt": "Point to the center of the red button",
    "shape": "button",
    "color": "red",
    "position": "center",
    "size": "large"
  }
]
```

**Available options:**
- `name`: Unique test identifier
- `prompt`: Natural language instruction for the model
- `shape`: `circle`, `square`, `triangle`, `x`, `button`
- `color`: `purple`, `red`, `blue`, `green`, `yellow`, `orange`, `gray`, `lightgray`, `transparent`
- `position`: `center`, `top_left`, `top_right`, `bottom_left`, `bottom_right`
- `size`: `small`, `medium` (default), `large`
- `expected_coords`: `[x, y]` - Optional exact coordinates
- `overlap`: `true` - For overlapping shapes test
- `background`: Background color name

## Metrics

The benchmark calculates several accuracy metrics:

- **Extraction Rate**: Percentage of successful coordinate extractions
- **Mean Distance**: Average pixel distance from ground truth
- **Median Distance**: Median pixel distance
- **Accuracy within 10px**: Percentage of predictions within 10 pixels
- **Accuracy within 5%**: Percentage within 5% of screen diagonal

## Model Colors in Viewer

The enhanced viewer (viewer_v2.html) uses the following color scheme:
- **Sonnet**: rgb(168, 2, 15) (Red)
- **Opus**: rgb(255, 132, 0) (Orange)
- **Gemini3**: rgb(0, 255, 76) (Green)
- **ChatGPT**: rgb(17, 160, 207) (Blue)
- **Haiku**: rgb(164, 11, 224) (Purple)
- **Ground Truth**: Black circle with white center

Colors are displayed as dots next to model names in the statistics section.

## Requirements

- Python 3.8+
- API keys for the models you want to test
- Modern web browser for viewing results

## License

This project is provided as-is for evaluation and research purposes.

