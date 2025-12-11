# Pixel Pointing Benchmark

A comprehensive evaluation framework for testing Vision Language Model (VLM) accuracy in pixel-level pointing tasks. This tool generates synthetic test images, evaluates multiple VLMs, and provides a visual web interface to compare model performance.

## Overview

This benchmark evaluates how accurately different VLMs can identify and point to specific locations in images when given natural language prompts. It's particularly useful for:

- Comparing VLM performance on pixel-accurate pointing tasks
- Testing model accuracy across different screen sizes and aspect ratios
- Visualizing model predictions with an interactive web interface
- Evaluating models for UI automation and device control applications

## Features

- **Synthetic Test Image Generation**: Creates test images with various shapes (circles, squares, triangles, buttons, X marks) in different positions and colors
- **Multi-Model Evaluation**: Supports multiple VLMs including:
  - Claude Sonnet 4
  - Claude Opus 4
  - Gemini 3 Pro
  - GPT-5.1
  - Claude Haiku 4
- **Multiple Screen Sizes**: Test models across different screen dimensions and aspect ratios
- **Comprehensive Metrics**: Calculates distance errors, extraction rates, and accuracy thresholds
- **Visual Web Viewer**: Interactive HTML interface showing where each model clicked with colored dots
- **Custom Test Cases**: Define your own test scenarios via JSON configuration

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

### Basic Evaluation

Run evaluation with default settings:
```bash
python evaluate_pixel_pointing.py
```

This will:
- Test all default models (sonnet, opus, gemini3, chatgpt)
- Use default screen size (819x1456 pixels)
- Generate test images and save results to `pixel_pointing_results/`

### Custom Options

**Select specific models:**
```bash
python evaluate_pixel_pointing.py --models sonnet opus gemini3
```

**Custom screen size:**
```bash
python evaluate_pixel_pointing.py --width 1080 --height 2400
```

**Test multiple screen sizes:**
```bash
python evaluate_pixel_pointing.py --test-multiple-sizes
```

This tests 6 different optimal screen sizes:
- 9:16 aspect (819x1456)
- 1:2 aspect (784x1568)
- 2:3 aspect (896x1344)
- 3:4 aspect (951x1268)
- 1:1 aspect (1092x1092)
- 16:9 aspect (1456x819)

**Use custom test cases:**
```bash
python evaluate_pixel_pointing.py --custom-tests example_custom_tests.json
```

**Don't save images (faster, smaller output):**
```bash
python evaluate_pixel_pointing.py --no-save-images
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

Then open `http://localhost:8000/viewer.html` in your browser.

**Option 3: Direct file access (may not work due to CORS)**

If you open `viewer.html` directly in a browser, you may encounter CORS errors. Use one of the server options above instead.

**Using the Viewer:**

1. Once the viewer loads, enter the path to your results directory (default: `pixel_pointing_results`)
2. Click "Load Results"
3. Browse through test images with visual overlays showing:
   - **Black circle with white center**: Ground truth location
   - **Colored dots**: Each model's prediction
   - **Legend**: Model-to-color mapping
   - **Statistics**: Distance errors for each model

## Output Structure

Results are organized in the output directory as follows:

```
pixel_pointing_results/
├── overall_summary.json          # Summary across all screen sizes
├── optimal_9_16/                 # Results for specific screen size
│   ├── images/                   # Test images (one per test case)
│   │   ├── simple_circle.png
│   │   ├── top_right_square.png
│   │   └── ...
│   ├── consolidated_results.json # All models' predictions per test
│   ├── summary.json              # Metrics summary
│   ├── sonnet_results.json       # Individual model results
│   ├── opus_results.json
│   └── ...
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

The web viewer uses the following color scheme:
- **Sonnet**: Red (#FF6B6B)
- **Opus**: Orange (#FFA500)
- **Gemini3**: Teal (#4ECDC4)
- **ChatGPT**: Light Green (#95E1D3)
- **Haiku**: Pink (#F38181)
- **Ground Truth**: Black circle with white center

## Requirements

- Python 3.8+
- API keys for the models you want to test
- Modern web browser for viewing results

## License

This project is provided as-is for evaluation and research purposes.

