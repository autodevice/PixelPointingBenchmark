# Codebase Structure

This document describes the internal structure and organization of the Pixel Pointing Benchmark codebase. For usage instructions, see [README.md](README.md).

## Directory Organization

```
PixelPointingBenchmark/
├── evaluation/              # Evaluation logic
│   ├── __init__.py         # Module exports
│   ├── vlm_evaluator.py    # VLM querying and coordinate extraction
│   ├── metrics.py          # Accuracy metrics and statistics
│   ├── results_manager.py  # Results storage, consolidation, and indexing
│   ├── runner.py           # Main evaluation runner
│   └── utils.py            # CLI utilities for result management
│
├── test_generation/        # Test image generation
│   ├── __init__.py         # Module exports
│   └── image_generator.py  # Synthetic image generation
│
├── test_suites/            # Test suite management
│   ├── __init__.py         # Module exports
│   ├── base.py             # Base classes for test suites
│   └── registry.py         # Test suite registry
│
├── evaluate.py             # Main entry point
├── serve_viewer.py         # Web server for viewer
├── viewer_v2.html          # Enhanced results viewer
└── viewer.html             # Legacy viewer
```

## Module Dependencies

### evaluation/
- **vlm_evaluator.py**: No internal dependencies
- **metrics.py**: No internal dependencies
- **results_manager.py**: No internal dependencies
- **runner.py**: Depends on vlm_evaluator, metrics, results_manager, test_suites.registry
- **utils.py**: Depends on results_manager

### test_generation/
- **image_generator.py**: No internal dependencies

### test_suites/
- **base.py**: Depends on test_generation.image_generator (lazy import)
- **registry.py**: Depends on base

## Module Responsibilities

### evaluation/
- **vlm_evaluator.py**: Handles communication with VLMs via LiteLLM, extracts coordinates from responses
- **metrics.py**: Calculates accuracy metrics (distance, extraction rate, statistics across passes)
- **results_manager.py**: Manages result storage, consolidation, indexing, and utilities
- **runner.py**: Orchestrates the evaluation process
- **utils.py**: CLI tool for result management operations

### test_generation/
- **image_generator.py**: Generates synthetic test images with various shapes and configurations

### test_suites/
- **base.py**: Abstract base classes for test suites (SyntheticTestSuite, ScreenshotTestSuite)
- **registry.py**: Manages registration and retrieval of test suites

## Results Management API

### ResultsManager Methods
- `save_run()`: Save individual evaluation runs with timestamps
- `load_runs()`: Load runs for a test suite and screen size
- `consolidate_results()`: Create consolidated results JSON for viewer
- `update_test_suites_index()`: Update test_suites.json index file
- `fix_consolidated_results()`: Fix missing models list in consolidated results

## Data Flow

1. **Test Generation**: `test_generation` creates synthetic images
2. **Test Suites**: `test_suites` manages test configurations
3. **Evaluation**: `evaluation.runner` orchestrates VLM queries
4. **Results**: `evaluation.results_manager` stores and consolidates results
5. **Visualization**: `viewer_v2.html` displays consolidated results

