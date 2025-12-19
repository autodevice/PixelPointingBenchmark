#!/usr/bin/env python3
"""Main entry point for pixel pointing evaluation."""

import argparse
from pathlib import Path
from dotenv import load_dotenv

from evaluation.runner import run_evaluation, MODELS
from test_suites.registry import registry

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLM pixel pointing accuracy"
    )
    parser.add_argument(
        "--test-suite",
        default="basic_shapes",
        help="Test suite to run (default: basic_shapes)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        help="Models to evaluate. If not provided, all available models are used.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=819,
        help="Image width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1456,
        help="Image height in pixels",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=3,
        help="Number of passes to run (for statistics, default: 3)",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Don't save generated test images",
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="List available test suites and exit",
    )

    args = parser.parse_args()

    if args.list_suites:
        print("Available test suites:")
        for suite_name in registry.list_all():
            suite = registry.get(suite_name)
            print(f"  - {suite_name}: {suite.description if suite.description else 'No description'}")
        return

    # If no models are specified, run all available models
    selected_models = args.models if args.models is not None else list(MODELS.keys())

    run_evaluation(
        test_suite_name=args.test_suite,
        models=selected_models,
        output_dir=args.output_dir,
        screen_width=args.width,
        screen_height=args.height,
        num_passes=args.num_passes,
        save_images=not args.no_save_images,
    )


if __name__ == "__main__":
    main()

