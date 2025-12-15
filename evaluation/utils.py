#!/usr/bin/env python3
"""Utility functions for managing evaluation results."""

import argparse
from pathlib import Path
from evaluation.results_manager import ResultsManager


def main():
    """CLI utility for managing results."""
    parser = argparse.ArgumentParser(
        description="Utility for managing evaluation results"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory (default: results)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    fix_parser = subparsers.add_parser("fix", help="Fix consolidated results")
    fix_parser.add_argument("--test-suite", required=True, help="Test suite name")
    fix_parser.add_argument("--screen-size", default="custom", help="Screen size (default: custom)")
    
    index_parser = subparsers.add_parser("index", help="Update test suites index")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ResultsManager(args.results_dir)
    
    if args.command == "fix":
        try:
            models = manager.fix_consolidated_results(args.test_suite, args.screen_size)
            print(f"Fixed consolidated results for {args.test_suite}/{args.screen_size}")
            print(f"Found {len(models)} models: {', '.join(models)}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
    
    elif args.command == "index":
        test_suites = manager.update_test_suites_index()
        print(f"Updated test_suites.json with {len(test_suites)} test suite(s):")
        for suite in test_suites:
            print(f"  - {suite}")
    
    return 0


if __name__ == "__main__":
    exit(main())

