"""Test suites module for managing different test configurations."""

from test_suites.base import TestSuite, SyntheticTestSuite, ScreenshotTestSuite
from test_suites.registry import TestSuiteRegistry, registry

__all__ = [
    "TestSuite",
    "SyntheticTestSuite",
    "ScreenshotTestSuite",
    "TestSuiteRegistry",
    "registry",
]

