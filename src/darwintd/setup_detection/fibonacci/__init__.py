"""
Fibonacci Setup Detection Engine.

Detects Fibonacci retracement and extension setups for swing trading opportunities.
"""

from .base import FibonacciDetector
from .v1_basic import FibonacciDetectorV1
from .v2_advanced import FibonacciDetectorV2

__all__ = ['FibonacciDetector', 'FibonacciDetectorV1', 'FibonacciDetectorV2']