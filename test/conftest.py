"""Pytest configuration for ahocorasick-ner tests.

Ensures optional dependencies are available for tests that need them.
"""
import sys

# Check for optional backend dependencies
try:
    import numpy
except ImportError:
    # numpy is optional but required for backend tests
    pass

try:
    import onnx
except ImportError:
    # onnx is optional but required for backend tests
    pass

try:
    import onnxruntime
except ImportError:
    # onnxruntime is optional but required for backend tests
    pass
