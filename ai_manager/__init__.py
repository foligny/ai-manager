"""
AI Manager - Python Client Library

A Python client library for the AI Manager training monitoring platform.
"""

from .client import AIManager
from .run import Run

__version__ = "0.1.0"
__all__ = ["AIManager", "Run"] 