"""
AI Manager - Python Client Library

A Python client library for the AI Manager training monitoring platform.
"""

from .client import AIManager
from .run import Run
from .types import (
    MetricInterface, MetricDict, ConfigDict, ArtifactInfo, 
    RunStatus, RunInfo, ArtifactType, MetricValue
)

__version__ = "0.1.0"
__all__ = [
    "AIManager", "Run", "MetricInterface", "MetricDict", 
    "ConfigDict", "ArtifactInfo", "RunStatus", "RunInfo", 
    "ArtifactType", "MetricValue"
] 