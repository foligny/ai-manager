"""
Metric schemas for tracking training metrics.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel


class MetricBase(BaseModel):
    """Base metric schema."""
    name: str
    value: float
    step: int = 0


class MetricCreate(MetricBase):
    """Schema for metric creation."""
    pass


class Metric(MetricBase):
    """Schema for metric response."""
    id: int
    run_id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True


class MetricBatch(BaseModel):
    """Schema for batch metric logging."""
    metrics: Dict[str, float]
    step: Optional[int] = None


class MetricHistory(BaseModel):
    """Schema for metric history."""
    name: str
    values: list[float]
    steps: list[int]
    timestamps: list[datetime]


class MetricSummary(BaseModel):
    """Schema for metric summary."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    mean_value: float
    total_points: int 