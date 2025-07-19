"""
Run schemas for experiment tracking.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class RunBase(BaseModel):
    """Base run schema."""
    name: str
    config: Dict[str, Any] = {}
    tags: List[str] = []
    notes: Optional[str] = None


class RunCreate(RunBase):
    """Schema for run creation."""
    pass


class RunUpdate(BaseModel):
    """Schema for run updates."""
    name: Optional[str] = None
    status: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    ended_at: Optional[datetime] = None


class RunSummary(BaseModel):
    """Schema for run summary."""
    id: int
    name: str
    status: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    tags: List[str] = []
    
    class Config:
        from_attributes = True


class Run(RunSummary):
    """Schema for run response."""
    project_id: int
    config: Dict[str, Any] = {}
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class RunWithMetrics(Run):
    """Schema for run with metrics."""
    metrics: List["Metric"] = []


class RunWithArtifacts(Run):
    """Schema for run with artifacts."""
    artifacts: List["Artifact"] = []


class RunWithLogs(Run):
    """Schema for run with logs."""
    logs: List["Log"] = [] 