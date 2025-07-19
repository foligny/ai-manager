"""
Artifact schemas for file and model storage.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel


class ArtifactBase(BaseModel):
    """Base artifact schema."""
    name: str
    type: str  # model, data, config, other
    metadata: Dict[str, Any] = {}


class ArtifactCreate(ArtifactBase):
    """Schema for artifact creation."""
    pass


class Artifact(ArtifactBase):
    """Schema for artifact response."""
    id: int
    run_id: int
    path: str
    size: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class LogBase(BaseModel):
    """Base log schema."""
    level: str  # INFO, WARNING, ERROR, DEBUG
    message: str


class LogCreate(LogBase):
    """Schema for log creation."""
    pass


class Log(LogBase):
    """Schema for log response."""
    id: int
    run_id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True 