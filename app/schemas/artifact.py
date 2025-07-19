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
    artifact_metadata: Dict[str, Any] = {}
    created_at: datetime
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_orm(cls, obj):
        """Custom from_orm to handle JSON string conversion."""
        if hasattr(obj, 'artifact_metadata') and isinstance(obj.artifact_metadata, str):
            import json
            try:
                obj.artifact_metadata = json.loads(obj.artifact_metadata)
            except (json.JSONDecodeError, TypeError):
                obj.artifact_metadata = {}
        return super().from_orm(obj)


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