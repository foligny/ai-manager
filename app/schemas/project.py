"""
Project schemas.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class ProjectBase(BaseModel):
    """Base project schema."""
    name: str
    description: Optional[str] = None
    is_public: bool = False


class ProjectCreate(ProjectBase):
    """Schema for project creation."""
    pass


class ProjectUpdate(BaseModel):
    """Schema for project updates."""
    name: Optional[str] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None


class Project(ProjectBase):
    """Schema for project response."""
    id: int
    owner_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ProjectWithRuns(Project):
    """Schema for project with runs."""
    runs: List["RunSummary"] = [] 