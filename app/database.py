"""
Database configuration and models for AI Manager.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Float, Boolean, 
    ForeignKey, JSON, LargeBinary
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from app.config import settings

# Create database engine
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    projects = relationship("Project", back_populates="owner")


class Project(Base):
    """Project model for organizing experiments."""
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    is_public = Column(Boolean, default=False)
    tags = Column(JSON, default=[])  # Project tags
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    runs = relationship("Run", back_populates="project")
    assigned_models = relationship("ProjectModel", back_populates="project")


class Run(Base):
    """Run model for individual training experiments."""
    __tablename__ = "runs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    status = Column(String, default="running")  # running, completed, failed, stopped
    config = Column(JSON, default={})  # Training configuration
    tags = Column(JSON, default=[])  # Run tags
    notes = Column(Text, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="runs")
    metrics = relationship("Metric", back_populates="run")
    artifacts = relationship("Artifact", back_populates="run")
    logs = relationship("Log", back_populates="run")


class Metric(Base):
    """Metric model for tracking training metrics."""
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    name = Column(String, index=True)
    value = Column(Float)
    step = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    run = relationship("Run", back_populates="metrics")


class Artifact(Base):
    """Artifact model for storing files and models."""
    __tablename__ = "artifacts"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    name = Column(String, index=True)
    type = Column(String)  # model, data, config, other
    path = Column(String)  # File path in storage
    size = Column(Integer)  # File size in bytes
    artifact_metadata = Column(JSON, default={})  # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    run = relationship("Run", back_populates="artifacts")


class Log(Base):
    """Log model for storing training logs."""
    __tablename__ = "logs"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    level = Column(String)  # INFO, WARNING, ERROR, DEBUG
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    run = relationship("Run", back_populates="logs")


class ProjectModel(Base):
    """Association table for projects and models."""
    __tablename__ = "project_models"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    model_name = Column(String, index=True)  # Name of the model file
    model_path = Column(String)  # Path to the model file
    model_type = Column(String)  # Type of model (e.g., "text", "image", "audio")
    model_capabilities = Column(JSON, default=[])  # Model capabilities
    assigned_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="assigned_models")


# Database dependency
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create all tables
def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine) 