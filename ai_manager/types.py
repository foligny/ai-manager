"""
Type-safe interfaces for AI Manager.
"""

from typing import Dict, Any, Union, Optional, List
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class MetricValue(BaseModel):
    """Type-safe metric value."""
    value: Union[float, int, bool, str]
    step: Optional[int] = None
    timestamp: Optional[datetime] = None


class MetricDict(BaseModel):
    """Type-safe metric dictionary."""
    metrics: Dict[str, Union[float, int, bool, str]]
    step: Optional[int] = None
    
    @validator('metrics')
    def validate_metrics(cls, v):
        """Validate that all metric values are valid."""
        for key, value in v.items():
            if not isinstance(value, (float, int, bool, str)):
                raise ValueError(f"Metric value must be float, int, bool, or str, got {type(value)} for {key}")
        return v


class ArtifactType(str, Enum):
    """Valid artifact types."""
    MODEL = "model"
    DATA = "data"
    CONFIG = "config"
    OTHER = "other"


class ArtifactInfo(BaseModel):
    """Type-safe artifact information."""
    file_path: str
    name: Optional[str] = None
    artifact_type: ArtifactType = ArtifactType.OTHER
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConfigDict(BaseModel):
    """Type-safe configuration dictionary."""
    config: Dict[str, Any]
    
    @validator('config')
    def validate_config(cls, v):
        """Validate configuration values."""
        for key, value in v.items():
            if not isinstance(value, (str, int, float, bool, list, dict)):
                raise ValueError(f"Config value must be JSON serializable, got {type(value)} for {key}")
        return v


class RunStatus(str, Enum):
    """Valid run statuses."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class RunInfo(BaseModel):
    """Type-safe run information."""
    name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    status: RunStatus = RunStatus.RUNNING


class MetricInterface:
    """Interface for metric logging with type safety."""
    
    def log(self, metrics: Union[Dict[str, Union[float, int, bool, str]], MetricDict], step: Optional[int] = None) -> None:
        """Log metrics with type validation."""
        if isinstance(metrics, dict):
            metric_dict = MetricDict(metrics=metrics, step=step)
        else:
            metric_dict = metrics
        
        # Validate and log
        self._log_metrics(metric_dict.metrics, metric_dict.step)
    
    def log_metric(self, name: str, value: Union[float, int, bool, str], step: Optional[int] = None) -> None:
        """Log a single metric with type validation."""
        metric_value = MetricValue(value=value, step=step)
        self.log({name: metric_value.value}, step)
    
    def log_config(self, config: Union[Dict[str, Any], ConfigDict]) -> None:
        """Log configuration with type validation."""
        if isinstance(config, dict):
            config_dict = ConfigDict(config=config)
        else:
            config_dict = config
        
        # Validate and log
        self._log_config(config_dict.config)
    
    def log_artifact(self, artifact: Union[str, ArtifactInfo]) -> None:
        """Log artifact with type validation."""
        if isinstance(artifact, str):
            artifact_info = ArtifactInfo(file_path=artifact)
        else:
            artifact_info = artifact
        
        # Validate and log
        self._log_artifact(artifact_info)
    
    def finish(self, status: RunStatus = RunStatus.COMPLETED) -> None:
        """Finish the run with type validation."""
        self._finish_run(status)
    
    # Abstract methods to be implemented by concrete classes
    def _log_metrics(self, metrics: Dict[str, Union[float, int, bool, str]], step: Optional[int]) -> None:
        """Abstract method for logging metrics."""
        raise NotImplementedError
    
    def _log_config(self, config: Dict[str, Any]) -> None:
        """Abstract method for logging configuration."""
        raise NotImplementedError
    
    def _log_artifact(self, artifact: ArtifactInfo) -> None:
        """Abstract method for logging artifacts."""
        raise NotImplementedError
    
    def _finish_run(self, status: RunStatus) -> None:
        """Abstract method for finishing the run."""
        raise NotImplementedError


class AIManagerInterface:
    """Interface for AI Manager with type safety."""
    
    def run(
        self,
        name: Optional[str] = None,
        config: Optional[Union[Dict[str, Any], ConfigDict]] = None,
        tags: Optional[List[str]] = None
    ) -> MetricInterface:
        """Start a new run with type validation."""
        run_info = RunInfo(
            name=name or f"run_{datetime.now().timestamp()}",
            config=config.dict() if isinstance(config, ConfigDict) else config or {},
            tags=tags or []
        )
        
        return self._create_run(run_info)
    
    # Abstract method to be implemented by concrete classes
    def _create_run(self, run_info: RunInfo) -> MetricInterface:
        """Abstract method for creating a run."""
        raise NotImplementedError 