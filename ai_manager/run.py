"""
Run class for logging metrics and artifacts during training.
"""

import os
import json
import time
from typing import Dict, Any, Optional, Union
import requests
from .types import MetricInterface, MetricDict, ConfigDict, ArtifactInfo, RunStatus, MetricValue


class Run(MetricInterface):
    """Represents a training run for logging metrics and artifacts."""
    
    def __init__(self, run_id: int, api_url: str, session: requests.Session):
        """Initialize a run.
        
        Args:
            run_id: ID of the run
            api_url: URL of the API
            session: Requests session with authentication
        """
        self.run_id = run_id
        self.api_url = api_url
        self.session = session
        self.config: Dict[str, Any] = {}
    
    def _log_metrics(self, metrics: Dict[str, Union[float, int, bool, str]], step: Optional[int]) -> None:
        """Log metrics for the current run (implements MetricInterface)."""
        if step is None:
            step = int(time.time())
        
        response = self.session.post(
            f"{self.api_url}/metrics/{self.run_id}/batch",
            json={
                "metrics": metrics,
                "step": step
            }
        )
        response.raise_for_status()
    
    def _log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration for the run (implements MetricInterface)."""
        self.config.update(config)
        
        response = self.session.put(
            f"{self.api_url}/runs/{self.run_id}",
            json={"config": self.config}
        )
        response.raise_for_status()
    
    def _log_artifact(self, artifact: ArtifactInfo) -> None:
        """Log an artifact (implements MetricInterface)."""
        if not os.path.exists(artifact.file_path):
            raise FileNotFoundError(f"File not found: {artifact.file_path}")
        
        name = artifact.name or os.path.basename(artifact.file_path)
        
        with open(artifact.file_path, 'rb') as f:
            files = {'file': (name, f, 'application/octet-stream')}
            data = {
                'name': name,
                'type': artifact.artifact_type.value,
                'metadata': json.dumps(artifact.metadata)
            }
            
            response = self.session.post(
                f"{self.api_url}/runs/{self.run_id}/artifacts",
                files=files,
                data=data
            )
            response.raise_for_status()
    
    def _finish_run(self, status: RunStatus) -> None:
        """Finish the run (implements MetricInterface)."""
        response = self.session.put(
            f"{self.api_url}/runs/{self.run_id}",
            json={
                "status": status.value,
                "ended_at": time.time()
            }
        )
        response.raise_for_status()
    
    # Note: Legacy methods removed to avoid conflicts with interface
    # Use the interface methods directly for type safety
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - mark run as completed or failed."""
        if exc_type is None:
            self.finish("completed")
        else:
            self.finish("failed") 