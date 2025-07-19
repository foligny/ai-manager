"""
Run class for logging metrics and artifacts during training.
"""

import os
import json
import time
from typing import Dict, Any, Optional
import requests


class Run:
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
        self.config = {}
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for the current run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step (optional)
        """
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
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a single metric.
        
        Args:
            name: Name of the metric
            value: Value of the metric
            step: Training step (optional)
        """
        self.log({name: value}, step)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration for the run.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
        
        response = self.session.put(
            f"{self.api_url}/runs/{self.run_id}",
            json={"config": self.config}
        )
        response.raise_for_status()
    
    def log_artifact(self, file_path: str, name: Optional[str] = None, artifact_type: str = "other"):
        """Log an artifact (file) for the run.
        
        Args:
            file_path: Path to the file to upload
            name: Name for the artifact (defaults to filename)
            artifact_type: Type of artifact (model, data, config, other)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if name is None:
            name = os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            files = {'file': (name, f, 'application/octet-stream')}
            data = {
                'name': name,
                'type': artifact_type,
                'metadata': json.dumps({})
            }
            
            response = self.session.post(
                f"{self.api_url}/runs/{self.run_id}/artifacts",
                files=files,
                data=data
            )
            response.raise_for_status()
    
    def finish(self, status: str = "completed"):
        """Mark the run as finished.
        
        Args:
            status: Final status of the run (completed, failed, stopped)
        """
        response = self.session.put(
            f"{self.api_url}/runs/{self.run_id}",
            json={
                "status": status,
                "ended_at": time.time()
            }
        )
        response.raise_for_status()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - mark run as completed or failed."""
        if exc_type is None:
            self.finish("completed")
        else:
            self.finish("failed") 