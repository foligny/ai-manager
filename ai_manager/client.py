"""
AI Manager client for easy integration with training scripts.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List, Union
import requests
from .run import Run
from .types import AIManagerInterface, RunInfo, ConfigDict, MetricInterface


class AIManager(AIManagerInterface):
    """Main client class for AI Manager with type safety."""
    
    def __init__(
        self,
        project_name: str,
        api_url: str = "http://localhost:8000",
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None
    ):
        """Initialize AI Manager client.
        
        Args:
            project_name: Name of the project
            api_url: URL of the AI Manager API
            username: Username for authentication
            password: Password for authentication
            token: JWT token for authentication
        """
        self.project_name = project_name
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
        
        # Set up authentication
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        elif username and password:
            self._authenticate(username, password)
        else:
            # Try to get token from environment
            token = os.getenv("AI_MANAGER_TOKEN")
            if token:
                self.session.headers.update({"Authorization": f"Bearer {token}"})
    
    def _authenticate(self, username: str, password: str):
        """Authenticate with the API."""
        response = self.session.post(
            f"{self.api_url}/auth/login",
            data={"username": username, "password": password}
        )
        response.raise_for_status()
        
        token_data = response.json()
        self.session.headers.update({
            "Authorization": f"Bearer {token_data['access_token']}"
        })
    
    def _get_or_create_project(self) -> int:
        """Get or create the project."""
        # Try to get existing project
        response = self.session.get(f"{self.api_url}/projects/")
        response.raise_for_status()
        
        projects = response.json()
        for project in projects:
            if project["name"] == self.project_name:
                return project["id"]
        
        # Create new project
        response = self.session.post(
            f"{self.api_url}/projects/",
            json={
                "name": self.project_name,
                "description": "",
                "is_public": False
            }
        )
        response.raise_for_status()
        
        project = response.json()
        return project["id"]
    
    def _create_run(self, run_info: RunInfo) -> MetricInterface:
        """Create a new run (implements AIManagerInterface)."""
        project_id = self._get_or_create_project()
        
        response = self.session.post(
            f"{self.api_url}/runs/",
            params={"project_id": project_id},
            json={
                "name": run_info.name,
                "config": run_info.config,
                "tags": run_info.tags
            }
        )
        response.raise_for_status()
        
        run_data = response.json()
        return Run(
            run_id=run_data["id"],
            api_url=self.api_url,
            session=self.session
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass 