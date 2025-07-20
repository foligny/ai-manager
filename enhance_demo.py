#!/usr/bin/env python3
"""
Enhance Demo Project
Adds tags to existing demo projects and creates additional demo runs.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List

class DemoEnhancer:
    """Enhances demo projects with tags and additional runs."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.token = None
        
    def login(self, username: str = "admin", password: str = "admin123") -> bool:
        """Login to AI Manager."""
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                data={"username": username, "password": password}
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                print(f"✅ Logged in successfully as {username}")
                return True
            else:
                print(f"❌ Login failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_projects(self) -> List[Dict[str, Any]]:
        """Get all projects."""
        try:
            response = self.session.get(f"{self.base_url}/projects/")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get projects: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error getting projects: {e}")
            return []
    
    def update_project_tags(self, project_id: int, tags: List[str]) -> bool:
        """Update project with tags."""
        try:
            response = self.session.put(
                f"{self.base_url}/projects/{project_id}",
                json={"tags": tags}
            )
            if response.status_code == 200:
                project = response.json()
                print(f"✅ Updated project {project['name']} with tags: {', '.join(tags)}")
                return True
            else:
                print(f"❌ Failed to update project: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error updating project: {e}")
            return False
    
    def create_demo_run(self, project_id: int, name: str, tags: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a demo run with tags."""
        run_data = {
            "name": name,
            "config": config,
            "tags": tags
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/projects/{project_id}/runs",
                json=run_data
            )
            if response.status_code == 200:
                run = response.json()
                print(f"✅ Created run: {run['name']} (ID: {run['id']})")
                print(f"   Tags: {', '.join(run['tags'])}")
                return run
            else:
                print(f"❌ Failed to create run: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error creating run: {e}")
            return None
    
    def simulate_training(self, run_id: int, epochs: int = 5):
        """Simulate training for a run."""
        print(f"🎯 Simulating training for run {run_id}...")
        
        for epoch in range(epochs):
            # Simulate realistic training metrics
            train_loss = 2.0 * (0.9 ** epoch) + 0.1
            val_loss = train_loss + 0.2 + (epoch * 0.01)
            train_acc = 0.3 + (0.6 * epoch / epochs)
            val_acc = train_acc - 0.1
            
            metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_accuracy": round(val_acc, 4),
                "learning_rate": 0.001
            }
            
            try:
                response = self.session.post(
                    f"{self.base_url}/runs/{run_id}/metrics",
                    json={
                        "metrics": metrics,
                        "step": epoch
                    }
                )
                
                if response.status_code == 200:
                    print(f"✅ Epoch {epoch}: loss={metrics['train_loss']:.4f}, acc={metrics['train_accuracy']:.4f}")
                else:
                    print(f"❌ Failed to log metrics for epoch {epoch}")
                    
            except Exception as e:
                print(f"❌ Error logging metrics: {e}")
            
            time.sleep(0.3)
        
        print(f"✅ Training simulation completed for run {run_id}")
    
    def enhance_demo_projects(self):
        """Enhance existing demo projects with tags and runs."""
        print("🎯 Enhancing Demo Projects")
        print("=" * 50)
        
        # Get existing projects
        projects = self.get_projects()
        print(f"Found {len(projects)} projects")
        
        for project in projects:
            print(f"\n📁 Processing project: {project['name']} (ID: {project['id']})")
            
            # Add tags based on project name
            if "demo" in project['name'].lower():
                if "image" in project['name'].lower():
                    tags = ["demo", "image-classification", "computer-vision", "pytorch", "cnn"]
                elif "text" in project['name'].lower():
                    tags = ["demo", "nlp", "text-processing", "transformer", "bert"]
                else:
                    tags = ["demo", "machine-learning", "experiment"]
                
                # Update project with tags
                self.update_project_tags(project['id'], tags)
                
                # Create additional demo runs
                self.create_demo_runs(project['id'], project['name'])
    
    def create_demo_runs(self, project_id: int, project_name: str):
        """Create demo runs for a project."""
        print(f"🏃 Creating demo runs for {project_name}...")
        
        # Define different run configurations based on project type
        if "image" in project_name.lower():
            runs = [
                {
                    "name": "baseline_cnn",
                    "tags": ["baseline", "cnn", "small-model"],
                    "config": {
                        "learning_rate": 0.001,
                        "batch_size": 32,
                        "epochs": 50,
                        "model_type": "cnn",
                        "optimizer": "adam"
                    }
                },
                {
                    "name": "resnet_experiment",
                    "tags": ["resnet", "large-model", "high-accuracy"],
                    "config": {
                        "learning_rate": 0.0001,
                        "batch_size": 16,
                        "epochs": 100,
                        "model_type": "resnet",
                        "optimizer": "adam"
                    }
                },
                {
                    "name": "attention_mechanism",
                    "tags": ["attention", "experiment", "research"],
                    "config": {
                        "learning_rate": 0.0005,
                        "batch_size": 8,
                        "epochs": 75,
                        "model_type": "transformer",
                        "optimizer": "adamw"
                    }
                }
            ]
        else:
            runs = [
                {
                    "name": "baseline_model",
                    "tags": ["baseline", "simple", "fast"],
                    "config": {
                        "learning_rate": 0.01,
                        "batch_size": 64,
                        "epochs": 30,
                        "model_type": "linear",
                        "optimizer": "sgd"
                    }
                },
                {
                    "name": "advanced_experiment",
                    "tags": ["advanced", "complex", "high-performance"],
                    "config": {
                        "learning_rate": 0.001,
                        "batch_size": 32,
                        "epochs": 80,
                        "model_type": "neural_network",
                        "optimizer": "adam"
                    }
                }
            ]
        
        # Create runs
        for run_config in runs:
            run = self.create_demo_run(
                project_id=project_id,
                name=run_config["name"],
                tags=run_config["tags"],
                config=run_config["config"]
            )
            
            if run:
                # Simulate some training
                self.simulate_training(run['id'], epochs=3)

def main():
    """Main function."""
    print("🎯 AI Manager Demo Enhancer")
    print("=" * 50)
    
    # Initialize enhancer
    enhancer = DemoEnhancer()
    
    # Login
    if not enhancer.login():
        print("❌ Cannot proceed without login")
        return
    
    # Enhance demo projects
    enhancer.enhance_demo_projects()
    
    print("\n✅ Demo enhancement completed!")
    print("Check the web dashboard at http://localhost:8000")
    print("You should see projects and runs with tags now.")

if __name__ == "__main__":
    main() 