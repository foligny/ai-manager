#!/usr/bin/env python3
"""
AI Manager Test Agent
A simple script to test the AI Manager with sample inputs and see outputs on screen.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List

class AIManagerTestAgent:
    """Test agent for AI Manager."""
    
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
    
    def create_test_project(self, name: str, description: str, tags: List[str] = None) -> Dict[str, Any]:
        """Create a test project with tags."""
        project_data = {
            "name": name,
            "description": description,
            "is_public": True,
            "tags": tags or []
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/projects/",
                json=project_data
            )
            if response.status_code == 200:
                project = response.json()
                print(f"✅ Created project: {project['name']} (ID: {project['id']})")
                if project.get('tags'):
                    print(f"   Tags: {', '.join(project['tags'])}")
                return project
            else:
                print(f"❌ Failed to create project: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error creating project: {e}")
            return None
    
    def create_test_run(self, project_id: int, name: str, tags: List[str] = None) -> Dict[str, Any]:
        """Create a test run with tags."""
        run_data = {
            "name": name,
            "config": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "model_type": "cnn"
            },
            "tags": tags or []
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/projects/{project_id}/runs",
                json=run_data
            )
            if response.status_code == 200:
                run = response.json()
                print(f"✅ Created run: {run['name']} (ID: {run['id']})")
                if run.get('tags'):
                    print(f"   Tags: {', '.join(run['tags'])}")
                return run
            else:
                print(f"❌ Failed to create run: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error creating run: {e}")
            return None
    
    def log_metrics(self, run_id: int, metrics: Dict[str, float], step: int = 0):
        """Log metrics for a run."""
        try:
            response = self.session.post(
                f"{self.base_url}/runs/{run_id}/metrics",
                json={
                    "metrics": metrics,
                    "step": step
                }
            )
            if response.status_code == 200:
                print(f"✅ Logged metrics for step {step}: {metrics}")
            else:
                print(f"❌ Failed to log metrics: {response.status_code}")
        except Exception as e:
            print(f"❌ Error logging metrics: {e}")
    
    def simulate_training(self, run_id: int, epochs: int = 10):
        """Simulate a training process with realistic metrics."""
        print(f"\n🎯 Simulating training for run {run_id}...")
        
        for epoch in range(epochs):
            # Simulate realistic training metrics
            train_loss = 2.0 * (0.9 ** epoch) + 0.1  # Decreasing loss
            val_loss = train_loss + 0.2 + (epoch * 0.01)  # Slight overfitting
            train_acc = 0.3 + (0.6 * epoch / epochs)  # Increasing accuracy
            val_acc = train_acc - 0.1  # Slightly lower validation accuracy
            
            metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_accuracy": round(val_acc, 4),
                "learning_rate": 0.001
            }
            
            self.log_metrics(run_id, metrics, epoch)
            time.sleep(0.5)  # Small delay to see progress
        
        print(f"✅ Training simulation completed for run {run_id}")
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects."""
        try:
            response = self.session.get(f"{self.base_url}/projects/")
            if response.status_code == 200:
                projects = response.json()
                print(f"\n📁 Found {len(projects)} projects:")
                for project in projects:
                    tags_str = f" [{', '.join(project.get('tags', []))}]" if project.get('tags') else ""
                    print(f"   • {project['name']} (ID: {project['id']}){tags_str}")
                return projects
            else:
                print(f"❌ Failed to list projects: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error listing projects: {e}")
            return []
    
    def list_runs(self, project_id: int) -> List[Dict[str, Any]]:
        """List runs for a project."""
        try:
            response = self.session.get(f"{self.base_url}/runs/?project_id={project_id}")
            if response.status_code == 200:
                runs = response.json()
                print(f"\n🏃 Found {len(runs)} runs for project {project_id}:")
                for run in runs:
                    tags_str = f" [{', '.join(run.get('tags', []))}]" if run.get('tags') else ""
                    print(f"   • {run['name']} (ID: {run['id']}) - {run['status']}{tags_str}")
                return runs
            else:
                print(f"❌ Failed to list runs: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error listing runs: {e}")
            return []
    
    def get_metrics(self, run_id: int) -> List[Dict[str, Any]]:
        """Get metrics for a run."""
        try:
            response = self.session.get(f"{self.base_url}/metrics/{run_id}")
            if response.status_code == 200:
                metrics = response.json()
                print(f"\n📊 Found {len(metrics)} metrics for run {run_id}")
                return metrics
            else:
                print(f"❌ Failed to get metrics: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error getting metrics: {e}")
            return []

def main():
    """Main test function."""
    print("🤖 AI Manager Test Agent")
    print("=" * 50)
    
    # Initialize test agent
    agent = AIManagerTestAgent()
    
    # Test 1: Login
    print("\n1️⃣ Testing Login...")
    if not agent.login():
        print("❌ Cannot proceed without login")
        return
    
    # Test 2: Create demo project with tags (or get existing)
    print("\n2️⃣ Creating Demo Project with Tags...")
    demo_project = agent.create_test_project(
        name="demo_image_classifier",
        description="A demo project for testing image classification with various models",
        tags=["demo", "image-classification", "cnn", "pytorch", "computer-vision"]
    )
    
    if not demo_project:
        print("📁 Project already exists, getting existing project...")
        projects = agent.list_projects()
        demo_project = next((p for p in projects if p['name'] == "demo_image_classifier"), None)
        if not demo_project:
            print("❌ Failed to get demo project")
            return
        print(f"✅ Using existing project: {demo_project['name']} (ID: {demo_project['id']})")
    
    # Test 3: Create runs with different tags
    print("\n3️⃣ Creating Test Runs with Tags...")
    
    # Run 1: Baseline model
    run1 = agent.create_test_run(
        project_id=demo_project['id'],
        name="baseline_cnn",
        tags=["baseline", "cnn", "small-model"]
    )
    
    # Run 2: Large model
    run2 = agent.create_test_run(
        project_id=demo_project['id'],
        name="large_resnet",
        tags=["large-model", "resnet", "high-accuracy"]
    )
    
    # Run 3: Experiment
    run3 = agent.create_test_run(
        project_id=demo_project['id'],
        name="experiment_attention",
        tags=["experiment", "attention", "research"]
    )
    
    # Test 4: Simulate training for each run
    print("\n4️⃣ Simulating Training...")
    if run1:
        agent.simulate_training(run1['id'], epochs=5)
    
    if run2:
        agent.simulate_training(run2['id'], epochs=8)
    
    if run3:
        agent.simulate_training(run3['id'], epochs=3)
    
    # Test 5: List everything
    print("\n5️⃣ Listing All Projects and Runs...")
    agent.list_projects()
    
    if demo_project:
        agent.list_runs(demo_project['id'])
    
    # Test 6: Show metrics
    print("\n6️⃣ Showing Metrics...")
    if run1:
        agent.get_metrics(run1['id'])
    
    print("\n✅ Test completed! Check the web dashboard at http://localhost:8000")
    print("   You should see the demo project with tags and training runs.")

if __name__ == "__main__":
    main() 