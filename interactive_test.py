#!/usr/bin/env python3
"""
Interactive AI Manager Test
Allows you to input your own test data and see outputs on screen.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List

class InteractiveAITester:
    """Interactive test interface for AI Manager."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.token = None
        
    def login(self) -> bool:
        """Interactive login."""
        print("\n🔐 Login to AI Manager")
        print("-" * 30)
        
        username = input("Username (default: admin): ").strip() or "admin"
        password = input("Password (default: admin123): ").strip() or "admin123"
        
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
    
    def create_project_interactive(self) -> Dict[str, Any]:
        """Create a project with user input."""
        print("\n📁 Create New Project")
        print("-" * 30)
        
        name = input("Project name: ").strip()
        if not name:
            print("❌ Project name is required")
            return None
            
        description = input("Description (optional): ").strip()
        
        print("Enter tags (comma-separated, e.g., demo,cnn,pytorch): ")
        tags_input = input("Tags: ").strip()
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()] if tags_input else []
        
        project_data = {
            "name": name,
            "description": description,
            "is_public": True,
            "tags": tags
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
    
    def create_run_interactive(self, project_id: int) -> Dict[str, Any]:
        """Create a run with user input."""
        print(f"\n🏃 Create New Run for Project {project_id}")
        print("-" * 40)
        
        name = input("Run name: ").strip()
        if not name:
            print("❌ Run name is required")
            return None
        
        print("Enter tags (comma-separated): ")
        tags_input = input("Tags: ").strip()
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()] if tags_input else []
        
        # Get configuration
        print("\nConfiguration:")
        lr = input("Learning rate (default: 0.001): ").strip() or "0.001"
        batch_size = input("Batch size (default: 32): ").strip() or "32"
        epochs = input("Epochs (default: 100): ").strip() or "100"
        model_type = input("Model type (default: cnn): ").strip() or "cnn"
        
        run_data = {
            "name": name,
            "config": {
                "learning_rate": float(lr),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "model_type": model_type
            },
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
                if run.get('tags'):
                    print(f"   Tags: {', '.join(run['tags'])}")
                return run
            else:
                print(f"❌ Failed to create run: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error creating run: {e}")
            return None
    
    def log_metrics_interactive(self, run_id: int):
        """Log metrics with user input."""
        print(f"\n📊 Log Metrics for Run {run_id}")
        print("-" * 35)
        
        while True:
            print("\nEnter metrics (or 'done' to finish):")
            print("Format: metric_name=value (e.g., loss=0.5, accuracy=0.85)")
            
            metrics_input = input("Metrics: ").strip()
            if metrics_input.lower() == 'done':
                break
            
            try:
                # Parse metrics input
                metrics = {}
                for pair in metrics_input.split(','):
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        metrics[key.strip()] = float(value.strip())
                
                if not metrics:
                    print("❌ No valid metrics provided")
                    continue
                
                step = input("Step (default: 0): ").strip()
                step = int(step) if step else 0
                
                # Log the metrics
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
                    
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
            except Exception as e:
                print(f"❌ Error logging metrics: {e}")
    
    def simulate_training_interactive(self, run_id: int):
        """Simulate training with user input."""
        print(f"\n🎯 Simulate Training for Run {run_id}")
        print("-" * 40)
        
        epochs = input("Number of epochs (default: 10): ").strip()
        epochs = int(epochs) if epochs else 10
        
        delay = input("Delay between epochs in seconds (default: 0.5): ").strip()
        delay = float(delay) if delay else 0.5
        
        print(f"\n🎯 Simulating {epochs} epochs with {delay}s delay...")
        
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
            
            time.sleep(delay)
        
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

def main():
    """Main interactive test function."""
    print("🤖 Interactive AI Manager Test")
    print("=" * 50)
    
    # Initialize tester
    tester = InteractiveAITester()
    
    # Login
    if not tester.login():
        print("❌ Cannot proceed without login")
        return
    
    while True:
        print("\n" + "=" * 50)
        print("🎯 Interactive AI Manager Test Menu")
        print("=" * 50)
        print("1. List all projects")
        print("2. Create new project")
        print("3. List runs for a project")
        print("4. Create new run")
        print("5. Log metrics manually")
        print("6. Simulate training")
        print("7. Exit")
        print("-" * 50)
        
        choice = input("Choose an option (1-7): ").strip()
        
        if choice == "1":
            tester.list_projects()
            
        elif choice == "2":
            project = tester.create_project_interactive()
            if project:
                print(f"✅ Project created with ID: {project['id']}")
                
        elif choice == "3":
            project_id = input("Enter project ID: ").strip()
            if project_id:
                tester.list_runs(int(project_id))
                
        elif choice == "4":
            project_id = input("Enter project ID: ").strip()
            if project_id:
                run = tester.create_run_interactive(int(project_id))
                if run:
                    print(f"✅ Run created with ID: {run['id']}")
                    
        elif choice == "5":
            run_id = input("Enter run ID: ").strip()
            if run_id:
                tester.log_metrics_interactive(int(run_id))
                
        elif choice == "6":
            run_id = input("Enter run ID: ").strip()
            if run_id:
                tester.simulate_training_interactive(int(run_id))
                
        elif choice == "7":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 