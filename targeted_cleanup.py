#!/usr/bin/env python3
"""
Targeted cleanup script for test/debug projects.
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "admin123"

def get_token():
    """Get authentication token."""
    response = requests.post(
        f"{BASE_URL}/auth/login",
        data={"username": USERNAME, "password": PASSWORD}
    )
    response.raise_for_status()
    return response.json()["access_token"]

def delete_project(project_id, project_name):
    """Delete a project and all its runs."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"🗑️  Deleting project {project_id}: '{project_name}'...")
    
    # First, delete all runs in the project
    response = requests.get(f"{BASE_URL}/runs/?project_id={project_id}", headers=headers)
    if response.status_code == 200:
        runs = response.json()
        for run in runs:
            run_id = run['id']
            run_name = run['name']
            status = run['status']
            print(f"  - Deleting run {run_id}: '{run_name}' ({status})...")
            delete_response = requests.delete(f"{BASE_URL}/runs/{run_id}", headers=headers)
            if delete_response.status_code == 200:
                print(f"    ✅ Run {run_id} deleted")
            else:
                print(f"    ❌ Failed to delete run {run_id}")
    
    # Then delete the project
    delete_response = requests.delete(f"{BASE_URL}/projects/{project_id}", headers=headers)
    if delete_response.status_code == 200:
        print(f"  ✅ Project {project_id} deleted")
        return True
    else:
        print(f"  ❌ Failed to delete project {project_id}")
        return False

def cleanup_test_projects():
    """Clean up test and debug projects."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get all projects
    response = requests.get(f"{BASE_URL}/projects/", headers=headers)
    response.raise_for_status()
    projects = response.json()
    
    # Define test/debug project patterns
    test_patterns = [
        'test',
        'debug',
        'simple_test',
        'db_test'
    ]
    
    test_projects = []
    productive_projects = []
    
    for project in projects:
        project_id = project['id']
        project_name = project['name']
        
        # Check if it's a test/debug project
        is_test = any(pattern in project_name.lower() for pattern in test_patterns)
        
        if is_test:
            test_projects.append(project)
        else:
            productive_projects.append(project)
    
    print("🎯 TARGETED CLEANUP ANALYSIS")
    print("=" * 50)
    
    print("\n📋 PRODUCTIVE PROJECTS (KEEP):")
    for project in productive_projects:
        print(f"  ✅ Project {project['id']}: '{project['name']}'")
    
    print(f"\n🗑️  TEST/DEBUG PROJECTS ({len(test_projects)} found):")
    for i, project in enumerate(test_projects, 1):
        print(f"  {i}. Project {project['id']}: '{project['name']}'")
    
    if not test_projects:
        print("\n🎉 No test projects to clean up!")
        return
    
    print(f"\n🧹 CLEANUP OPTIONS")
    print("=" * 50)
    print("1. Delete all test/debug projects")
    print("2. Delete specific test projects")
    print("3. Just show analysis (no deletion)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🗑️  DELETING ALL TEST PROJECTS")
        print("=" * 50)
        for project in test_projects:
            delete_project(project['id'], project['name'])
    
    elif choice == "2":
        print("\n🗑️  SELECTIVE DELETION")
        print("=" * 50)
        for i, project in enumerate(test_projects, 1):
            print(f"{i}. Project {project['id']}: '{project['name']}'")
        
        try:
            indices = input("\nEnter project numbers to delete (comma-separated): ").strip()
            indices = [int(x.strip()) - 1 for x in indices.split(",")]
            
            for idx in indices:
                if 0 <= idx < len(test_projects):
                    project = test_projects[idx]
                    delete_project(project['id'], project['name'])
                else:
                    print(f"Invalid index: {idx + 1}")
        except ValueError:
            print("Invalid input format")
    
    elif choice == "3":
        print("\n📋 Analysis complete. No projects deleted.")
    
    else:
        print("Invalid choice")

def show_project_summary():
    """Show a summary of all projects."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/projects/", headers=headers)
    response.raise_for_status()
    projects = response.json()
    
    print("📊 PROJECT SUMMARY")
    print("=" * 50)
    
    total_runs = 0
    completed_runs = 0
    failed_runs = 0
    running_runs = 0
    
    for project in projects:
        project_id = project['id']
        project_name = project['name']
        
        # Get runs for this project
        response = requests.get(f"{BASE_URL}/runs/?project_id={project_id}", headers=headers)
        if response.status_code == 200:
            runs = response.json()
            project_runs = len(runs)
            total_runs += project_runs
            
            # Count run statuses
            for run in runs:
                status = run['status']
                if status == 'completed':
                    completed_runs += 1
                elif status == 'failed':
                    failed_runs += 1
                elif status == 'running':
                    running_runs += 1
            
            print(f"🔹 Project {project_id}: '{project_name}' - {project_runs} runs")
    
    print(f"\n📈 OVERALL STATISTICS")
    print("=" * 50)
    print(f"Total projects: {len(projects)}")
    print(f"Total runs: {total_runs}")
    print(f"Completed runs: {completed_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Running runs: {running_runs}")
    print(f"Success rate: {completed_runs/(completed_runs+failed_runs)*100:.1f}%" if (completed_runs+failed_runs) > 0 else "Success rate: N/A")

if __name__ == "__main__":
    print("🎯 AI Manager Targeted Cleanup Tool")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Show project summary")
        print("2. Clean up test/debug projects")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            show_project_summary()
        elif choice == "2":
            cleanup_test_projects()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice") 