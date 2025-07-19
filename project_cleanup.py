#!/usr/bin/env python3
"""
Project cleanup and analysis script for AI Manager.
"""

import requests
import json
from datetime import datetime

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

def analyze_projects():
    """Analyze all projects and their runs."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get all projects
    response = requests.get(f"{BASE_URL}/projects/", headers=headers)
    response.raise_for_status()
    projects = response.json()
    
    print("🔍 PROJECT ANALYSIS")
    print("=" * 50)
    
    valid_projects = []
    invalid_projects = []
    
    for project in projects:
        project_id = project['id']
        project_name = project['name']
        created_at = project['created_at']
        
        # Get runs for this project
        response = requests.get(f"{BASE_URL}/runs/?project_id={project_id}", headers=headers)
        if response.status_code == 200:
            runs = response.json()
            run_count = len(runs)
            
            # Check if project has any runs
            if run_count > 0:
                valid_projects.append({
                    'id': project_id,
                    'name': project_name,
                    'runs': run_count,
                    'created_at': created_at,
                    'status': 'VALID'
                })
                print(f"✅ Project {project_id}: '{project_name}' - {run_count} runs")
            else:
                invalid_projects.append({
                    'id': project_id,
                    'name': project_name,
                    'runs': 0,
                    'created_at': created_at,
                    'status': 'EMPTY'
                })
                print(f"⚠️  Project {project_id}: '{project_name}' - NO RUNS")
        else:
            invalid_projects.append({
                'id': project_id,
                'name': project_name,
                'runs': 0,
                'created_at': created_at,
                'status': 'ERROR'
            })
            print(f"❌ Project {project_id}: '{project_name}' - ERROR")
    
    print("\n📊 SUMMARY")
    print("=" * 50)
    print(f"Total projects: {len(projects)}")
    print(f"Valid projects: {len(valid_projects)}")
    print(f"Invalid projects: {len(invalid_projects)}")
    
    return valid_projects, invalid_projects

def delete_project(project_id, project_name):
    """Delete a project."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"🗑️  Deleting project {project_id}: '{project_name}'...")
    
    # First, delete all runs in the project
    response = requests.get(f"{BASE_URL}/runs/?project_id={project_id}", headers=headers)
    if response.status_code == 200:
        runs = response.json()
        for run in runs:
            run_id = run['id']
            print(f"  - Deleting run {run_id}...")
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

def cleanup_invalid_projects():
    """Clean up invalid projects."""
    valid_projects, invalid_projects = analyze_projects()
    
    if not invalid_projects:
        print("\n🎉 No invalid projects to clean up!")
        return
    
    print(f"\n🧹 CLEANUP OPTIONS")
    print("=" * 50)
    print("1. Delete all invalid projects")
    print("2. Delete specific projects")
    print("3. Just show analysis (no deletion)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🗑️  DELETING ALL INVALID PROJECTS")
        print("=" * 50)
        for project in invalid_projects:
            delete_project(project['id'], project['name'])
    
    elif choice == "2":
        print("\n🗑️  SELECTIVE DELETION")
        print("=" * 50)
        for i, project in enumerate(invalid_projects, 1):
            print(f"{i}. Project {project['id']}: '{project['name']}' ({project['status']})")
        
        try:
            indices = input("\nEnter project numbers to delete (comma-separated): ").strip()
            indices = [int(x.strip()) - 1 for x in indices.split(",")]
            
            for idx in indices:
                if 0 <= idx < len(invalid_projects):
                    project = invalid_projects[idx]
                    delete_project(project['id'], project['name'])
                else:
                    print(f"Invalid index: {idx + 1}")
        except ValueError:
            print("Invalid input format")
    
    elif choice == "3":
        print("\n📋 Analysis complete. No projects deleted.")
    
    else:
        print("Invalid choice")

def show_project_details():
    """Show detailed information about all projects."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/projects/", headers=headers)
    response.raise_for_status()
    projects = response.json()
    
    print("📋 DETAILED PROJECT INFORMATION")
    print("=" * 60)
    
    for project in projects:
        project_id = project['id']
        project_name = project['name']
        description = project.get('description', 'No description')
        created_at = project['created_at']
        is_public = project.get('is_public', False)
        
        print(f"\n🔹 Project {project_id}: '{project_name}'")
        print(f"   Description: {description}")
        print(f"   Created: {created_at}")
        print(f"   Public: {is_public}")
        
        # Get runs for this project
        response = requests.get(f"{BASE_URL}/runs/?project_id={project_id}", headers=headers)
        if response.status_code == 200:
            runs = response.json()
            print(f"   Runs: {len(runs)}")
            
            for run in runs:
                run_id = run['id']
                run_name = run['name']
                status = run['status']
                started_at = run['started_at']
                ended_at = run.get('ended_at', 'Still running')
                
                print(f"     - Run {run_id}: '{run_name}' ({status})")
                print(f"       Started: {started_at}")
                if ended_at != 'Still running':
                    print(f"       Ended: {ended_at}")

if __name__ == "__main__":
    print("🧹 AI Manager Project Cleanup Tool")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Analyze projects")
        print("2. Show detailed project information")
        print("3. Clean up invalid projects")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            analyze_projects()
        elif choice == "2":
            show_project_details()
        elif choice == "3":
            cleanup_invalid_projects()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice") 