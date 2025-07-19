#!/usr/bin/env python3
"""
Performance analysis script for AI Manager.
"""

import requests
import json
import matplotlib.pyplot as plt
import numpy as np
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

def get_run_metrics(run_id):
    """Get all metrics for a specific run."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/metrics/{run_id}", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get metrics for run {run_id}")
        return []

def get_run_details(run_id):
    """Get details for a specific run."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/runs/{run_id}", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get details for run {run_id}")
        return None

def analyze_run_performance(run_id):
    """Analyze performance for a specific run."""
    print(f"🔍 Analyzing Run {run_id}")
    print("=" * 50)
    
    # Get run details
    run_details = get_run_details(run_id)
    if not run_details:
        return
    
    print(f"Run Name: {run_details['name']}")
    print(f"Status: {run_details['status']}")
    print(f"Started: {run_details['started_at']}")
    if run_details.get('ended_at'):
        print(f"Ended: {run_details['ended_at']}")
    
    # Get metrics
    metrics = get_run_metrics(run_id)
    if not metrics:
        print("No metrics found for this run.")
        return
    
    # Group metrics by name
    metric_groups = {}
    for metric in metrics:
        name = metric['name']
        if name not in metric_groups:
            metric_groups[name] = []
        metric_groups[name].append(metric)
    
    print(f"\n📈 Performance Metrics:")
    print("-" * 30)
    
    for metric_name, metric_data in metric_groups.items():
        values = [m['value'] for m in metric_data]
        steps = [m['step'] for m in metric_data]
        
        if len(values) > 0:
            initial_value = values[0]
            final_value = values[-1]
            improvement = final_value - initial_value
            improvement_pct = (improvement / initial_value * 100) if initial_value != 0 else 0
            
            print(f"\n📊 {metric_name.upper()}:")
            print(f"   Initial: {initial_value:.4f}")
            print(f"   Final: {final_value:.4f}")
            print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
            print(f"   Min: {min(values):.4f}")
            print(f"   Max: {max(values):.4f}")
            print(f"   Steps: {len(values)}")
    
    return metric_groups

def compare_runs(run_ids):
    """Compare performance between multiple runs."""
    print("🔄 COMPARING RUNS")
    print("=" * 50)
    
    run_comparisons = {}
    
    for run_id in run_ids:
        run_details = get_run_details(run_id)
        if not run_details:
            continue
            
        metrics = get_run_metrics(run_id)
        if not metrics:
            continue
        
        # Get final values for each metric
        final_metrics = {}
        for metric in metrics:
            name = metric['name']
            if name not in final_metrics or metric['step'] > final_metrics[name]['step']:
                final_metrics[name] = metric
        
        run_comparisons[run_id] = {
            'name': run_details['name'],
            'status': run_details['status'],
            'final_metrics': final_metrics
        }
    
    # Compare final values
    if len(run_comparisons) < 2:
        print("Need at least 2 runs to compare.")
        return
    
    print(f"📊 Final Performance Comparison:")
    print("-" * 40)
    
    # Get all unique metric names
    all_metrics = set()
    for run_data in run_comparisons.values():
        all_metrics.update(run_data['final_metrics'].keys())
    
    for metric_name in sorted(all_metrics):
        print(f"\n📈 {metric_name.upper()}:")
        values = []
        for run_id, run_data in run_comparisons.items():
            if metric_name in run_data['final_metrics']:
                value = run_data['final_metrics'][metric_name]['value']
                values.append((run_id, value))
                print(f"   Run {run_id} ({run_data['name']}): {value:.4f}")
        
        if len(values) > 1:
            best_run = max(values, key=lambda x: x[1])
            worst_run = min(values, key=lambda x: x[1])
            improvement = best_run[1] - worst_run[1]
            print(f"   Best: Run {best_run[0]} ({best_run[1]:.4f})")
            print(f"   Worst: Run {worst_run[0]} ({worst_run[1]:.4f})")
            print(f"   Range: {improvement:.4f}")

def plot_training_curves(run_ids, metric_names=None):
    """Plot training curves for multiple runs."""
    if metric_names is None:
        metric_names = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric_name in enumerate(metric_names):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        for run_id in run_ids:
            metrics = get_run_metrics(run_id)
            if not metrics:
                continue
            
            # Filter metrics by name
            metric_data = [m for m in metrics if m['name'] == metric_name]
            if not metric_data:
                continue
            
            # Sort by step
            metric_data.sort(key=lambda x: x['step'])
            
            steps = [m['step'] for m in metric_data]
            values = [m['value'] for m in metric_data]
            
            run_details = get_run_details(run_id)
            run_name = run_details['name'] if run_details else f"Run {run_id}"
            
            ax.plot(steps, values, marker='o', label=f"{run_name} (Run {run_id})")
        
        ax.set_title(f'{metric_name.upper()} Over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel(metric_name.title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("📊 Training curves saved as 'training_curves.png'")
    plt.show()

def show_available_runs():
    """Show all available runs for analysis."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get all projects
    response = requests.get(f"{BASE_URL}/projects/", headers=headers)
    if response.status_code != 200:
        print("Failed to get projects")
        return []
    
    projects = response.json()
    all_runs = []
    
    for project in projects:
        project_id = project['id']
        response = requests.get(f"{BASE_URL}/runs/?project_id={project_id}", headers=headers)
        if response.status_code == 200:
            runs = response.json()
            for run in runs:
                run['project_name'] = project['name']
                all_runs.append(run)
    
    print("📋 AVAILABLE RUNS FOR ANALYSIS")
    print("=" * 50)
    
    for run in all_runs:
        status_icon = "✅" if run['status'] == 'completed' else "🔄" if run['status'] == 'running' else "❌"
        print(f"{status_icon} Run {run['id']}: '{run['name']}' ({run['status']}) - Project: {run['project_name']}")
    
    return all_runs

def main():
    """Main analysis function."""
    print("📊 AI Manager Performance Analysis")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Show available runs")
        print("2. Analyze specific run")
        print("3. Compare multiple runs")
        print("4. Plot training curves")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            show_available_runs()
        
        elif choice == "2":
            run_id = input("Enter run ID to analyze: ").strip()
            try:
                analyze_run_performance(int(run_id))
            except ValueError:
                print("Invalid run ID")
        
        elif choice == "3":
            run_ids_input = input("Enter run IDs to compare (comma-separated): ").strip()
            try:
                run_ids = [int(x.strip()) for x in run_ids_input.split(",")]
                compare_runs(run_ids)
            except ValueError:
                print("Invalid run IDs")
        
        elif choice == "4":
            run_ids_input = input("Enter run IDs to plot (comma-separated): ").strip()
            try:
                run_ids = [int(x.strip()) for x in run_ids_input.split(",")]
                plot_training_curves(run_ids)
            except ValueError:
                print("Invalid run IDs")
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 