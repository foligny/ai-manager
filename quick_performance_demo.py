#!/usr/bin/env python3
"""
Quick performance demo showing before/after improvements.
"""

import requests

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

def analyze_completed_runs():
    """Analyze all completed runs to show performance improvements."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get all projects
    response = requests.get(f"{BASE_URL}/projects/", headers=headers)
    projects = response.json()
    
    print("📊 PERFORMANCE IMPROVEMENTS ANALYSIS")
    print("=" * 60)
    
    for project in projects:
        project_id = project['id']
        project_name = project['name']
        
        # Get runs for this project
        response = requests.get(f"{BASE_URL}/runs/?project_id={project_id}", headers=headers)
        runs = response.json()
        
        completed_runs = [run for run in runs if run['status'] == 'completed']
        
        if completed_runs:
            print(f"\n🔹 Project: {project_name}")
            print("-" * 40)
            
            for run in completed_runs:
                run_id = run['id']
                run_name = run['name']
                
                # Get metrics for this run
                response = requests.get(f"{BASE_URL}/metrics/{run_id}", headers=headers)
                metrics = response.json()
                
                if metrics:
                    # Find key metrics
                    train_acc_metrics = [m for m in metrics if m['name'] == 'TRAIN_ACCURACY']
                    val_acc_metrics = [m for m in metrics if m['name'] == 'VAL_ACCURACY']
                    train_loss_metrics = [m for m in metrics if m['name'] == 'TRAIN_LOSS']
                    val_loss_metrics = [m for m in metrics if m['name'] == 'VAL_LOSS']
                    
                    print(f"\n  📈 Run {run_id}: '{run_name}'")
                    
                    if train_acc_metrics:
                        initial_acc = train_acc_metrics[0]['value']
                        final_acc = train_acc_metrics[-1]['value']
                        acc_improvement = final_acc - initial_acc
                        print(f"     Training Accuracy: {initial_acc:.1f}% → {final_acc:.1f}% ({acc_improvement:+.1f}%)")
                    
                    if val_acc_metrics:
                        initial_val_acc = val_acc_metrics[0]['value']
                        final_val_acc = val_acc_metrics[-1]['value']
                        val_acc_improvement = final_val_acc - initial_val_acc
                        print(f"     Validation Accuracy: {initial_val_acc:.1f}% → {final_val_acc:.1f}% ({val_acc_improvement:+.1f}%)")
                    
                    if train_loss_metrics:
                        initial_loss = train_loss_metrics[0]['value']
                        final_loss = train_loss_metrics[-1]['value']
                        loss_improvement = final_loss - initial_loss
                        print(f"     Training Loss: {initial_loss:.4f} → {final_loss:.4f} ({loss_improvement:+.4f})")
                    
                    if val_loss_metrics:
                        initial_val_loss = val_loss_metrics[0]['value']
                        final_val_loss = val_loss_metrics[-1]['value']
                        val_loss_improvement = final_val_loss - initial_val_loss
                        print(f"     Validation Loss: {initial_val_loss:.4f} → {final_val_loss:.4f} ({val_loss_improvement:+.4f})")

def compare_baseline_vs_improved():
    """Compare baseline vs improved runs."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    print("\n🔄 BASELINE vs IMPROVED COMPARISON")
    print("=" * 50)
    
    # Find sample_image_classifier project
    response = requests.get(f"{BASE_URL}/projects/", headers=headers)
    projects = response.json()
    
    sample_project = None
    for project in projects:
        if 'sample_image_classifier' in project['name']:
            sample_project = project
            break
    
    if not sample_project:
        print("Sample image classifier project not found.")
        return
    
    # Get runs for this project
    response = requests.get(f"{BASE_URL}/runs/?project_id={sample_project['id']}", headers=headers)
    runs = response.json()
    
    completed_runs = [run for run in runs if run['status'] == 'completed']
    
    if len(completed_runs) >= 3:
        print(f"📊 Comparing {len(completed_runs)} completed runs:")
        
        for run in completed_runs:
            run_id = run['id']
            run_name = run['name']
            
            # Get final metrics
            response = requests.get(f"{BASE_URL}/metrics/{run_id}", headers=headers)
            metrics = response.json()
            
            if metrics:
                # Find final accuracy
                final_acc_metrics = [m for m in metrics if m['name'] == 'FINAL_TEST_ACCURACY']
                if final_acc_metrics:
                    final_acc = final_acc_metrics[0]['value']
                    print(f"  ✅ {run_name}: {final_acc:.1f}% accuracy")
                else:
                    print(f"  ✅ {run_name}: No final accuracy recorded")

def show_training_summary():
    """Show a summary of training performance."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    print("\n📋 TRAINING PERFORMANCE SUMMARY")
    print("=" * 50)
    
    # Get all projects
    response = requests.get(f"{BASE_URL}/projects/", headers=headers)
    projects = response.json()
    
    total_runs = 0
    completed_runs = 0
    failed_runs = 0
    running_runs = 0
    
    for project in projects:
        project_id = project['id']
        project_name = project['name']
        
        # Get runs for this project
        response = requests.get(f"{BASE_URL}/runs/?project_id={project_id}", headers=headers)
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
        
        if project_runs > 0:
            print(f"🔹 {project_name}: {project_runs} runs")
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Total runs: {total_runs}")
    print(f"   Completed: {completed_runs}")
    print(f"   Failed: {failed_runs}")
    print(f"   Running: {running_runs}")
    
    if (completed_runs + failed_runs) > 0:
        success_rate = completed_runs / (completed_runs + failed_runs) * 100
        print(f"   Success rate: {success_rate:.1f}%")

if __name__ == "__main__":
    print("🚀 AI Manager Performance Demo")
    print("=" * 50)
    
    show_training_summary()
    analyze_completed_runs()
    compare_baseline_vs_improved()
    
    print("\n🎉 Performance analysis complete!")
    print("\n💡 Tips for viewing improvements:")
    print("   • Check the web dashboard at http://localhost:8000")
    print("   • Look for training curves showing loss/accuracy over time")
    print("   • Compare different runs to see which hyperparameters work best")
    print("   • Focus on validation accuracy for generalization performance") 