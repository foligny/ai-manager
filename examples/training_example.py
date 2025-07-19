"""
Example training script using AI Manager.

This script demonstrates how to integrate AI Manager with a training loop.
"""

import time
import random
import numpy as np
from ai_manager import AIManager


def simulate_training():
    """Simulate a training loop with metrics logging."""
    
    # Initialize AI Manager
    manager = AIManager(
        project_name="example_project",
        api_url="http://localhost:8000"
    )
    
    # Start a new run
    with manager.run(
        name="example_training_run",
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "model": "resnet50"
        },
        tags=["example", "demo"]
    ) as run:
        
        print("Starting training...")
        
        # Simulate training loop
        for epoch in range(100):
            # Simulate training metrics
            train_loss = 1.0 * np.exp(-epoch / 20) + random.uniform(0, 0.1)
            train_acc = 1.0 - 0.5 * np.exp(-epoch / 15) + random.uniform(-0.05, 0.05)
            val_loss = train_loss + random.uniform(0, 0.2)
            val_acc = train_acc - random.uniform(0, 0.1)
            
            # Log metrics
            run.log({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch": epoch
            }, step=epoch)
            
            # Log some additional metrics occasionally
            if epoch % 10 == 0:
                run.log({
                    "learning_rate": 0.001 * (0.95 ** (epoch // 10)),
                    "gpu_memory": random.uniform(0.7, 0.9),
                    "cpu_usage": random.uniform(0.3, 0.6)
                }, step=epoch)
            
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            time.sleep(0.1)  # Simulate training time
        
        # Log final model
        print("Training completed!")
        
        # In a real scenario, you would save your model here
        # run.log_artifact("model.pth", artifact_type="model")


def simulate_multiple_runs():
    """Simulate multiple training runs for comparison."""
    
    manager = AIManager(
        project_name="hyperparameter_tuning",
        api_url="http://localhost:8000"
    )
    
    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.1]
    
    for lr in learning_rates:
        with manager.run(
            name=f"lr_{lr}",
            config={
                "learning_rate": lr,
                "batch_size": 32,
                "epochs": 50
            },
            tags=["hyperparameter_tuning", f"lr_{lr}"]
        ) as run:
            
            print(f"Training with learning rate: {lr}")
            
            for epoch in range(50):
                # Simulate different performance based on learning rate
                if lr == 0.001:
                    loss = 1.0 * np.exp(-epoch / 15) + random.uniform(0, 0.05)
                elif lr == 0.01:
                    loss = 1.0 * np.exp(-epoch / 12) + random.uniform(0, 0.08)
                else:  # lr = 0.1
                    loss = 1.0 * np.exp(-epoch / 8) + random.uniform(0, 0.15)
                
                accuracy = 1.0 - loss + random.uniform(-0.05, 0.05)
                
                run.log({
                    "loss": loss,
                    "accuracy": accuracy,
                    "epoch": epoch
                }, step=epoch)
                
                time.sleep(0.05)
            
            print(f"Completed training with lr={lr}")


if __name__ == "__main__":
    print("AI Manager Training Example")
    print("=" * 40)
    
    # Run single training example
    print("\n1. Single Training Run")
    simulate_training()
    
    # Run multiple training runs for comparison
    print("\n2. Multiple Training Runs (Hyperparameter Tuning)")
    simulate_multiple_runs()
    
    print("\nTraining examples completed!")
    print("Check the dashboard at http://localhost:8000 to see the results.") 