"""
Main training script that integrates with AI Manager for monitoring and logging.
This script demonstrates how to use the AI Manager with a real training task.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add parent directory to path to import AI Manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_manager import AIManager, RunStatus, ArtifactType, ArtifactInfo
from data_generator import generate_synthetic_data, create_data_loaders, save_data_info
from model import create_model, ModelTrainer


def train_with_ai_manager():
    """Train the model with AI Manager integration."""
    
    print("🚀 Starting AI Manager Integration Test")
    print("=" * 50)
    
    # Initialize AI Manager
    manager = AIManager(
        project_name="sample_image_classifier",
        api_url="http://localhost:8000",
        username="admin",
        password="admin123"
    )
    
    # Generate data
    print("📊 Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=100, image_size=32, n_classes=3)
    save_data_info(X, y, 'data_info.json')
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(X, y, batch_size=8)
    
    # Create model
    print("🏗️ Creating model...")
    model = create_model()
    trainer = ModelTrainer(model, device='cpu')
    
    # Log model information
    model_info = model.get_model_info()
    print(f"Model created: {model_info['architecture']}")
    print(f"Parameters: {model_info['total_parameters']:,}")
    print(f"Size: {model_info['model_size_mb']:.2f} MB")
    
    # Start AI Manager run
    with manager.run(
        name=f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_architecture": model_info['architecture'],
            "total_parameters": model_info['total_parameters'],
            "model_size_mb": model_info['model_size_mb'],
            "input_size": model_info['input_size'],
            "num_classes": model_info['num_classes'],
            "batch_size": 8,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "early_stopping_patience": 10,
            "max_epochs": 50,
            "data_samples": len(X),
            "train_samples": len(train_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "device": "cpu"
        }
    ) as run:
        
        print("🎯 Starting training with AI Manager monitoring...")
        
        # Training loop with AI Manager logging
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 10
        max_epochs = 50
        
        for epoch in range(max_epochs):
            # Train epoch
            train_loss, train_acc = trainer.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = trainer.validate(test_loader)
            
            # Update learning rate
            trainer.scheduler.step()
            current_lr = trainer.optimizer.param_groups[0]['lr']
            
            # Log metrics to AI Manager
            run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": current_lr,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter
            })
            
            # Print progress
            print(f'Epoch {epoch+1}/{max_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Learning Rate: {current_lr:.6f}')
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(os.getcwd(), 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                
                # Log best model as artifact
                artifact = ArtifactInfo(
                    file_path=best_model_path,
                    name=f"best_model_epoch_{epoch+1}",
                    artifact_type=ArtifactType.MODEL,
                    metadata={
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc
                    }
                )
                run.log_artifact(artifact)
                
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Final evaluation
        print("📈 Final evaluation...")
        test_accuracy, classification_report_str = trainer.evaluate(test_loader)
        
        # Log final results
        run.log({
            "final_test_accuracy": test_accuracy,
            "total_epochs": epoch + 1,
            "early_stopped": patience_counter >= early_stopping_patience
        })
        
        # Save final model
        final_model_path = os.path.join(os.getcwd(), 'final_model.pth')
        trainer.save_model(final_model_path)
        
        # Log final model as artifact
        final_artifact = ArtifactInfo(
            file_path=final_model_path,
            name="final_model",
            artifact_type=ArtifactType.MODEL,
            metadata={
                "final_test_accuracy": test_accuracy,
                "total_epochs": epoch + 1,
                "early_stopped": patience_counter >= early_stopping_patience
            }
        )
        run.log_artifact(final_artifact)
        
        # Save training history plot
        trainer.plot_training_history()
        
        # Log training history plot as artifact
        history_path = os.path.join(os.getcwd(), 'training_history.png')
        history_artifact = ArtifactInfo(
            file_path=history_path,
            name="training_history_plot",
            artifact_type=ArtifactType.OTHER,
            metadata={
                "plot_type": "training_history",
                "final_epoch": epoch + 1
            }
        )
        run.log_artifact(history_artifact)
        
        # Log data info as artifact
        data_path = os.path.join(os.getcwd(), 'data_info.json')
        data_artifact = ArtifactInfo(
            file_path=data_path,
            name="dataset_info",
            artifact_type=ArtifactType.DATA,
            metadata={
                "data_type": "synthetic_images",
                "num_samples": len(X),
                "num_classes": 3
            }
        )
        run.log_artifact(data_artifact)
        
        # Finish the run
        run.finish(RunStatus.COMPLETED)
        
        print("✅ Training completed successfully!")
        print(f"Final test accuracy: {test_accuracy:.2f}%")
        print("📊 Check the AI Manager dashboard for detailed metrics and artifacts!")


def reset_and_retrain():
    """Reset the model and retrain to demonstrate multiple runs."""
    
    print("🔄 Reset and Retrain Demo")
    print("=" * 50)
    
    # Initialize AI Manager
    manager = AIManager(
        project_name="sample_image_classifier",
        api_url="http://localhost:8000",
        username="admin",
        password="admin123"
    )
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=100, image_size=32, n_classes=3)
    train_loader, test_loader = create_data_loaders(X, y, batch_size=8)
    
    # Multiple training runs with different configurations
    configs = [
        {"learning_rate": 0.001, "batch_size": 8, "description": "baseline"},
        {"learning_rate": 0.01, "batch_size": 8, "description": "high_lr"},
        {"learning_rate": 0.0001, "batch_size": 16, "description": "low_lr_large_batch"}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n🎯 Training Run {i+1}: {config['description']}")
        
        # Create fresh model
        model = create_model()
        trainer = ModelTrainer(model, device='cpu')
        
        # Update trainer with new config
        trainer.optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        with manager.run(
            name=f"run_{i+1}_{config['description']}_{datetime.now().strftime('%H%M%S')}",
            config={
                "run_number": i + 1,
                "description": config['description'],
                "learning_rate": config['learning_rate'],
                "batch_size": config['batch_size'],
                "model_architecture": "SmallCNN",
                "device": "cpu"
            }
        ) as run:
            
            # Short training run for demo
            for epoch in range(10):  # Reduced epochs for demo
                train_loss, train_acc = trainer.train_epoch(train_loader)
                val_loss, val_acc = trainer.validate(test_loader)
                
                # Log metrics
                run.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": config['learning_rate']
                })
                
                print(f'  Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # Final evaluation
            test_accuracy, _ = trainer.evaluate(test_loader)
            run.log({"final_test_accuracy": test_accuracy})
            
            # Save model
            model_path = os.path.join(os.getcwd(), f'model_run_{i+1}.pth')
            torch.save(model.state_dict(), model_path)
            
            # Log model artifact
            artifact = ArtifactInfo(
                file_path=model_path,
                name=f"model_run_{i+1}_{config['description']}",
                artifact_type=ArtifactType.MODEL,
                metadata={
                    "run_number": i + 1,
                    "description": config['description'],
                    "final_accuracy": test_accuracy
                }
            )
            run.log_artifact(artifact)
            
            run.finish(RunStatus.COMPLETED)
            
            print(f"  ✅ Run {i+1} completed: {test_accuracy:.2f}% accuracy")
    
    print("\n🎉 All training runs completed!")
    print("📊 Check the AI Manager dashboard to compare the different runs!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train model with AI Manager integration')
    parser.add_argument('--mode', choices=['train', 'retrain'], default='train',
                       help='Training mode: train (single run) or retrain (multiple runs)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_with_ai_manager()
    else:
        reset_and_retrain() 