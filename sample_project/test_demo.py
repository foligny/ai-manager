"""
Simple demo script to test the complete AI Manager integration.
This script demonstrates the full workflow from data generation to training.
"""

import sys
import os
import numpy as np
import torch
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_manager import AIManager, RunStatus, ArtifactType, ArtifactInfo
from data_generator import generate_synthetic_data, create_data_loaders
from model import create_model, ModelTrainer


def quick_demo():
    """Quick demo of the complete workflow."""
    
    print("🚀 AI Manager Integration Demo")
    print("=" * 50)
    
    # 1. Generate data
    print("📊 Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=100, image_size=32, n_classes=3)
    train_loader, test_loader = create_data_loaders(X, y, batch_size=8)
    print(f"✅ Generated {len(X)} samples, {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    
    # 2. Create model
    print("🏗️ Creating model...")
    model = create_model()
    trainer = ModelTrainer(model, device='cpu')
    model_info = model.get_model_info()
    print(f"✅ Model created: {model_info['architecture']}, {model_info['total_parameters']:,} parameters")
    
    # 3. Initialize AI Manager
    print("🔧 Initializing AI Manager...")
    manager = AIManager(
        project_name="demo_image_classifier",
        api_url="http://localhost:8000",
        username="admin",
        password="admin123"
    )
    
    # 4. Training with AI Manager
    print("🎯 Starting training with AI Manager monitoring...")
    
    with manager.run(
        name=f"demo_run_{datetime.now().strftime('%H%M%S')}",
        config={
            "model_architecture": model_info['architecture'],
            "total_parameters": model_info['total_parameters'],
            "model_size_mb": model_info['model_size_mb'],
            "batch_size": 8,
            "learning_rate": 0.001,
            "max_epochs": 5,  # Short demo
            "data_samples": len(X)
        }
    ) as run:
        
        # Short training loop for demo
        for epoch in range(5):
            # Train
            train_loss, train_acc = trainer.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = trainer.validate(test_loader)
            
            # Update learning rate
            trainer.scheduler.step()
            current_lr = trainer.optimizer.param_groups[0]['lr']
            
            # Log to AI Manager
            run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": current_lr
            })
            
            print(f"  Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Final evaluation
        test_accuracy, _ = trainer.evaluate(test_loader)
        run.log({"final_test_accuracy": test_accuracy})
        
        # Save and log model (artifact API not implemented yet)
        torch.save(model.state_dict(), 'demo_model.pth')
        # artifact = ArtifactInfo(
        #     file_path="demo_model.pth",
        #     name="demo_model",
        #     artifact_type=ArtifactType.MODEL,
        #     metadata={"final_accuracy": test_accuracy}
        # )
        # run.log_artifact(artifact)
        
        # Finish run
        run.finish(RunStatus.COMPLETED)
        
        print(f"✅ Demo completed! Final accuracy: {test_accuracy:.2f}%")
        print("📊 Check the AI Manager dashboard for detailed metrics!")


def test_type_safety():
    """Test the type-safe interface."""
    
    print("\n🔍 Testing Type-Safe Interface")
    print("=" * 50)
    
    manager = AIManager(
        project_name="type_safety_test",
        api_url="http://localhost:8000",
        username="admin",
        password="admin123"
    )
    
    with manager.run(name="type_test") as run:
        # ✅ Correct usage examples
        print("✅ Testing correct usage...")
        
        # Valid metric logging
        run.log({
            "loss": 0.5,                    # float
            "accuracy": 0.85,               # float
            "epoch": 10,                    # int
            "is_training": True,            # bool
            "model_type": "transformer"     # str
        })
        
        # Valid single metric
        run.log_metric("learning_rate", 0.001)
        
        # Valid configuration
        run.log_config({
            "batch_size": 32,
            "optimizer": "adam",
            "learning_rate": 0.001
        })
        
        # Valid artifact (commented out for now - API not implemented)
        # torch.save(torch.randn(10, 10), "test.pth")
        # artifact = ArtifactInfo(
        #     file_path="test.pth",
        #     name="test_model",
        #     artifact_type=ArtifactType.MODEL
        # )
        # run.log_artifact(artifact)
        
        # Valid run status
        run.finish(RunStatus.COMPLETED)
        
        print("✅ All type-safe operations completed successfully!")
        print("🔍 Type checking would catch errors like:")
        print("  - run.log({'loss': [0.5, 0.4]})  # List not allowed")
        print("  - run.finish('invalid')  # Must use enum")
        print("  - run.log_artifact('invalid_type')  # Must use enum")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Manager Demo')
    parser.add_argument('--mode', choices=['demo', 'type_test'], default='demo',
                       help='Demo mode: demo (full workflow) or type_test (type safety)')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        quick_demo()
    else:
        test_type_safety() 