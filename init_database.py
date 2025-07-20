#!/usr/bin/env python3
"""
Database initialization script.
Creates admin user and demo data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal, User, Project, Run, Metric, Base, engine
from app.core.auth import get_password_hash
from datetime import datetime
import json

def init_database():
    """Initialize database with admin user and demo data."""
    print("🗄️ Initializing database...")
    
    # Create tables first
    print("📋 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")
    
    db = SessionLocal()
    
    try:
        
        # Check if admin user exists
        admin_user = db.query(User).filter(User.username == "admin").first()
        
        if not admin_user:
            print("👤 Creating admin user...")
            admin_user = User(
                username="admin",
                email="admin@example.com",
                hashed_password=get_password_hash("admin123"),
                is_active=True,
                is_superuser=True
            )
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)
            print("✅ Admin user created")
        else:
            print("✅ Admin user already exists")
        
        # Create demo projects
        print("📁 Creating demo projects...")
        
        # Check if demo projects exist
        demo_project = db.query(Project).filter(Project.name == "demo_image_classifier").first()
        
        if not demo_project:
            demo_project = Project(
                name="demo_image_classifier",
                description="A demo project for testing image classification with various models",
                owner_id=admin_user.id,
                is_public=True,
                tags=["demo", "image-classification", "computer-vision", "pytorch", "cnn"]
            )
            db.add(demo_project)
            db.commit()
            db.refresh(demo_project)
            print("✅ Demo project created")
        else:
            print("✅ Demo project already exists")
        
        # Create demo runs
        print("🏃 Creating demo runs...")
        
        # Check if demo runs exist
        existing_runs = db.query(Run).filter(Run.project_id == demo_project.id).count()
        
        if existing_runs == 0:
            # Create baseline run
            baseline_run = Run(
                name="baseline_cnn",
                project_id=demo_project.id,
                status="completed",
                config={
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 50,
                    "model_type": "cnn",
                    "optimizer": "adam"
                },
                tags=["baseline", "cnn", "small-model"],
                started_at=datetime.utcnow(),
                ended_at=datetime.utcnow()
            )
            db.add(baseline_run)
            db.commit()
            db.refresh(baseline_run)
            
            # Create metrics for baseline run
            for epoch in range(10):
                train_loss = 2.0 * (0.9 ** epoch) + 0.1
                val_loss = train_loss + 0.2 + (epoch * 0.01)
                train_acc = 0.3 + (0.6 * epoch / 10)
                val_acc = train_acc - 0.1
                
                # Log multiple metrics per epoch
                metrics_data = [
                    {"name": "epoch", "value": epoch, "step": epoch},
                    {"name": "train_loss", "value": round(train_loss, 4), "step": epoch},
                    {"name": "val_loss", "value": round(val_loss, 4), "step": epoch},
                    {"name": "train_accuracy", "value": round(train_acc, 4), "step": epoch},
                    {"name": "val_accuracy", "value": round(val_acc, 4), "step": epoch},
                    {"name": "learning_rate", "value": 0.001, "step": epoch}
                ]
                
                for metric_data in metrics_data:
                    metric = Metric(
                        run_id=baseline_run.id,
                        name=metric_data["name"],
                        value=metric_data["value"],
                        step=metric_data["step"]
                    )
                    db.add(metric)
            
            # Create resnet run
            resnet_run = Run(
                name="resnet_experiment",
                project_id=demo_project.id,
                status="completed",
                config={
                    "learning_rate": 0.0001,
                    "batch_size": 16,
                    "epochs": 100,
                    "model_type": "resnet",
                    "optimizer": "adam"
                },
                tags=["resnet", "large-model", "high-accuracy"],
                started_at=datetime.utcnow(),
                ended_at=datetime.utcnow()
            )
            db.add(resnet_run)
            db.commit()
            db.refresh(resnet_run)
            
            # Create metrics for resnet run
            for epoch in range(15):
                train_loss = 1.8 * (0.85 ** epoch) + 0.05
                val_loss = train_loss + 0.15 + (epoch * 0.005)
                train_acc = 0.25 + (0.7 * epoch / 15)
                val_acc = train_acc - 0.08
                
                metrics_data = [
                    {"name": "epoch", "value": epoch, "step": epoch},
                    {"name": "train_loss", "value": round(train_loss, 4), "step": epoch},
                    {"name": "val_loss", "value": round(val_loss, 4), "step": epoch},
                    {"name": "train_accuracy", "value": round(train_acc, 4), "step": epoch},
                    {"name": "val_accuracy", "value": round(val_acc, 4), "step": epoch},
                    {"name": "learning_rate", "value": 0.0001, "step": epoch}
                ]
                
                for metric_data in metrics_data:
                    metric = Metric(
                        run_id=resnet_run.id,
                        name=metric_data["name"],
                        value=metric_data["value"],
                        step=metric_data["step"]
                    )
                    db.add(metric)
            
            # Create attention experiment run
            attention_run = Run(
                name="attention_mechanism",
                project_id=demo_project.id,
                status="completed",
                config={
                    "learning_rate": 0.0005,
                    "batch_size": 8,
                    "epochs": 75,
                    "model_type": "transformer",
                    "optimizer": "adamw"
                },
                tags=["attention", "experiment", "research"],
                started_at=datetime.utcnow(),
                ended_at=datetime.utcnow()
            )
            db.add(attention_run)
            db.commit()
            db.refresh(attention_run)
            
            # Create metrics for attention run
            for epoch in range(8):
                train_loss = 2.2 * (0.88 ** epoch) + 0.12
                val_loss = train_loss + 0.25 + (epoch * 0.015)
                train_acc = 0.2 + (0.75 * epoch / 8)
                val_acc = train_acc - 0.12
                
                metrics_data = [
                    {"name": "epoch", "value": epoch, "step": epoch},
                    {"name": "train_loss", "value": round(train_loss, 4), "step": epoch},
                    {"name": "val_loss", "value": round(val_loss, 4), "step": epoch},
                    {"name": "train_accuracy", "value": round(train_acc, 4), "step": epoch},
                    {"name": "val_accuracy", "value": round(val_acc, 4), "step": epoch},
                    {"name": "learning_rate", "value": 0.0005, "step": epoch}
                ]
                
                for metric_data in metrics_data:
                    metric = Metric(
                        run_id=attention_run.id,
                        name=metric_data["name"],
                        value=metric_data["value"],
                        step=metric_data["step"]
                    )
                    db.add(metric)
            
            db.commit()
            print("✅ Demo runs and metrics created")
        else:
            print("✅ Demo runs already exist")
        
        print("🎉 Database initialization completed!")
        print("📊 You can now:")
        print("   - Login with admin/admin123")
        print("   - View demo projects and runs")
        print("   - See training metrics and charts")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    init_database() 