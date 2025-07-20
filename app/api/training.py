"""
Training API endpoints for triggering training runs from the dashboard.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any
import asyncio
import json
import os
import sys

from app.database import get_db
from app.api.auth import get_current_user
from app.schemas.run import RunCreate, Run
from app.schemas.metric import MetricCreate
from app.database import SessionLocal, Run as RunModel, Metric as MetricModel

router = APIRouter(tags=["training"])


@router.post("/start/{project_id}")
async def start_training(
    project_id: int,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start a training run for the specified project."""
    
    # Create a new run
    run_data = RunCreate(
        name=f"training_{project_id}_{int(asyncio.get_event_loop().time())}",
        project_id=project_id,
        status="running"
    )
    
    db_run = RunModel(**run_data.dict())
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    
    # Start training in background
    background_tasks.add_task(run_training, db_run.id, project_id)
    
    return {"message": "Training started", "run_id": db_run.id}


async def run_training(run_id: int, project_id: int):
    """Background task to run training and log metrics."""
    
    print(f"Starting training for run {run_id}, project {project_id}")
    
    db = SessionLocal()
    try:
        # Simulate training process
        epochs = 10
        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}/{epochs} for run {run_id}")
            
            # Simulate training metrics
            metrics = {
                "epoch": epoch,
                "train_loss": 1.0 - (epoch * 0.08) + (epoch * 0.01 * (epoch % 3)),  # Decreasing with noise
                "train_accuracy": 0.5 + (epoch * 0.04) + (epoch * 0.02 * (epoch % 2)),  # Increasing with noise
                "val_loss": 1.1 - (epoch * 0.07) + (epoch * 0.015 * (epoch % 4)),
                "val_accuracy": 0.48 + (epoch * 0.035) + (epoch * 0.025 * (epoch % 3)),
                "learning_rate": 0.001 * (0.9 ** epoch)
            }
            
            # Log each metric
            for metric_name, value in metrics.items():
                metric_data = MetricCreate(
                    run_id=run_id,
                    name=metric_name,
                    value=value,
                    step=epoch
                )
                db_metric = MetricModel(**metric_data.dict())
                db.add(db_metric)
            
            db.commit()
            
            # Emit Socket.IO event for real-time updates
            try:
                from app.main import sio
                await sio.emit('training_update', {
                    'run_id': run_id,
                    'epoch': epoch,
                    'metrics': metrics,
                    'status': 'running'
                }, room=f"run_{run_id}")
                print(f"Emitted training_update for run {run_id}, epoch {epoch}")
            except Exception as e:
                print(f"Error emitting Socket.IO event: {e}")
            
            # Simulate training time
            await asyncio.sleep(2)
        
        # Mark run as completed
        run = db.query(RunModel).filter(RunModel.id == run_id).first()
        if run:
            run.status = "completed"
            db.commit()
            print(f"Training completed for run {run_id}")
            
            # Emit completion event
            try:
                await sio.emit('training_complete', {
                    'run_id': run_id,
                    'status': 'completed'
                }, room=f"run_{run_id}")
                print(f"Emitted training_complete for run {run_id}")
            except Exception as e:
                print(f"Error emitting completion event: {e}")
            
    except Exception as e:
        print(f"Training failed for run {run_id}: {e}")
        # Mark run as failed
        run = db.query(RunModel).filter(RunModel.id == run_id).first()
        if run:
            run.status = "failed"
            db.commit()
            
            # Emit failure event
            try:
                await sio.emit('training_failed', {
                    'run_id': run_id,
                    'status': 'failed',
                    'error': str(e)
                }, room=f"run_{run_id}")
                print(f"Emitted training_failed for run {run_id}")
            except Exception as emit_error:
                print(f"Error emitting failure event: {emit_error}")
    finally:
        db.close()
        print(f"Training process finished for run {run_id}")


@router.get("/status/{run_id}")
async def get_training_status(
    run_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the status of a training run."""
    
    run = db.query(RunModel).filter(RunModel.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return {
        "run_id": run.id,
        "status": run.status,
        "name": run.name,
        "created_at": run.created_at
    } 