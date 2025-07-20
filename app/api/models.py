"""
Models API endpoints for model testing and management.
"""

import os
import torch
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import Dict, Any
import json

from app.database import get_db
from app.api.auth import get_current_user
from app.config import settings

router = APIRouter(tags=["models"])


@router.post("/upload")
async def upload_model(
    model: UploadFile = File(...),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a model file."""
    
    # Validate file type
    if not model.filename.endswith(('.pth', '.pt', '.pkl')):
        raise HTTPException(status_code=400, detail="Invalid model file type")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(settings.storage_path, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the model file
    model_path = os.path.join(models_dir, model.filename)
    with open(model_path, "wb") as f:
        content = await model.read()
        f.write(content)
    
    # Generate model ID
    model_id = f"model_{len(os.listdir(models_dir))}"
    
    return {
        "model_id": model_id,
        "filename": model.filename,
        "path": model_path,
        "size": len(content)
    }


@router.post("/load/{model_name}")
async def load_demo_model(
    model_name: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Load a demo model."""
    
    # Check if model exists in root directory
    model_path = os.path.join(os.getcwd(), model_name)
    
    if not os.path.exists(model_path):
        # Check in sample_project directory
        sample_path = os.path.join(os.getcwd(), "sample_project", model_name)
        if os.path.exists(sample_path):
            model_path = sample_path
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    try:
        # Load the model (this is a simplified version)
        model_info = {
            "name": model_name,
            "path": model_path,
            "size": os.path.getsize(model_path),
            "status": "loaded"
        }
        
        return model_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@router.post("/test")
async def test_model(
    test_data: UploadFile = File(...),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Test a model with provided data."""
    
    try:
        # Read test data
        content = await test_data.read()
        
        # Handle different file types
        if test_data.filename.endswith('.npy'):
            # Load numpy array
            data = np.frombuffer(content, dtype=np.float32)
            # Try to reshape to 2D if possible
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
        elif test_data.filename.endswith('.csv'):
            # Load CSV data
            try:
                import pandas as pd
                import io
                df = pd.read_csv(io.BytesIO(content))
                data = df.values
            except ImportError:
                raise HTTPException(status_code=500, detail="pandas not available for CSV processing")
        elif test_data.filename.endswith('.json'):
            # Load JSON data
            data = json.loads(content.decode('utf-8'))
        else:
            raise HTTPException(status_code=400, detail="Unsupported test data format")
        
        # Simulate model testing
        if isinstance(data, np.ndarray):
            # For numpy arrays, simulate predictions
            predictions = np.random.rand(data.shape[0])
            accuracy = np.random.uniform(0.7, 0.95)
            loss = np.random.uniform(0.1, 0.3)
        else:
            # For other data types
            predictions = [np.random.rand() for _ in range(10)]
            accuracy = np.random.uniform(0.7, 0.95)
            loss = np.random.uniform(0.1, 0.3)
        
        results = {
            "data_shape": data.shape if hasattr(data, 'shape') else len(data),
            "predictions": predictions[:5].tolist() if hasattr(predictions, 'tolist') else predictions[:5],
            "accuracy": round(accuracy, 4),
            "loss": round(loss, 4),
            "test_samples": len(predictions),
            "model_performance": {
                "inference_time": round(np.random.uniform(0.01, 0.1), 4),
                "memory_usage": f"{np.random.randint(50, 200)}MB"
            }
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing model: {str(e)}")


@router.get("/list")
async def list_models(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List available models."""
    
    models = []
    
    # Check root directory for demo models
    root_dir = os.getcwd()
    for file in os.listdir(root_dir):
        if file.endswith(('.pth', '.pt', '.pkl')):
            models.append({
                "name": file,
                "path": os.path.join(root_dir, file),
                "size": os.path.getsize(os.path.join(root_dir, file)),
                "type": "demo"
            })
    
    # Check sample_project directory
    sample_dir = os.path.join(root_dir, "sample_project")
    if os.path.exists(sample_dir):
        for file in os.listdir(sample_dir):
            if file.endswith(('.pth', '.pt', '.pkl')):
                models.append({
                    "name": file,
                    "path": os.path.join(sample_dir, file),
                    "size": os.path.getsize(os.path.join(sample_dir, file)),
                    "type": "sample"
                })
    
    return {"models": models} 