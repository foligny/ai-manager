"""
Artifacts API routes for file and model storage.
"""

import os
import shutil
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.database import get_db, User, Run, Artifact, Project
from app.schemas.artifact import ArtifactCreate, Artifact as ArtifactSchema
from app.api.auth import get_current_user
from app.config import settings

router = APIRouter(tags=["artifacts"])


@router.post("/{run_id}/artifacts", response_model=ArtifactSchema)
async def upload_artifact(
    run_id: int,
    file: UploadFile = File(...),
    name: str = Form(...),
    type: str = Form("other"),
    metadata: str = Form("{}"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Upload an artifact for a run."""
    # Check if run exists and user has access
    run = db.query(Run).join(Project).filter(Run.id == run_id).first()
    
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found"
        )
    
    if run.project.owner_id != current_user.id and not run.project.is_public:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Create storage directory if it doesn't exist
    storage_dir = os.path.join(settings.storage_path, str(run_id))
    os.makedirs(storage_dir, exist_ok=True)
    
    # Save file
    file_path = os.path.join(storage_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Parse metadata if it's a JSON string
    import json
    try:
        metadata_dict = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        metadata_dict = {"raw_metadata": metadata}
    
    # Create artifact record
    db_artifact = Artifact(
        run_id=run_id,
        name=name,
        type=type,
        path=file_path,
        size=file_size,
        artifact_metadata=metadata_dict
    )
    
    db.add(db_artifact)
    db.commit()
    db.refresh(db_artifact)
    
    return db_artifact


@router.get("/{run_id}/artifacts", response_model=List[ArtifactSchema])
def list_artifacts(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """List all artifacts for a run."""
    # Check if run exists and user has access
    run = db.query(Run).join(Project).filter(Run.id == run_id).first()
    
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found"
        )
    
    if run.project.owner_id != current_user.id and not run.project.is_public:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    artifacts = db.query(Artifact).filter(Artifact.run_id == run_id).all()
    return artifacts


@router.get("/artifacts/{artifact_id}")
def download_artifact(
    artifact_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Download an artifact."""
    artifact = db.query(Artifact).join(Run).join(Project).filter(Artifact.id == artifact_id).first()
    
    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found"
        )
    
    # Check if user has access to this artifact's run
    if artifact.run.project.owner_id != current_user.id and not artifact.run.project.is_public:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    if not os.path.exists(artifact.path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact file not found"
        )
    
    return FileResponse(
        path=artifact.path,
        filename=artifact.name,
        media_type='application/octet-stream'
    )


@router.delete("/artifacts/{artifact_id}")
def delete_artifact(
    artifact_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Delete an artifact."""
    artifact = db.query(Artifact).join(Run).join(Project).filter(Artifact.id == artifact_id).first()
    
    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found"
        )
    
    # Check if user has access to this artifact's run
    if artifact.run.project.owner_id != current_user.id and not artifact.run.project.is_public:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Delete file if it exists
    if os.path.exists(artifact.path):
        os.remove(artifact.path)
    
    # Delete database record
    db.delete(artifact)
    db.commit()
    
    return {"message": "Artifact deleted successfully"} 