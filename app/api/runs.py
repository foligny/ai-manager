"""
Run API routes for experiment tracking.
"""

from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db, User, Project, Run
from app.schemas.run import RunCreate, Run as RunSchema, RunUpdate, RunSummary
from app.api.auth import get_current_user

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("/", response_model=List[RunSummary])
def list_runs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    project_id: int = None,
    skip: int = 0,
    limit: int = 100
) -> Any:
    """List all runs for the current user."""
    query = db.query(Run).join(Project)
    
    # Filter by project if specified
    if project_id:
        query = query.filter(Run.project_id == project_id)
    
    # Filter by user access
    query = query.filter(
        (Project.owner_id == current_user.id) | (Project.is_public == True)
    )
    
    runs = query.offset(skip).limit(limit).all()
    return runs


@router.post("/", response_model=RunSchema)
def create_run(
    run: RunCreate,
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Create a new run."""
    # Check if project exists and user has access
    project = db.query(Project).filter(Project.id == project_id).first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    if project.owner_id != current_user.id and not project.is_public:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    db_run = Run(
        **run.dict(),
        project_id=project_id
    )
    
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    
    return db_run


@router.get("/{run_id}", response_model=RunSchema)
def get_run(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Get a specific run."""
    run = db.query(Run).join(Project).filter(Run.id == run_id).first()
    
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found"
        )
    
    # Check if user has access to this run's project
    if run.project.owner_id != current_user.id and not run.project.is_public:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    return run


@router.put("/{run_id}", response_model=RunSchema)
def update_run(
    run_id: int,
    run_update: RunUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Update a run."""
    run = db.query(Run).join(Project).filter(Run.id == run_id).first()
    
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found"
        )
    
    if run.project.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Update run fields
    update_data = run_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(run, field, value)
    
    db.commit()
    db.refresh(run)
    
    return run


@router.delete("/{run_id}")
def delete_run(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Delete a run."""
    run = db.query(Run).join(Project).filter(Run.id == run_id).first()
    
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found"
        )
    
    if run.project.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    db.delete(run)
    db.commit()
    
    return {"message": "Run deleted successfully"} 