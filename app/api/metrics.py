"""
Metrics API routes for logging and retrieving training metrics.
"""

from typing import Any, List, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db, User, Run, Metric, Project
from app.schemas.metric import MetricCreate, Metric as MetricSchema, MetricBatch, MetricHistory, MetricSummary
from app.api.auth import get_current_user

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.post("/{run_id}", response_model=MetricSchema)
def log_metric(
    run_id: int,
    metric: MetricCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Log a single metric for a run."""
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
    
    db_metric = Metric(
        **metric.dict(),
        run_id=run_id
    )
    
    db.add(db_metric)
    db.commit()
    db.refresh(db_metric)
    
    return db_metric


@router.post("/{run_id}/batch")
def log_metrics_batch(
    run_id: int,
    metric_batch: MetricBatch,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Log multiple metrics for a run."""
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
    
    metrics = []
    for name, value in metric_batch.metrics.items():
        metric = Metric(
            name=name,
            value=value,
            step=metric_batch.step or 0,
            run_id=run_id
        )
        metrics.append(metric)
    
    db.add_all(metrics)
    db.commit()
    
    return {"message": f"Logged {len(metrics)} metrics successfully"}


@router.get("/{run_id}", response_model=List[MetricSchema])
def get_run_metrics(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    metric_name: str = None,
    skip: int = 0,
    limit: int = 1000
) -> Any:
    """Get metrics for a specific run."""
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
    
    query = db.query(Metric).filter(Metric.run_id == run_id)
    
    if metric_name:
        query = query.filter(Metric.name == metric_name)
    
    metrics = query.offset(skip).limit(limit).all()
    return metrics


@router.get("/{run_id}/history/{metric_name}", response_model=MetricHistory)
def get_metric_history(
    run_id: int,
    metric_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Get metric history for a specific metric in a run."""
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
    
    metrics = db.query(Metric).filter(
        Metric.run_id == run_id,
        Metric.name == metric_name
    ).order_by(Metric.step).all()
    
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metric not found"
        )
    
    return MetricHistory(
        name=metric_name,
        values=[m.value for m in metrics],
        steps=[m.step for m in metrics],
        timestamps=[m.timestamp for m in metrics]
    )


@router.get("/{run_id}/summary", response_model=List[MetricSummary])
def get_metrics_summary(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Get summary statistics for all metrics in a run."""
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
    
    # Get summary statistics for each metric
    summaries = []
    metric_names = db.query(Metric.name).filter(Metric.run_id == run_id).distinct().all()
    
    for (metric_name,) in metric_names:
        metrics = db.query(Metric).filter(
            Metric.run_id == run_id,
            Metric.name == metric_name
        ).all()
        
        if metrics:
            values = [m.value for m in metrics]
            current_value = values[-1] if values else 0
            min_value = min(values) if values else 0
            max_value = max(values) if values else 0
            mean_value = sum(values) / len(values) if values else 0
            
            summary = MetricSummary(
                name=metric_name,
                current_value=current_value,
                min_value=min_value,
                max_value=max_value,
                mean_value=mean_value,
                total_points=len(values)
            )
            summaries.append(summary)
    
    return summaries 