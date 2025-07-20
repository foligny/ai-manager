"""
Main FastAPI application for AI Manager.
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import socketio
from typing import List, Optional
import os
from pathlib import Path

# Import your existing modules
from app.api import auth, projects, runs, metrics, artifacts, training, models
from app.database import engine, Base
from app.config import settings

# Create FastAPI app
fastapi_app = FastAPI(title="AI Manager", version="1.0.0")

# Create Socket.IO app
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, fastapi_app)

# Setup static files and templates
static_path = Path(__file__).parent.parent / "static"
templates_path = Path(__file__).parent.parent / "static" / "templates"

# Create directories if they don't exist
static_path.mkdir(exist_ok=True)
templates_path.mkdir(exist_ok=True)

# Mount static files
fastapi_app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=str(templates_path))

# Include API routers
fastapi_app.include_router(auth.router, prefix="/auth", tags=["authentication"])
fastapi_app.include_router(projects.router, prefix="/projects", tags=["projects"])
fastapi_app.include_router(runs.router, prefix="/runs", tags=["runs"])
fastapi_app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
fastapi_app.include_router(artifacts.router, prefix="/artifacts", tags=["artifacts"])
fastapi_app.include_router(training.router, prefix="/training", tags=["training"])
fastapi_app.include_router(models.router, prefix="/models", tags=["models"])

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, run_id: int):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket, run_id: int):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_to_run(self, message: str, run_id: int):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Socket.IO events
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def join_run(sid, data):
    run_id = data.get('run_id')
    if run_id:
        sio.enter_room(sid, f"run_{run_id}")
        print(f"Client {sid} joined run {run_id}")

@sio.event
async def leave_run(sid, data):
    run_id = data.get('run_id')
    if run_id:
        sio.leave_room(sid, f"run_{run_id}")
        print(f"Client {sid} left run {run_id}")

# Startup event
@fastapi_app.on_event("startup")
async def startup_event():
    # Create database tables
    Base.metadata.create_all(bind=engine)
    print("🚀 AI Manager started successfully!")

# Root endpoint - serve the main dashboard
@fastapi_app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# Login page
@fastapi_app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# WebSocket endpoint
@fastapi_app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: int):
    """WebSocket endpoint for real-time metric updates."""
    await manager.connect(websocket, run_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, run_id)

# Health check endpoint
@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "2023-01-01T00:00:00"}

# Export the Socket.IO ASGI app (wraps FastAPI app)
app = socket_app 