"""
Main FastAPI application for AI Manager.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
import asyncio
from typing import Dict, List
import os

from app.config import settings
from app.database import create_tables
from app.api import auth, projects, runs, metrics

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI Training Monitoring Platform",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(runs.router)
app.include_router(metrics.router)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, run_id: int):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, run_id: int):
        if run_id in self.active_connections:
            self.active_connections[run_id].remove(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast_to_run(self, message: str, run_id: int):
        if run_id in self.active_connections:
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_text(message)
                except:
                    # Remove disconnected clients
                    self.active_connections[run_id].remove(connection)

manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    # Create database tables
    create_tables()
    
    # Create storage directory if it doesn't exist
    os.makedirs(settings.storage_path, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Manager - Training Monitoring</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            .metric-card {
                transition: transform 0.2s;
            }
            .metric-card:hover {
                transform: translateY(-2px);
            }
            .run-card {
                border-left: 4px solid #007bff;
            }
            .run-card.completed {
                border-left-color: #28a745;
            }
            .run-card.failed {
                border-left-color: #dc3545;
            }
            .run-card.running {
                border-left-color: #ffc107;
            }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
            <div class="container">
                <a class="navbar-brand" href="#">AI Manager</a>
                <div class="navbar-nav ms-auto">
                    <a class="nav-link" href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div class="container mt-4">
            <div class="row">
                <div class="col-md-3">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">Projects</h5>
                        </div>
                        <div class="card-body">
                            <div id="projects-list">
                                <div class="text-center">
                                    <div class="spinner-border" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                </div>
                            </div>
                            <button class="btn btn-primary btn-sm w-100 mt-2" onclick="createProject()">
                                New Project
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-9">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">Training Runs</h5>
                        </div>
                        <div class="card-body">
                            <div id="runs-list">
                                <div class="text-center">
                                    <div class="spinner-border" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card mt-4">
                        <div class="card-header">
                            <h5 class="card-title mb-0">Metrics</h5>
                        </div>
                        <div class="card-body">
                            <div id="metrics-chart" style="height: 400px;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            let currentProject = null;
            let currentRun = null;
            let websocket = null;

            // Load projects on page load
            document.addEventListener('DOMContentLoaded', function() {
                loadProjects();
            });

            async function loadProjects() {
                try {
                    const response = await fetch('/projects/');
                    const projects = await response.json();
                    displayProjects(projects);
                } catch (error) {
                    console.error('Error loading projects:', error);
                }
            }

            function displayProjects(projects) {
                const container = document.getElementById('projects-list');
                if (projects.length === 0) {
                    container.innerHTML = '<p class="text-muted">No projects found</p>';
                    return;
                }

                container.innerHTML = projects.map(project => `
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <a href="#" onclick="selectProject(${project.id})" class="text-decoration-none">
                            ${project.name}
                        </a>
                        <span class="badge bg-secondary">${project.runs?.length || 0}</span>
                    </div>
                `).join('');
            }

            async function selectProject(projectId) {
                currentProject = projectId;
                await loadRuns(projectId);
            }

            async function loadRuns(projectId) {
                try {
                    const response = await fetch(`/runs/?project_id=${projectId}`);
                    const runs = await response.json();
                    displayRuns(runs);
                } catch (error) {
                    console.error('Error loading runs:', error);
                }
            }

            function displayRuns(runs) {
                const container = document.getElementById('runs-list');
                if (runs.length === 0) {
                    container.innerHTML = '<p class="text-muted">No runs found</p>';
                    return;
                }

                container.innerHTML = runs.map(run => `
                    <div class="card run-card ${run.status} mb-2">
                        <div class="card-body p-3">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 class="card-title mb-1">${run.name}</h6>
                                    <small class="text-muted">
                                        Started: ${new Date(run.started_at).toLocaleString()}
                                    </small>
                                </div>
                                <div class="text-end">
                                    <span class="badge bg-${getStatusColor(run.status)}">${run.status}</span>
                                    <br>
                                    <button class="btn btn-sm btn-outline-primary mt-1" onclick="selectRun(${run.id})">
                                        View
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('');
            }

            function getStatusColor(status) {
                switch (status) {
                    case 'running': return 'warning';
                    case 'completed': return 'success';
                    case 'failed': return 'danger';
                    default: return 'secondary';
                }
            }

            async function selectRun(runId) {
                currentRun = runId;
                await loadMetrics(runId);
                connectWebSocket(runId);
            }

            async function loadMetrics(runId) {
                try {
                    const response = await fetch(`/metrics/${runId}/summary`);
                    const metrics = await response.json();
                    displayMetrics(metrics);
                } catch (error) {
                    console.error('Error loading metrics:', error);
                }
            }

            function displayMetrics(metrics) {
                const container = document.getElementById('metrics-chart');
                
                if (metrics.length === 0) {
                    container.innerHTML = '<p class="text-muted">No metrics available</p>';
                    return;
                }

                // Create sample chart (in a real app, you'd fetch actual metric data)
                const trace = {
                    x: Array.from({length: 100}, (_, i) => i),
                    y: Array.from({length: 100}, (_, i) => Math.random() * 100),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Loss'
                };

                const layout = {
                    title: 'Training Metrics',
                    xaxis: { title: 'Step' },
                    yaxis: { title: 'Value' }
                };

                Plotly.newPlot('metrics-chart', [trace], layout);
            }

            function connectWebSocket(runId) {
                if (websocket) {
                    websocket.close();
                }

                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/${runId}`;
                
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = function(event) {
                    console.log('WebSocket connected');
                };
                
                websocket.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateMetrics(data);
                };
                
                websocket.onclose = function(event) {
                    console.log('WebSocket disconnected');
                };
            }

            function updateMetrics(data) {
                // Update metrics in real-time
                console.log('Received metric update:', data);
                // In a real implementation, you'd update the chart here
            }

            async function createProject() {
                const name = prompt('Enter project name:');
                if (!name) return;

                try {
                    const response = await fetch('/projects/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            name: name,
                            description: '',
                            is_public: false
                        })
                    });

                    if (response.ok) {
                        await loadProjects();
                    } else {
                        alert('Error creating project');
                    }
                } catch (error) {
                    console.error('Error creating project:', error);
                }
            }
        </script>
    </body>
    </html>
    """


@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: int):
    """WebSocket endpoint for real-time metric updates."""
    await manager.connect(websocket, run_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, run_id)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.version}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    ) 