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
from app.api import auth, projects, runs, metrics, artifacts

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
app.include_router(artifacts.router)

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
    
    # Create default user if it doesn't exist
    from app.database import SessionLocal, User
    from app.core.auth import get_password_hash
    
    db = SessionLocal()
    try:
        # Check if default user exists
        default_user = db.query(User).filter(User.username == "admin").first()
        if not default_user:
            # Create default user
            default_user = User(
                username="admin",
                email="admin@example.com",
                hashed_password=get_password_hash("admin123"),
                is_active=True
            )
            db.add(default_user)
            db.commit()
            print("Default user created: admin/admin123")
    except Exception as e:
        print(f"Error creating default user: {e}")
    finally:
        db.close()


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
                    <span id="user-info" class="nav-link" style="display: none;">
                        Welcome, <span id="username"></span> | 
                        <a href="#" onclick="logout()" class="text-white">Logout</a>
                    </span>
                    <a class="nav-link" href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <!-- Login Form -->
        <div id="login-container" class="container mt-5">
            <div class="row justify-content-center">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">Login to AI Manager</h5>
                        </div>
                        <div class="card-body">
                            <form id="login-form">
                                <div class="mb-3">
                                    <label for="username" class="form-label">Username</label>
                                    <input type="text" class="form-control" id="username-input" required>
                                </div>
                                <div class="mb-3">
                                    <label for="password" class="form-label">Password</label>
                                    <input type="password" class="form-control" id="password-input" required>
                                </div>
                                <button type="submit" class="btn btn-primary w-100">Login</button>
                            </form>
                            <hr>
                            <p class="text-center mb-0">
                                <small class="text-muted">Don't have an account? 
                                    <a href="#" onclick="showRegister()">Register here</a>
                                </small>
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Register Form -->
        <div id="register-container" class="container mt-5" style="display: none;">
            <div class="row justify-content-center">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">Register for AI Manager</h5>
                        </div>
                        <div class="card-body">
                            <form id="register-form">
                                <div class="mb-3">
                                    <label for="reg-email" class="form-label">Email</label>
                                    <input type="email" class="form-control" id="reg-email" required>
                                </div>
                                <div class="mb-3">
                                    <label for="reg-username" class="form-label">Username</label>
                                    <input type="text" class="form-control" id="reg-username" required>
                                </div>
                                <div class="mb-3">
                                    <label for="reg-password" class="form-label">Password</label>
                                    <input type="password" class="form-control" id="reg-password" required>
                                </div>
                                <button type="submit" class="btn btn-primary w-100">Register</button>
                            </form>
                            <hr>
                            <p class="text-center mb-0">
                                <small class="text-muted">Already have an account? 
                                    <a href="#" onclick="showLogin()">Login here</a>
                                </small>
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Dashboard -->
        <div id="dashboard-container" class="container mt-4" style="display: none;">
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
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h5 class="card-title mb-0">Training Runs</h5>
                            <button id="new-run-btn" class="btn btn-success btn-sm" onclick="createRun()" style="display: none;">
                                New Run
                            </button>
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
            let authToken = null;
            let currentUser = null;

            // Check if user is already logged in
            document.addEventListener('DOMContentLoaded', function() {
                const savedToken = localStorage.getItem('authToken');
                const savedUser = localStorage.getItem('currentUser');
                
                if (savedToken && savedUser) {
                    authToken = savedToken;
                    currentUser = JSON.parse(savedUser);
                    showDashboard();
                    loadProjects();
                } else {
                    showLogin();
                }
            });

            function showLogin() {
                document.getElementById('login-container').style.display = 'block';
                document.getElementById('register-container').style.display = 'none';
                document.getElementById('dashboard-container').style.display = 'none';
                document.getElementById('user-info').style.display = 'none';
            }

            function showRegister() {
                document.getElementById('login-container').style.display = 'none';
                document.getElementById('register-container').style.display = 'block';
                document.getElementById('dashboard-container').style.display = 'none';
                document.getElementById('user-info').style.display = 'none';
            }

            function showDashboard() {
                document.getElementById('login-container').style.display = 'none';
                document.getElementById('register-container').style.display = 'none';
                document.getElementById('dashboard-container').style.display = 'block';
                document.getElementById('user-info').style.display = 'block';
                document.getElementById('username').textContent = currentUser.username;
            }

            function logout() {
                localStorage.removeItem('authToken');
                localStorage.removeItem('currentUser');
                authToken = null;
                currentUser = null;
                showLogin();
            }

            // Login form handler
            document.getElementById('login-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const username = document.getElementById('username-input').value;
                const password = document.getElementById('password-input').value;
                
                try {
                    const response = await fetch('/auth/login', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        authToken = data.access_token;
                        
                        // Get user info
                        const userResponse = await fetch('/auth/me', {
                            headers: {
                                'Authorization': `Bearer ${authToken}`
                            }
                        });
                        
                        if (userResponse.ok) {
                            currentUser = await userResponse.json();
                            localStorage.setItem('authToken', authToken);
                            localStorage.setItem('currentUser', JSON.stringify(currentUser));
                            showDashboard();
                            loadProjects();
                        }
                    } else {
                        alert('Login failed. Please check your credentials.');
                    }
                } catch (error) {
                    console.error('Login error:', error);
                    alert('Login failed. Please try again.');
                }
            });

            // Register form handler
            document.getElementById('register-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const email = document.getElementById('reg-email').value;
                const username = document.getElementById('reg-username').value;
                const password = document.getElementById('reg-password').value;
                
                try {
                    const response = await fetch('/auth/register', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            email: email,
                            username: username,
                            password: password
                        })
                    });
                    
                    if (response.ok) {
                        alert('Registration successful! Please login.');
                        showLogin();
                        document.getElementById('login-form').reset();
                    } else {
                        const error = await response.json();
                        alert(`Registration failed: ${error.detail}`);
                    }
                } catch (error) {
                    console.error('Registration error:', error);
                    alert('Registration failed. Please try again.');
                }
            });

            async function loadProjects() {
                try {
                    const response = await fetch('/projects/', {
                        headers: {
                            'Authorization': `Bearer ${authToken}`
                        }
                    });
                    
                    if (response.status === 401) {
                        // Token expired or invalid
                        logout();
                        return;
                    }
                    
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
                        <a href="#" onclick="selectProject(${project.id}, '${project.name}')" class="text-decoration-none project-link" data-project-id="${project.id}">
                            ${project.name}
                        </a>
                        <span class="badge bg-secondary">${project.runs?.length || 0}</span>
                    </div>
                `).join('');
            }

            async function selectProject(projectId, projectName) {
                currentProject = projectId;
                
                // Update UI to show selected project
                document.querySelectorAll('.project-link').forEach(link => {
                    link.classList.remove('fw-bold', 'text-primary');
                });
                document.querySelector(`[data-project-id="${projectId}"]`).classList.add('fw-bold', 'text-primary');
                
                // Update runs section header and show new run button
                document.querySelector('.card-header h5').textContent = `Training Runs - ${projectName}`;
                document.getElementById('new-run-btn').style.display = 'block';
                
                await loadRuns(projectId);
            }

            async function loadRuns(projectId) {
                try {
                    const response = await fetch(`/runs/?project_id=${projectId}`, {
                        headers: {
                            'Authorization': `Bearer ${authToken}`
                        }
                    });
                    
                    if (response.status === 401) {
                        logout();
                        return;
                    }
                    
                    const runs = await response.json();
                    displayRuns(runs);
                } catch (error) {
                    console.error('Error loading runs:', error);
                }
            }

            function displayRuns(runs) {
                const container = document.getElementById('runs-list');
                if (runs.length === 0) {
                    container.innerHTML = '<p class="text-muted">No runs found for this project. Click "New Run" to start training.</p>';
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
                                        ${run.ended_at ? `<br>Ended: ${new Date(run.ended_at).toLocaleString()}` : ''}
                                    </small>
                                    ${run.tags && run.tags.length > 0 ? `
                                        <div class="mt-1">
                                            ${run.tags.map(tag => `<span class="badge bg-light text-dark me-1">${tag}</span>`).join('')}
                                        </div>
                                    ` : ''}
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
                    const response = await fetch(`/metrics/${runId}/summary`, {
                        headers: {
                            'Authorization': `Bearer ${authToken}`
                        }
                    });
                    
                    if (response.status === 401) {
                        logout();
                        return;
                    }
                    
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
                            'Authorization': `Bearer ${authToken}`
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

            async function createRun() {
                if (!currentProject) {
                    alert('Please select a project first');
                    return;
                }

                const name = prompt('Enter run name:');
                if (!name) return;

                const tags = prompt('Enter tags (comma-separated, optional):');
                const tagList = tags ? tags.split(',').map(tag => tag.trim()).filter(tag => tag) : [];

                try {
                    const response = await fetch(`/runs/?project_id=${currentProject}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${authToken}`
                        },
                        body: JSON.stringify({
                            name: name,
                            config: {},
                            tags: tagList
                        })
                    });

                    if (response.ok) {
                        await loadRuns(currentProject);
                        alert('Run created successfully! You can now start logging metrics.');
                    } else {
                        alert('Error creating run');
                    }
                } catch (error) {
                    console.error('Error creating run:', error);
                    alert('Error creating run');
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