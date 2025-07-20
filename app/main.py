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
import socketio

from app.config import settings
from app.database import create_tables
from app.api import auth, projects, runs, metrics, artifacts, training, models

# Create FastAPI app
fastapi_app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI Training Monitoring Platform",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*'
)
socket_app = socketio.ASGIApp(sio, fastapi_app)

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
fastapi_app.include_router(auth.router)
fastapi_app.include_router(projects.router)
fastapi_app.include_router(runs.router)
fastapi_app.include_router(metrics.router)
fastapi_app.include_router(artifacts.router)
fastapi_app.include_router(training.router)
fastapi_app.include_router(models.router)

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

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def join_run(sid, data):
    """Join a specific training run room."""
    run_id = data.get('run_id')
    if run_id:
        await sio.enter_room(sid, f"run_{run_id}")
        print(f"Client {sid} joined run {run_id}")

@sio.event
async def leave_run(sid, data):
    """Leave a specific training run room."""
    run_id = data.get('run_id')
    if run_id:
        await sio.leave_room(sid, f"run_{run_id}")
        print(f"Client {sid} left run {run_id}")


@fastapi_app.on_event("startup")
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


@fastapi_app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the enhanced professional dashboard."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Manager - Professional Training Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
        <style>
            :root {
                --primary-color: #6366f1;
                --secondary-color: #8b5cf6;
                --success-color: #10b981;
                --warning-color: #f59e0b;
                --danger-color: #ef4444;
                --dark-bg: #1f2937;
                --card-bg: #374151;
                --text-light: #f9fafb;
                --border-color: #4b5563;
            }
            
            body {
                background-color: var(--dark-bg);
                color: var(--text-light);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .navbar {
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
                border-bottom: 1px solid var(--border-color);
            }
            
            .sidebar {
                background-color: var(--card-bg);
                border-right: 1px solid var(--border-color);
                height: 100vh;
                position: fixed;
                width: 280px;
                overflow-y: auto;
            }
            
            .main-content {
                margin-left: 280px;
                padding: 20px;
            }
            
            .card {
                background-color: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .metric-card {
                transition: transform 0.2s, box-shadow 0.2s;
                cursor: pointer;
            }
            
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
            }
            
            .run-item {
                padding: 8px 12px;
                margin: 2px 0;
                border-radius: 6px;
                cursor: pointer;
                transition: background-color 0.2s;
                border-left: 3px solid transparent;
            }
            
            .run-item:hover {
                background-color: rgba(99, 102, 241, 0.1);
            }
            
            .run-item.active {
                background-color: rgba(99, 102, 241, 0.2);
                border-left-color: var(--primary-color);
            }
            
            .run-item.completed {
                border-left-color: var(--success-color);
            }
            
            .run-item.failed {
                border-left-color: var(--danger-color);
            }
            
            .run-item.running {
                border-left-color: var(--warning-color);
            }
            
            .chart-container {
                background-color: var(--card-bg);
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                min-height: 400px;
                height: 400px;
            }
            
            .metric-value {
                font-size: 2rem;
                font-weight: bold;
                color: var(--primary-color);
            }
            
            .metric-label {
                color: #9ca3af;
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            
            .status-badge {
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 500;
            }
            
            .status-completed { background-color: rgba(16, 185, 129, 0.2); color: #10b981; }
            .status-running { background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; }
            .status-failed { background-color: rgba(239, 68, 68, 0.2); color: #ef4444; }
            
            .search-box {
                background-color: var(--dark-bg);
                border: 1px solid var(--border-color);
                color: var(--text-light);
                border-radius: 6px;
                padding: 8px 12px;
            }
            
            .search-box:focus {
                outline: none;
                border-color: var(--primary-color);
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
            }
            
            .btn-primary {
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }
            
            .btn-primary:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
            }
            
            .chart-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
                gap: 25px;
                margin-top: 20px;
            }
            
            .comparison-table {
                background-color: var(--card-bg);
                border-radius: 8px;
                overflow: hidden;
            }
            
            .comparison-table th {
                background-color: rgba(99, 102, 241, 0.1);
                color: var(--text-light);
                font-weight: 600;
                padding: 12px;
            }
            
            .comparison-table td {
                padding: 12px;
                border-bottom: 1px solid var(--border-color);
            }
            
            .loading {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 200px;
                color: var(--text-light);
            }
            
            .spinner {
                border: 3px solid var(--border-color);
                border-top: 3px solid var(--primary-color);
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <!-- Navigation -->
        <nav class="navbar navbar-expand-lg navbar-dark">
            <div class="container-fluid">
                <a class="navbar-brand" href="#">
                    <i class="fas fa-brain me-2"></i>
                    AI Manager Pro
                </a>
                <div class="navbar-nav ms-auto">
                    <span id="user-info" class="nav-link" style="display: none;">
                        <i class="fas fa-user me-1"></i>
                        <span id="username"></span> | 
                        <a href="#" onclick="logout()" class="text-white">Logout</a>
                    </span>
                    <a class="nav-link" href="/docs">
                        <i class="fas fa-book me-1"></i>API Docs
                    </a>
                </div>
            </div>
        </nav>

        <!-- Login Form -->
        <div id="login-container" class="container mt-5">
            <div class="row justify-content-center">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">
                                <i class="fas fa-sign-in-alt me-2"></i>
                                Login to AI Manager Pro
                            </h5>
                        </div>
                        <div class="card-body">
                            <form id="login-form">
                                <div class="mb-3">
                                    <label for="username" class="form-label">Username</label>
                                    <input type="text" class="form-control search-box" id="username-input" required>
                                </div>
                                <div class="mb-3">
                                    <label for="password" class="form-label">Password</label>
                                    <input type="password" class="form-control search-box" id="password-input" required>
                                </div>
                                <button type="submit" class="btn btn-primary w-100">
                                    <i class="fas fa-sign-in-alt me-2"></i>Login
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Main Dashboard -->
        <div id="dashboard-container" style="display: none;">
            <!-- Sidebar -->
            <div class="sidebar">
                <div class="p-3">
                    <h6 class="text-muted mb-3">
                        <i class="fas fa-flask me-2"></i>EXPERIMENTS
                    </h6>
                    <div class="mb-3">
                        <input type="text" class="form-control search-box w-100" 
                               placeholder="Search runs..." id="search-runs">
                    </div>
                    <div id="runs-list">
                        <div class="loading">
                            <div class="spinner"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Main Content -->
            <div class="main-content">
                <!-- Header -->
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <div>
                        <h2 id="project-title">AI Manager Dashboard</h2>
                        <p class="text-muted mb-0" id="project-description">Select a project to view experiments</p>
                    </div>
                    <div class="d-flex gap-2">
                        <button class="btn btn-success" onclick="triggerTraining()">
                            <i class="fas fa-play me-2"></i>Start Training
                        </button>
                        <button class="btn btn-primary" onclick="refreshData()">
                            <i class="fas fa-sync-alt me-2"></i>Refresh
                        </button>
                        <button class="btn btn-outline-primary" onclick="exportData()">
                            <i class="fas fa-download me-2"></i>Export
                        </button>
                    </div>
                </div>

                <!-- Metrics Overview -->
                <div class="row mb-4" id="metrics-overview">
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <div class="metric-value" id="total-runs">0</div>
                                <div class="metric-label">Total Runs</div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <div class="metric-value" id="completed-runs">0</div>
                                <div class="metric-label">Completed</div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <div class="metric-value" id="running-runs">0</div>
                                <div class="metric-label">Running</div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <div class="metric-value" id="success-rate">0%</div>
                                <div class="metric-label">Success Rate</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Charts Grid -->
                <div class="chart-grid" id="charts-container">
                    <div class="loading">
                        <div class="spinner"></div>
                        <div class="mt-3">Loading charts...</div>
                    </div>
                </div>

                <!-- Model Testing Section -->
                <div class="card mt-4">
                    <div class="card-header">
                        <h5 class="mb-0">
                            <i class="fas fa-cogs me-2"></i>Model Testing
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Upload Model</h6>
                                <div class="mb-3">
                                    <input type="file" class="form-control" id="model-file" accept=".pth,.pt,.pkl">
                                </div>
                                <button class="btn btn-primary" onclick="uploadModel()">
                                    <i class="fas fa-upload"></i> Upload Model
                                </button>
                            </div>
                            <div class="col-md-6">
                                <h6>Test Data</h6>
                                <div class="mb-3">
                                    <input type="file" class="form-control" id="test-data-file" accept=".npy,.csv,.json">
                                </div>
                                <button class="btn btn-success" onclick="testModel()">
                                    <i class="fas fa-play"></i> Test Model
                                </button>
                            </div>
                        </div>
                        
                        <div class="mt-4">
                            <h6>Demo Models</h6>
                            <div class="row">
                                <div class="col-md-4">
                                    <div class="card">
                                        <div class="card-body">
                                            <h6>Demo Model</h6>
                                            <p class="text-muted small">demo_model.pth</p>
                                            <button class="btn btn-outline-primary btn-sm" onclick="loadDemoModel('demo_model.pth')">
                                                Load Model
                                            </button>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="card">
                                        <div class="card-body">
                                            <h6>Run 1 Model</h6>
                                            <p class="text-muted small">model_run_1.pth</p>
                                            <button class="btn btn-outline-primary btn-sm" onclick="loadDemoModel('model_run_1.pth')">
                                                Load Model
                                            </button>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="card">
                                        <div class="card-body">
                                            <h6>Run 2 Model</h6>
                                            <p class="text-muted small">model_run_2.pth</p>
                                            <button class="btn btn-outline-primary btn-sm" onclick="loadDemoModel('model_run_2.pth')">
                                                Load Model
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="mt-4">
                            <h6>Test Results</h6>
                            <div id="test-results" class="alert alert-info" style="display: none;">
                                <div id="test-output"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Run Comparison Table -->
                <div class="card mt-4" id="comparison-table" style="display: none;">
                    <div class="card-header">
                        <h5 class="card-title mb-0">
                            <i class="fas fa-table me-2"></i>Run Comparison
                        </h5>
                    </div>
                    <div class="card-body p-0">
                        <div class="table-responsive">
                            <table class="table comparison-table mb-0">
                                <thead>
                                    <tr>
                                        <th>Run Name</th>
                                        <th>Status</th>
                                        <th>Duration</th>
                                        <th>Final Loss</th>
                                        <th>Final Accuracy</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody id="comparison-tbody">
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentUser = null;
            let currentProject = null;
            let selectedRuns = [];
            let allRuns = [];
            let charts = {};
            let socket = null;
            
            // Initialize Socket.IO connection
            function initializeSocket() {
                console.log('Initializing Socket.IO connection...');
                socket = io('https://localhost:8000');
                
                socket.on('connect', () => {
                    console.log('Connected to Socket.IO server with ID:', socket.id);
                });
                
                socket.on('disconnect', () => {
                    console.log('Disconnected from Socket.IO server');
                });
                
                socket.on('connect_error', (error) => {
                    console.error('Socket.IO connection error:', error);
                });
                
                socket.on('training_update', (data) => {
                    console.log('Training update received:', data);
                    updateTrainingStatus(data);
                    updateChartsRealTime(data.metrics, data.run_id);
                });
                
                socket.on('training_complete', (data) => {
                    console.log('Training completed:', data);
                    updateTrainingStatus(data);
                    loadRuns(); // Refresh runs list
                });
                
                socket.on('training_failed', (data) => {
                    console.log('Training failed:', data);
                    updateTrainingStatus(data);
                    loadRuns(); // Refresh runs list
                });
            }
            
            function updateTrainingStatus(data) {
                // Update the running count in the overview
                const runningElement = document.getElementById('running-runs');
                if (runningElement) {
                    const currentRunning = allRuns.filter(r => r.status === 'running').length;
                    runningElement.textContent = currentRunning;
                }
                
                // Update the specific run status in the sidebar
                const runItems = document.querySelectorAll('.run-item');
                runItems.forEach(item => {
                    const runName = item.querySelector('.fw-bold').textContent;
                    if (runName.includes(`training_${data.run_id}`)) {
                        const statusBadge = item.querySelector('.status-badge');
                        if (statusBadge) {
                            statusBadge.textContent = data.status;
                            statusBadge.className = `status-badge status-${data.status}`;
                        }
                    }
                });
            }

            // Global AJAX interceptor for 401 handling
            function setupAjaxInterceptor() {
                // Override fetch to handle 401 responses globally
                const originalFetch = window.fetch;
                window.fetch = async function(...args) {
                    try {
                        const response = await originalFetch(...args);
                        
                        // Check for 401 Unauthorized
                        if (response.status === 401) {
                            console.log('401 Unauthorized detected, redirecting to login');
                            localStorage.removeItem('token');
                            showLogin();
                            return response; // Return response so calling code can handle it
                        }
                        
                        return response;
                    } catch (error) {
                        console.error('Network error:', error);
                        // On network errors, also redirect to login
                        localStorage.removeItem('token');
                        showLogin();
                        throw error;
                    }
                };
            }

            // Authentication
            async function login() {
                const username = document.getElementById('username-input').value;
                const password = document.getElementById('password-input').value;
                
                try {
                    const response = await fetch('https://localhost:8000/auth/login', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        localStorage.setItem('token', data.access_token);
                        currentUser = { username: username };
                        showDashboard();
                        loadProjects();
                    } else {
                        alert('Login failed. Please check your credentials.');
                    }
                } catch (error) {
                    console.error('Login error:', error);
                    alert('Login failed. Please try again.');
                }
            }

            function logout() {
                localStorage.removeItem('token');
                currentUser = null;
                showLogin();
            }

            function showLogin() {
                document.getElementById('login-container').style.display = 'block';
                document.getElementById('dashboard-container').style.display = 'none';
                document.getElementById('user-info').style.display = 'none';
            }

            function showDashboard() {
                document.getElementById('login-container').style.display = 'none';
                document.getElementById('dashboard-container').style.display = 'block';
                document.getElementById('user-info').style.display = 'block';
                document.getElementById('username').textContent = currentUser.username;
                
                // Initialize Socket.IO connection
                initializeSocket();
                
                // Show loading state initially
                document.getElementById('runs-list').innerHTML = '<div class="loading"><div class="spinner"></div></div>';
                document.getElementById('charts-container').innerHTML = '<div class="loading"><div class="spinner"></div><div class="mt-3">Loading charts...</div></div>';
            }

            // Data Loading
            async function loadProjects() {
                try {
                    const response = await fetch('https://localhost:8000/projects/', {
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('token')}`
                        }
                    });
                    
                    if (response.ok) {
                        const projects = await response.json();
                        if (projects.length > 0) {
                            currentProject = projects[0];
                            loadRuns();
                            updateProjectInfo();
                        }
                    }
                } catch (error) {
                    console.error('Error loading projects:', error);
                }
            }

            async function loadRuns() {
                if (!currentProject) return;
                
                try {
                    const response = await fetch(`https://localhost:8000/runs/?project_id=${currentProject.id}`, {
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('token')}`
                        }
                    });
                    
                    if (response.ok) {
                        allRuns = await response.json();
                        updateRunsList();
                        updateMetricsOverview();
                        loadCharts();
                    }
                } catch (error) {
                    console.error('Error loading runs:', error);
                }
            }

            function updateProjectInfo() {
                if (currentProject) {
                    document.getElementById('project-title').textContent = currentProject.name;
                    document.getElementById('project-description').textContent = currentProject.description || 'No description';
                }
            }

            function updateRunsList() {
                const runsList = document.getElementById('runs-list');
                runsList.innerHTML = '';
                
                // Sort runs by creation time (newest first)
                const sortedRuns = [...allRuns].sort((a, b) => b.id - a.id);
                
                // Only show the 5 most recent runs to avoid clutter
                const recentRuns = sortedRuns.slice(0, 5);
                
                console.log('All runs:', allRuns);
                console.log('Sorted runs (newest first):', sortedRuns);
                console.log('Recent runs to display:', recentRuns);
                
                recentRuns.forEach((run, index) => {
                    const runItem = document.createElement('div');
                    runItem.className = `run-item ${run.status}`;
                    
                    // Format creation time
                    const createdTime = new Date(run.created_at).toLocaleString();
                    
                    // Add indicator for the latest run
                    const isLatest = index === 0;
                    const latestIndicator = isLatest ? ' <span class="badge bg-primary">LATEST</span>' : '';
                    
                    runItem.innerHTML = `
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <div class="fw-bold">${run.name}${latestIndicator}</div>
                                <small class="text-muted">ID: ${run.id} - ${createdTime}</small>
                            </div>
                            <span class="status-badge status-${run.status}">${run.status}</span>
                        </div>
                    `;
                    runItem.onclick = () => selectRun(run);
                    runsList.appendChild(runItem);
                });
                
                // Show count of total runs if there are more
                if (sortedRuns.length > 5) {
                    const moreRunsDiv = document.createElement('div');
                    moreRunsDiv.className = 'text-center text-muted mt-2';
                    moreRunsDiv.innerHTML = `<small>Showing 5 of ${sortedRuns.length} runs</small>`;
                    runsList.appendChild(moreRunsDiv);
                }
            }

            function updateMetricsOverview() {
                const total = allRuns.length;
                const completed = allRuns.filter(r => r.status === 'completed').length;
                const running = allRuns.filter(r => r.status === 'running').length;
                const successRate = total > 0 ? Math.round((completed / total) * 100) : 0;
                
                document.getElementById('total-runs').textContent = total;
                document.getElementById('completed-runs').textContent = completed;
                document.getElementById('running-runs').textContent = running;
                document.getElementById('success-rate').textContent = successRate + '%';
            }

            async function loadCharts() {
                const chartsContainer = document.getElementById('charts-container');
                chartsContainer.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
                
                if (allRuns.length === 0) {
                    chartsContainer.innerHTML = '<div class="text-center text-muted">No runs available</div>';
                    return;
                }
                
                // Sort runs by ID (newest first) and get the latest run
                const sortedRuns = [...allRuns].sort((a, b) => b.id - a.id);
                const latestRun = sortedRuns[0];
                
                console.log('Latest run:', latestRun);
                
                // Visually select the latest run in the sidebar
                selectRunVisually(latestRun);
                
                // Load metrics for the latest run (if it has metrics)
                if (latestRun.status === 'completed') {
                    await loadRunMetrics(latestRun.id);
                } else {
                    // Show message for running or pending runs
                    chartsContainer.innerHTML = `
                        <div class="text-center">
                            <div class="alert alert-info">
                                <h5>Latest Run: ${latestRun.name}</h5>
                                <p>Status: <span class="badge bg-${latestRun.status === 'running' ? 'warning' : 'secondary'}">${latestRun.status}</span></p>
                                <p>This run is currently ${latestRun.status}. Charts will appear when training completes.</p>
                            </div>
                        </div>
                    `;
                }
            }

            async function loadRunMetrics(runId) {
                try {
                    const response = await fetch(`https://localhost:8000/metrics/${runId}`, {
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('token')}`
                        }
                    });
                    
                    if (response.ok) {
                        const metrics = await response.json();
                        createCharts(metrics, runId);
                    } else {
                        // Show error message for other status codes
                        console.error('Error loading metrics:', response.status);
                    }
                } catch (error) {
                    console.error('Error loading metrics:', error);
                }
            }

            function createCharts(metrics, runId) {
                const chartsContainer = document.getElementById('charts-container');
                chartsContainer.innerHTML = '';
                
                // Group metrics by name
                const metricGroups = {};
                metrics.forEach(metric => {
                    if (!metricGroups[metric.name]) {
                        metricGroups[metric.name] = [];
                    }
                    metricGroups[metric.name].push(metric);
                });
                
                // Create charts for each metric
                Object.keys(metricGroups).forEach(metricName => {
                    const metricData = metricGroups[metricName];
                    const chartDiv = document.createElement('div');
                    chartDiv.className = 'chart-container';
                    chartDiv.id = `chart-${metricName}`;
                    
                    const chartTitle = document.createElement('h5');
                    chartTitle.textContent = metricName.replace(/_/g, ' ').toUpperCase();
                    chartDiv.appendChild(chartTitle);
                    
                    chartsContainer.appendChild(chartDiv);
                    
                    // Small delay to ensure container is properly sized
                    setTimeout(() => {
                        // Format x-axis values properly
                        const xValues = metricData.map((m, index) => {
                            // Use step number directly, or index if step is not available
                            return m.step || index;
                        });
                        
                        // Debug: Log the data to see what we're working with
                        console.log(`Chart data for ${metricName}:`, {
                            xValues: xValues,
                            yValues: metricData.map(m => m.value),
                            metricData: metricData
                        });
                        
                        // Create Plotly chart with better formatting
                        const trace = {
                        x: xValues,
                        y: metricData.map(m => m.value),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: `Run ${runId}`,
                        line: { color: '#6366f1', width: 2 },
                        marker: { size: 6, color: '#6366f1' }
                    };
                    
                    const layout = {
                        title: {
                            text: metricName.replace(/_/g, ' ').toUpperCase(),
                            font: { size: 16, color: '#f9fafb' }
                        },
                        xaxis: { 
                            title: { text: 'Step', font: { color: '#f9fafb' } },
                            tickfont: { color: '#f9fafb' },
                            gridcolor: '#4b5563',
                            zerolinecolor: '#4b5563',
                            type: 'linear'
                        },
                        yaxis: { 
                            title: { text: metricName.replace(/_/g, ' ').toUpperCase(), font: { color: '#f9fafb' } },
                            tickfont: { color: '#f9fafb' },
                            gridcolor: '#4b5563',
                            zerolinecolor: '#4b5563',
                            type: 'linear'
                        },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#f9fafb' },
                        margin: { t: 60, r: 20, b: 60, l: 80 },
                        hovermode: 'x unified',
                        showlegend: false,
                        height: 350,
                        autosize: true
                    };
                    
                    const config = {
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                        displaylogo: false,
                        useResizeHandler: true
                    };
                    
                    Plotly.newPlot(chartDiv.id, [trace], layout, config);
                    }, 100); // Small delay for proper sizing
                });
            }

            function selectRunVisually(run) {
                // Remove active class from all runs
                document.querySelectorAll('.run-item').forEach(item => {
                    item.classList.remove('active');
                });
                
                // Add active class to the run item that matches the run ID
                const runItems = document.querySelectorAll('.run-item');
                runItems.forEach(item => {
                    if (item.textContent.includes(run.name)) {
                        item.classList.add('active');
                    }
                });
            }
            
            function selectRun(run) {
                // Remove active class from all runs
                document.querySelectorAll('.run-item').forEach(item => {
                    item.classList.remove('active');
                });
                
                // Add active class to selected run
                event.currentTarget.classList.add('active');
                
                // Load metrics for selected run
                loadRunMetrics(run.id);
            }

            function refreshData() {
                loadRuns();
            }

            async function triggerTraining() {
                if (!currentProject) {
                    alert('Please select a project first');
                    return;
                }
                
                console.log('Starting training for project:', currentProject);
                
                try {
                    const response = await fetch(`https://localhost:8000/training/start/${currentProject.id}`, {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('token')}`
                        }
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        console.log('Training started successfully:', result);
                        alert(`Training started! Run ID: ${result.run_id}`);
                        
                        // Join Socket.IO room for this run
                        if (socket) {
                            console.log('Joining Socket.IO room for run:', result.run_id);
                            socket.emit('join_run', { run_id: result.run_id });
                        } else {
                            console.error('Socket not connected!');
                        }
                        
                        // Refresh the runs list after a short delay
                        setTimeout(() => {
                            console.log('Refreshing runs list...');
                            loadRuns();
                        }, 1000);
                    } else {
                        console.error('Failed to start training:', response.status);
                        alert('Failed to start training');
                    }
                } catch (error) {
                    console.error('Error starting training:', error);
                    alert('Error starting training');
                }
            }
            
            async function monitorTraining(runId) {
                // Join Socket.IO room for real-time updates
                if (socket) {
                    socket.emit('join_run', { run_id: runId });
                }
            }
            
            function updateChartsRealTime(metrics, runId) {
                // Update existing charts with new data
                Object.keys(metrics).forEach(metricName => {
                    const chartDiv = document.getElementById(`chart-${metricName}`);
                    if (chartDiv) {
                        // Add new data point to existing chart
                        Plotly.extendTraces(chartDiv.id, {
                            x: [[metrics[metricName].step]],
                            y: [[metrics[metricName].value]]
                        });
                    }
                });
            }
            
            function exportData() {
                const data = {
                    project: currentProject,
                    runs: allRuns,
                    timestamp: new Date().toISOString()
                };
                
                const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `ai-manager-export-${new Date().toISOString().split('T')[0]}.json`;
                a.click();
                URL.revokeObjectURL(url);
            }

            // Model Testing Functions
            async function uploadModel() {
                const fileInput = document.getElementById('model-file');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a model file to upload');
                    return;
                }
                
                const formData = new FormData();
                formData.append('model', file);
                
                try {
                    const response = await fetch('https://localhost:8000/models/upload', {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('token')}`
                        },
                        body: formData
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        alert(`Model uploaded successfully! Model ID: ${result.model_id}`);
                        showTestResults(`Model uploaded: ${file.name}\nModel ID: ${result.model_id}`);
                    } else {
                        alert('Failed to upload model');
                    }
                } catch (error) {
                    console.error('Error uploading model:', error);
                    alert('Error uploading model');
                }
            }

            async function loadDemoModel(modelName) {
                try {
                    const response = await fetch(`https://localhost:8000/models/load/${modelName}`, {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('token')}`
                        }
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        alert(`Demo model loaded successfully! Model: ${modelName}`);
                        showTestResults(`Demo model loaded: ${modelName}\nModel info: ${JSON.stringify(result, null, 2)}`);
                    } else {
                        alert('Failed to load demo model');
                    }
                } catch (error) {
                    console.error('Error loading demo model:', error);
                    alert('Error loading demo model');
                }
            }

            async function testModel() {
                const dataInput = document.getElementById('test-data-file');
                const dataFile = dataInput.files[0];
                
                if (!dataFile) {
                    alert('Please select test data file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('test_data', dataFile);
                
                try {
                    const response = await fetch('https://localhost:8000/models/test', {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('token')}`
                        },
                        body: formData
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        showTestResults(`Test completed successfully!\n\nResults:\n${JSON.stringify(result, null, 2)}`);
                    } else {
                        alert('Failed to test model');
                    }
                } catch (error) {
                    console.error('Error testing model:', error);
                    alert('Error testing model');
                }
            }

            function showTestResults(message) {
                const resultsDiv = document.getElementById('test-results');
                const outputDiv = document.getElementById('test-output');
                
                outputDiv.innerHTML = message.replace(/\n/g, '<br>');
                resultsDiv.style.display = 'block';
                
                // Scroll to results
                resultsDiv.scrollIntoView({ behavior: 'smooth' });
            }

            // Event Listeners
            document.getElementById('login-form').addEventListener('submit', function(e) {
                e.preventDefault();
                login();
            });

            // Setup global AJAX interceptor
            setupAjaxInterceptor();
            
            // Check if user is already logged in
            const token = localStorage.getItem('token');
            if (token) {
                currentUser = { username: 'admin' }; // Default for demo
                showDashboard();
                // Validate token by trying to load projects
                validateTokenAndLoadData();
            } else {
                showLogin();
            }
            
            async function validateTokenAndLoadData() {
                try {
                    const response = await fetch('https://localhost:8000/projects/', {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    });
                    
                    if (response.ok) {
                        loadProjects();
                    }
                } catch (error) {
                    console.error('Error validating token:', error);
                }
            }
        </script>
    </body>
    </html>
    """


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


@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "2023-01-01T00:00:00"}

# Export the Socket.IO ASGI app (wraps FastAPI app)
app = socket_app 