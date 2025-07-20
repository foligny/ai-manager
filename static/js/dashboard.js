// Dashboard Module
class DashboardManager {
    constructor() {
        this.currentProject = null;
        this.projects = [];
        this.runs = [];
        this.socket = null;
        this.setupSocketIO();
        this.loadDashboard();
    }

    setupSocketIO() {
        // Initialize Socket.IO connection
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to Socket.IO');
        });

        this.socket.on('metric_update', (data) => {
            this.handleMetricUpdate(data);
        });

        this.socket.on('run_status_update', (data) => {
            this.handleRunStatusUpdate(data);
        });
    }

    async loadDashboard() {
        try {
            await this.loadProjects();
            await this.updateOverview();
        } catch (error) {
            console.error('Failed to load dashboard:', error);
        }
    }

    async loadProjects() {
        try {
            console.log('Loading projects...');
            this.projects = await api.getProjects();
            console.log('Projects loaded:', this.projects);
            this.renderProjects();
            
            if (this.projects.length > 0) {
                this.currentProject = this.projects[0];
                await this.loadRuns();
            }
        } catch (error) {
            console.error('Failed to load projects:', error);
        }
    }

    async loadRuns() {
        if (!this.currentProject) return;

        try {
            this.runs = await api.getRuns(this.currentProject.id);
            this.renderRuns();
            
            if (this.runs.length > 0) {
                const latestRun = this.runs[0];
                await this.loadRunMetrics(latestRun.id);
            }
        } catch (error) {
            console.error('Failed to load runs:', error);
        }
    }

    async loadRunMetrics(runId) {
        try {
            const metrics = await api.getMetrics(runId);
            if (metrics.length > 0) {
                window.chartsManager.renderCharts(metrics, runId);
            }
        } catch (error) {
            console.error('Failed to load metrics:', error);
        }
    }

    renderProjects() {
        const projectsList = document.getElementById('projects-list');
        if (!projectsList) return;

        projectsList.innerHTML = this.projects.map(project => {
            const tagsHtml = project.tags && project.tags.length > 0 
                ? `<div class="mt-1"><small>${project.tags.map(tag => `<span class="badge bg-info me-1">${tag}</span>`).join('')}</small></div>`
                : '';
            
            return `
                <div class="project-item mb-2 p-2 rounded bg-secondary text-light cursor-pointer" 
                     onclick="dashboard.selectProject(${project.id})">
                    <div class="d-flex justify-content-between align-items-center">
                        <span><i class="fas fa-folder me-2"></i>${project.name}</span>
                        <span class="badge bg-primary">${project.runs_count || 0}</span>
                    </div>
                    ${tagsHtml}
                </div>
            `;
        }).join('');
    }

    renderRuns() {
        const runsList = document.getElementById('runs-list');
        if (!runsList) return;

        // Sort runs by creation time (newest first)
        const sortedRuns = [...this.runs].sort((a, b) => b.id - a.id);
        
        // Only show the 5 most recent runs to avoid clutter
        const recentRuns = sortedRuns.slice(0, 5);

        runsList.innerHTML = recentRuns.map(run => {
            const statusClass = run.status === 'running' ? 'bg-warning' : 
                              run.status === 'completed' ? 'bg-success' : 'bg-secondary';
            
            // Format creation time
            const createdAt = new Date(run.created_at).toLocaleString();
            
            // Render tags if they exist
            const tagsHtml = run.tags && run.tags.length > 0 
                ? `<div class="mt-1"><small>${run.tags.map(tag => `<span class="badge bg-info me-1">${tag}</span>`).join('')}</small></div>`
                : '';
            
            return `
                <div class="run-item mb-2 p-2 rounded ${statusClass} text-light cursor-pointer" 
                     onclick="dashboard.selectRun(${run.id})">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <div><i class="fas fa-play me-2"></i>${run.name}</div>
                            <small class="text-muted">${createdAt}</small>
                        </div>
                        <span class="badge ${statusClass === 'bg-warning' ? 'bg-warning' : 'bg-light text-dark'}">
                            ${run.status}
                        </span>
                    </div>
                    ${tagsHtml}
                </div>
            `;
        }).join('');

        // Show count of total runs if there are more
        if (this.runs.length > 5) {
            runsList.innerHTML += `
                <div class="text-center text-muted mt-2">
                    <small>+${this.runs.length - 5} more runs</small>
                </div>
            `;
        }
    }

    async updateOverview() {
        const totalProjects = this.projects.length;
        const totalRuns = this.runs.length;
        const runningCount = this.runs.filter(run => run.status === 'running').length;
        const completedCount = this.runs.filter(run => run.status === 'completed').length;

        document.getElementById('total-projects').textContent = totalProjects;
        document.getElementById('total-runs').textContent = totalRuns;
        document.getElementById('running-count').textContent = runningCount;
        document.getElementById('completed-count').textContent = completedCount;
    }

    async selectProject(projectId) {
        this.currentProject = this.projects.find(p => p.id === projectId);
        if (this.currentProject) {
            await this.loadRuns();
        }
    }

    async selectRun(runId) {
        // Remove active class from all runs
        document.querySelectorAll('.run-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // Add active class to selected run
        event.currentTarget.classList.add('active');
        
        // Load metrics for selected run
        await this.loadRunMetrics(runId);
    }

    handleMetricUpdate(data) {
        console.log('Metric update received:', data);
        // Update charts with new data
        if (window.chartsManager) {
            window.chartsManager.updateChart(data);
        }
    }

    handleRunStatusUpdate(data) {
        console.log('Run status update received:', data);
        // Refresh runs list
        this.loadRuns();
    }
}

// Global dashboard manager instance
const dashboard = new DashboardManager(); 