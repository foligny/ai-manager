// Dashboard Module
class DashboardManager {
    constructor() {
        this.currentProject = null;
        this.projects = [];
        this.runs = [];
        this.models = [];
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
            await this.loadModels();
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

    async loadModels() {
        try {
            console.log('Loading models...');
            const response = await api.getModels();
            console.log('Models API response:', response);
            this.models = response.models || [];
            console.log('Models loaded:', this.models.length, 'models');
            this.renderModels();
        } catch (error) {
            console.error('Failed to load models:', error);
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
                <div class="project-item mb-2 p-2 rounded bg-secondary text-light" 
                     data-project-id="${project.id}">
                    <div class="d-flex justify-content-between align-items-center">
                        <div class="flex-grow-1 cursor-pointer" onclick="dashboard.selectProject(${project.id})">
                            <span><i class="fas fa-folder me-2"></i>${project.name}</span>
                        </div>
                        <div class="d-flex align-items-center">
                            <span class="badge bg-primary me-2">${project.runs_count || 0}</span>
                            <button class="btn btn-sm btn-outline-light" onclick="dashboard.editProject(${project.id})">
                                <i class="fas fa-edit"></i>
                            </button>
                        </div>
                    </div>
                    ${tagsHtml}
                </div>
            `;
        }).join('');
    }

    renderModels() {
        const modelsList = document.getElementById('models-list');
        if (!modelsList) {
            console.error('Models list element not found');
            return;
        }
        console.log('Rendering', this.models.length, 'models');

        modelsList.innerHTML = this.models.map(model => {
            const sizeMB = (model.size / (1024 * 1024)).toFixed(1);
            const capabilitiesHtml = model.capabilities && model.capabilities.length > 0 
                ? `<div class="mt-1"><small>${model.capabilities.map(cap => `<span class="badge bg-success me-1">${cap}</span>`).join('')}</small></div>`
                : '';
            
            const tagsHtml = model.tags && model.tags.length > 0 
                ? `<div class="mt-1"><small>${model.tags.map(tag => `<span class="badge bg-info me-1">${tag}</span>`).join('')}</small></div>`
                : '';
            
            return `
                <div class="model-item mb-2 p-2 rounded bg-secondary text-light cursor-pointer" 
                     onclick="dashboard.selectModel('${model.name}')">
                    <div class="d-flex justify-content-between align-items-center">
                        <span><i class="fas fa-robot me-2"></i>${model.name}</span>
                        <span class="badge bg-primary">${sizeMB}MB</span>
                    </div>
                    <div class="mt-1">
                        <small class="text-muted">Type: ${model.type}</small>
                    </div>
                    ${capabilitiesHtml}
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
        const totalModels = this.models.length;
        const runningCount = this.runs.filter(run => run.status === 'running').length;
        const completedCount = this.runs.filter(run => run.status === 'completed').length;

        document.getElementById('total-projects').textContent = totalProjects;
        document.getElementById('total-runs').textContent = totalRuns;
        document.getElementById('total-models').textContent = totalModels;
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

    async selectModel(modelName) {
        try {
            // Navigate to inference page with the selected model
            window.location.href = `/inference?model=${encodeURIComponent(modelName)}`;
        } catch (error) {
            console.error('Failed to select model:', error);
        }
    }

    async editProject(projectId) {
        try {
            const project = this.projects.find(p => p.id === projectId);
            if (!project) {
                console.error('Project not found:', projectId);
                return;
            }

            // Populate modal with project data
            document.getElementById('project-name').value = project.name;
            document.getElementById('project-description').value = project.description || '';
            document.getElementById('project-tags').value = project.tags ? project.tags.join(', ') : '';

            // Store current project ID
            this.editingProjectId = projectId;

            // Populate available models dropdown
            this.populateAvailableModelsDropdown();

            // Load assigned models
            await this.loadAssignedModels(projectId);

            // Show modal
            const modal = new bootstrap.Modal(document.getElementById('projectModal'));
            modal.show();
        } catch (error) {
            console.error('Failed to edit project:', error);
        }
    }

    populateAvailableModelsDropdown() {
        const select = document.getElementById('available-models-select');
        select.innerHTML = '<option value="">Select a model to add...</option>';
        
        this.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = `${model.name} (${model.type})`;
            select.appendChild(option);
        });
    }

    async loadAssignedModels(projectId) {
        try {
            console.log('Loading assigned models for project:', projectId);
            
            // Get assigned models from API
            const response = await fetch(`/projects/${projectId}/models`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });

            if (response.ok) {
                const assignedModels = await response.json();
                console.log('Assigned models loaded:', assignedModels);
                
                const container = document.getElementById('assigned-models');
                if (assignedModels.length === 0) {
                    container.innerHTML = '<small class="text-muted">No models assigned to this project</small>';
                } else {
                                    container.innerHTML = assignedModels.map(model => {
                    const tagsInfo = model.model_capabilities && model.model_capabilities.length > 0 ? 
                        `<small class="text-muted d-block">Capabilities: ${model.model_capabilities.join(', ')}</small>` : '';
                    return `
                        <div class="d-flex justify-content-between align-items-center mb-2 p-2 bg-secondary rounded">
                            <div>
                                <span><i class="fas fa-robot me-2"></i>${model.model_name}</span>
                                ${tagsInfo}
                            </div>
                            <button class="btn btn-sm btn-outline-danger" onclick="event.preventDefault(); dashboard.removeModelFromProject('${model.model_name}')" type="button">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    `;
                }).join('');
                }
                console.log('Assigned models HTML updated');
            } else {
                console.error('Failed to load assigned models:', response.status);
            }
        } catch (error) {
            console.error('Failed to load assigned models:', error);
        }
    }

    async addModelToProject() {
        const select = document.getElementById('available-models-select');
        const modelName = select.value;
        
        if (!modelName) {
            alert('Please select a model to add');
            return;
        }

        try {
            // Find the model in our models list to get its details
            const model = this.models.find(m => m.name === modelName);
            if (!model) {
                throw new Error('Model not found');
            }

            // Call API to assign model to project
            const response = await fetch(`/projects/${this.editingProjectId}/models`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({
                    model_name: modelName,
                    model_path: model.path || '',
                    model_type: model.type || 'unknown',
                    model_capabilities: model.capabilities || []
                })
            });

            if (response.ok) {
                // Refresh the assigned models list
                await this.loadAssignedModels(this.editingProjectId);
                
                // Reset dropdown
                select.value = '';
                
                this.showSuccess(`Model ${modelName} added to project successfully!`);
            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to add model to project');
            }
        } catch (error) {
            console.error('Failed to add model to project:', error);
            this.showError('Failed to add model to project: ' + error.message);
        }
    }

    async removeModelFromProject(modelName) {
        try {
            console.log('Removing model:', modelName, 'from project:', this.editingProjectId);
            
            // Call API to remove model from project
            const response = await fetch(`/projects/${this.editingProjectId}/models/${encodeURIComponent(modelName)}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });

            if (response.ok) {
                console.log('Model removed successfully, refreshing list...');
                
                // Refresh the assigned models list
                await this.loadAssignedModels(this.editingProjectId);
                
                this.showSuccess(`Model ${modelName} removed from project successfully!`);
            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to remove model from project');
            }
        } catch (error) {
            console.error('Failed to remove model from project:', error);
            this.showError('Failed to remove model from project: ' + error.message);
        }
    }

    async importFromHuggingFace() {
        const modelName = document.getElementById('huggingface-model').value.trim();
        
        if (!modelName) {
            alert('Please enter a Hugging Face model name');
            return;
        }

        try {
            this.showLoading('Importing model from Hugging Face...');
            
            // This would call the backend API to download from Hugging Face
            const response = await fetch('/models/import-huggingface', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({ 
                    model_name: modelName,
                    project_id: this.editingProjectId 
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.showSuccess(`Model ${modelName} imported successfully!`);
                
                // Add to assigned models
                const container = document.getElementById('assigned-models');
                const tagsInfo = result.tags && result.tags.length > 0 ? 
                    `<small class="text-muted d-block">Tags: ${result.tags.join(', ')}</small>` : '';
                const modelHtml = `
                    <div class="d-flex justify-content-between align-items-center mb-2 p-2 bg-secondary rounded">
                        <div>
                            <span><i class="fas fa-robot me-2"></i>${result.model_name}</span>
                            ${tagsInfo}
                        </div>
                        <button class="btn btn-sm btn-outline-danger" onclick="event.preventDefault(); dashboard.removeModelFromProject('${result.model_name}')" type="button">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                `;
                
                if (container.innerHTML.includes('No models assigned')) {
                    container.innerHTML = modelHtml;
                } else {
                    container.innerHTML += modelHtml;
                }

                // Clear input
                document.getElementById('huggingface-model').value = '';
                
                // Refresh models list
                await this.loadModels();
            } else {
                throw new Error('Failed to import model');
            }
        } catch (error) {
            console.error('Failed to import model:', error);
            this.showError('Failed to import model from Hugging Face');
        }
    }

    async saveProject() {
        try {
            const projectData = {
                name: document.getElementById('project-name').value,
                description: document.getElementById('project-description').value,
                tags: document.getElementById('project-tags').value.split(',').map(tag => tag.trim()).filter(tag => tag)
            };

            // Call the backend API to update the project
            const response = await fetch(`/projects/${this.editingProjectId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify(projectData)
            });

            if (response.ok) {
                const updatedProject = await response.json();
                console.log('Project saved successfully:', updatedProject);
                
                this.showSuccess('Project saved successfully!');
                
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('projectModal'));
                modal.hide();
                
                // Refresh projects
                await this.loadProjects();
            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to save project');
            }
        } catch (error) {
            console.error('Failed to save project:', error);
            this.showError('Failed to save project: ' + error.message);
        }
    }

    showLoading(message) {
        // Simple loading indicator
        console.log('Loading:', message);
    }

    showSuccess(message) {
        // Show success message as a toast notification
        this.showToast(message, 'success');
    }

    showError(message) {
        // Show error message as a toast notification
        this.showToast(message, 'error');
    }

    showToast(message, type = 'info') {
        // Create a toast notification that doesn't interfere with modals
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type === 'success' ? 'success' : type === 'error' ? 'danger' : 'info'} border-0 position-fixed`;
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 250px;';
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        document.body.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast, { delay: 3000 });
        bsToast.show();
        
        // Remove the toast element after it's hidden
        toast.addEventListener('hidden.bs.toast', () => {
            document.body.removeChild(toast);
        });
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