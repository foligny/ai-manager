// API Client Module
class APIClient {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
        this.token = localStorage.getItem('token');
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        // Add auth token if available
        if (this.token) {
            config.headers.Authorization = `Bearer ${this.token}`;
        }

        try {
            const response = await fetch(url, config);
            
            // Handle 401 Unauthorized
            if (response.status === 401) {
                localStorage.removeItem('token');
                window.location.href = '/login';
                return;
            }

            return response;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    // Authentication
    async login(username, password) {
        const response = await this.request('/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
        });

        if (response.ok) {
            const data = await response.json();
            this.token = data.access_token;
            localStorage.setItem('token', this.token);
            return data;
        } else {
            throw new Error('Login failed');
        }
    }

    async logout() {
        this.token = null;
        localStorage.removeItem('token');
    }

    // Projects
    async getProjects() {
        console.log('API: Getting projects...');
        const response = await this.request('/projects/');
        console.log('API: Projects response:', response.status, response.ok);
        const data = response.ok ? await response.json() : [];
        console.log('API: Projects data:', data);
        return data;
    }

    async createProject(name, description = '') {
        const response = await this.request('/projects/', {
            method: 'POST',
            body: JSON.stringify({ name, description })
        });
        return response.ok ? await response.json() : null;
    }

    // Runs
    async getRuns(projectId) {
        const response = await this.request(`/runs/?project_id=${projectId}`);
        return response.ok ? await response.json() : [];
    }

    async createRun(projectId, name, description = '') {
        const response = await this.request(`/runs/?project_id=${projectId}`, {
            method: 'POST',
            body: JSON.stringify({ name, description })
        });
        return response.ok ? await response.json() : null;
    }

    // Metrics
    async getMetrics(runId) {
        const response = await this.request(`/metrics/${runId}`);
        return response.ok ? await response.json() : [];
    }

    async logMetrics(runId, metrics) {
        const response = await this.request(`/metrics/${runId}`, {
            method: 'POST',
            body: JSON.stringify(metrics)
        });
        return response.ok;
    }

    // Training
    async startTraining(projectId) {
        const response = await this.request(`/training/start/${projectId}`, {
            method: 'POST'
        });
        return response.ok ? await response.json() : null;
    }

    async getTrainingStatus(runId) {
        const response = await this.request(`/training/status/${runId}`);
        return response.ok ? await response.json() : null;
    }

    // Models
    async getModels() {
        const response = await this.request('/models/list');
        return response.ok ? await response.json() : { models: [] };
    }

    async uploadModel(formData) {
        const response = await this.request('/models/upload', {
            method: 'POST',
            headers: {}, // Let browser set Content-Type for FormData
            body: formData
        });
        return response.ok ? await response.json() : null;
    }

    async testModel(formData) {
        const response = await this.request('/models/test', {
            method: 'POST',
            headers: {}, // Let browser set Content-Type for FormData
            body: formData
        });
        return response.ok ? await response.json() : null;
    }
}

// Global API client instance
const api = new APIClient(); 