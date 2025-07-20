// Authentication Module
class AuthManager {
    constructor() {
        this.currentUser = null;
        this.setupEventListeners();
        this.checkAuthStatus();
    }

    setupEventListeners() {
        const loginForm = document.getElementById('login-form');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleLogin();
            });
        }
    }

    async handleLogin() {
        const username = document.getElementById('username-input').value;
        const password = document.getElementById('password-input').value;

        try {
            const data = await api.login(username, password);
            this.currentUser = { username };
            this.showDashboard();
        } catch (error) {
            alert('Login failed. Please check your credentials.');
            console.error('Login error:', error);
        }
    }

    async checkAuthStatus() {
        const token = localStorage.getItem('token');
        if (token) {
            try {
                // Validate token by trying to load projects
                const projects = await api.getProjects();
                this.currentUser = { username: 'admin' };
                // Only redirect to dashboard if we're on the login page
                if (window.location.pathname === '/login') {
                    this.showDashboard();
                }
            } catch (error) {
                console.error('Token validation failed:', error);
                localStorage.removeItem('token'); // Clear invalid token
                this.showLogin();
            }
        } else {
            // Only redirect to login if we're not already on the login page
            if (window.location.pathname !== '/login') {
                this.showLogin();
            }
        }
    }

    showLogin() {
        // Only redirect to login if we're not already on the login page
        if (window.location.pathname !== '/login') {
            window.location.href = '/login';
        }
    }

    showDashboard() {
        window.location.href = '/';
    }

    logout() {
        api.logout();
        this.currentUser = null;
        this.showLogin();
    }
}

// Global auth manager instance
const auth = new AuthManager();

// Global logout function
function logout() {
    auth.logout();
} 