// Inference Interface Module
class InferenceManager {
    constructor() {
        this.currentModel = null;
        this.models = [];
        this.setupEventListeners();
        this.checkAuthAndLoadModels();
    }

    async checkAuthAndLoadModels() {
        // Check if user is logged in
        const token = localStorage.getItem('token');
        if (!token) {
            console.log('No token found, redirecting to login...');
            window.location.href = '/login';
            return;
        }
        
        console.log('Token found, loading models...');
        await this.loadModels();
        
        // Check for model parameter in URL
        const urlParams = new URLSearchParams(window.location.search);
        const modelParam = urlParams.get('model');
        if (modelParam) {
            console.log('Model parameter found:', modelParam);
            this.selectModel(modelParam);
        }
    }

    setupEventListeners() {
        // Model upload form
        document.getElementById('upload-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.uploadModel();
        });

        // Image inference form
        document.getElementById('image-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.runImageInference();
        });

        // Data testing form
        document.getElementById('data-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.runDataTest();
        });

        // Image preview
        document.getElementById('image-input').addEventListener('change', (e) => {
            this.previewImage(e.target.files[0]);
        });

        // Data preview
        document.getElementById('data-input').addEventListener('change', (e) => {
            this.previewData(e.target.files[0]);
        });

        // Audio preview
        document.getElementById('audio-input').addEventListener('change', (e) => {
            this.previewAudio(e.target.files[0]);
        });

        // Audio inference form
        document.getElementById('audio-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.runAudioInference();
        });
    }

    async loadModels() {
        try {
            console.log('Loading models...');
            const response = await api.getModels();
            console.log('Models response:', response);
            this.models = response.models || [];
            console.log('Models loaded:', this.models);
            this.renderModels();
        } catch (error) {
            console.error('Failed to load models:', error);
            if (error.message.includes('401') || error.message.includes('Unauthorized')) {
                console.log('Authentication failed, redirecting to login...');
                window.location.href = '/login';
            } else {
                this.showError('Failed to load models');
            }
        }
    }

    renderModels() {
        const modelsList = document.getElementById('models-list');
        
        if (this.models.length === 0) {
            modelsList.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-robot fa-2x mb-2"></i>
                    <p>No models available</p>
                    <small>Upload a model to get started</small>
                </div>
            `;
            return;
        }

        modelsList.innerHTML = this.models.map(model => {
            const capabilities = model.capabilities || [];
            const formats = model.supported_formats || [];
            
            const capabilityBadges = capabilities.map(cap => 
                `<span class="badge bg-info me-1 mb-1">${cap}</span>`
            ).join('');
            
            const formatBadges = formats.map(format => 
                `<span class="badge bg-secondary me-1 mb-1">.${format}</span>`
            ).join('');
            
            return `
                <div class="model-item mb-3 p-3 rounded bg-secondary text-light cursor-pointer" 
                     data-model-name="${model.name}"
                     onclick="inference.selectModel('${model.name}')">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <div><i class="fas fa-robot me-2"></i>${model.name}</div>
                            <small class="text-muted">${this.formatFileSize(model.size)} • ${model.type}</small>
                            <div class="mt-2">
                                <small class="text-muted">Capabilities:</small><br>
                                ${capabilityBadges}
                            </div>
                            <div class="mt-1">
                                <small class="text-muted">Formats:</small><br>
                                ${formatBadges}
                            </div>
                        </div>
                        <span class="badge bg-primary">Select</span>
                    </div>
                </div>
            `;
        }).join('');
    }

    async uploadModel() {
        const fileInput = document.getElementById('model-file');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showError('Please select a model file');
            return;
        }

        try {
            this.showLoading('Uploading model...');
            
            const formData = new FormData();
            formData.append('model', file);
            
            const response = await api.uploadModel(formData);
            
            this.showSuccess('Model uploaded successfully!');
            this.loadModels(); // Refresh model list
            
        } catch (error) {
            console.error('Failed to upload model:', error);
            this.showError('Failed to upload model');
        }
    }

    selectModel(modelName) {
        this.currentModel = modelName;
        
        // Find the selected model
        const selectedModel = this.models.find(model => model.name === modelName);
        if (!selectedModel) {
            console.warn(`Model not found: ${modelName}`);
            return;
        }
        
        // Update UI to show selected model
        document.querySelectorAll('.model-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // Add active class to the correct model item
        const modelItem = document.querySelector(`[data-model-name="${modelName}"]`);
        if (modelItem) {
            modelItem.classList.add('active');
        }
        
        // Show/hide upload sections based on model capabilities
        this.updateUploadSections(selectedModel);
        
        this.showSuccess(`Selected model: ${modelName}`);
    }

    updateUploadSections(model) {
        const capabilities = model.capabilities || [];
        const formats = model.supported_formats || [];
        
        // Hide the "no model selected" message
        const noModelMessage = document.getElementById('no-model-message');
        if (noModelMessage) {
            noModelMessage.style.display = 'none';
        }
        
        // Image upload section
        const imageSection = document.getElementById('image-inference-section');
        if (imageSection) {
            if (capabilities.some(cap => cap.includes('image'))) {
                imageSection.style.display = 'block';
                // Update accepted file types
                const imageInput = document.getElementById('image-input');
                if (imageInput) {
                    const imageFormats = formats.filter(f => ['jpg', 'jpeg', 'png', 'bmp', 'tiff'].includes(f));
                    imageInput.accept = imageFormats.map(f => `.${f}`).join(',');
                }
            } else {
                imageSection.style.display = 'none';
            }
        }
        
        // Data upload section
        const dataSection = document.getElementById('data-inference-section');
        if (dataSection) {
            if (capabilities.some(cap => ['text', 'tabular', 'csv', 'generic'].includes(cap))) {
                dataSection.style.display = 'block';
                // Update accepted file types
                const dataInput = document.getElementById('data-input');
                if (dataInput) {
                    const dataFormats = formats.filter(f => ['csv', 'json', 'npy', 'txt', 'xlsx'].includes(f));
                    dataInput.accept = dataFormats.map(f => `.${f}`).join(',');
                }
            } else {
                dataSection.style.display = 'none';
            }
        }
        
        // Audio upload section
        const audioSection = document.getElementById('audio-inference-section');
        if (audioSection) {
            if (capabilities.some(cap => cap.includes('audio'))) {
                audioSection.style.display = 'block';
                // Update accepted file types
                const audioInput = document.getElementById('audio-input');
                if (audioInput) {
                    const audioFormats = formats.filter(f => ['wav', 'mp3', 'flac', 'm4a'].includes(f));
                    audioInput.accept = audioFormats.map(f => `.${f}`).join(',');
                }
            } else {
                audioSection.style.display = 'none';
            }
        }
    }

    previewImage(file) {
        if (!file) return;

        const reader = new FileReader();
        const preview = document.getElementById('image-preview');
        
        reader.onload = (e) => {
            preview.innerHTML = `
                <img src="${e.target.result}" class="img-fluid" style="max-height: 180px;">
                <p class="text-muted mt-2">${file.name}</p>
            `;
        };
        
        reader.readAsDataURL(file);
    }

    previewData(file) {
        if (!file) return;

        const preview = document.getElementById('data-preview');
        preview.innerHTML = `
            <div class="text-center">
                <i class="fas fa-file fa-2x text-muted mb-2"></i>
                <p class="text-muted">${file.name}</p>
                <small class="text-muted">${this.formatFileSize(file.size)}</small>
            </div>
        `;
    }

    previewAudio(file) {
        if (!file) return;

        const preview = document.getElementById('audio-preview');
        preview.innerHTML = `
            <div class="text-center">
                <i class="fas fa-microphone fa-2x text-muted mb-2"></i>
                <p class="text-muted">${file.name}</p>
                <small class="text-muted">${this.formatFileSize(file.size)}</small>
                <audio controls class="mt-2 w-100">
                    <source src="${URL.createObjectURL(file)}" type="${file.type}">
                    Your browser does not support the audio element.
                </audio>
            </div>
        `;
    }

    async runImageInference() {
        if (!this.currentModel) {
            this.showError('Please select a model first');
            return;
        }

        const fileInput = document.getElementById('image-input');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showError('Please select an image');
            return;
        }

        try {
            this.showLoading('Running image inference...');
            
            // For now, we'll simulate image inference
            // In a real implementation, you'd send the image to your model API
            await this.simulateInference('image', file);
            
        } catch (error) {
            console.error('Failed to run image inference:', error);
            this.showError('Failed to run image inference');
        }
    }

    async runDataTest() {
        if (!this.currentModel) {
            this.showError('Please select a model first');
            return;
        }

        const fileInput = document.getElementById('data-input');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showError('Please select a data file');
            return;
        }

        try {
            this.showLoading('Running data test...');
            
            const formData = new FormData();
            formData.append('test_data', file);
            
            const response = await api.testModel(formData);
            this.displayResults(response);
            
        } catch (error) {
            console.error('Failed to run data test:', error);
            this.showError('Failed to run data test');
        }
    }

    async runAudioInference() {
        if (!this.currentModel) {
            this.showError('Please select a model first');
            return;
        }

        const fileInput = document.getElementById('audio-input');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showError('Please select an audio file');
            return;
        }

        try {
            this.showLoading('Running audio inference...');
            
            // Simulate audio inference
            const results = await this.simulateInference('audio', file);
            
        } catch (error) {
            console.error('Failed to run audio inference:', error);
            this.showError('Failed to run audio inference');
        }
    }

    async simulateInference(type, file) {
        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Generate mock results
        const results = {
            type: type,
            filename: file.name,
            model: this.currentModel,
            predictions: this.generateMockPredictions(type),
            confidence: (Math.random() * 0.3 + 0.7).toFixed(3),
            processing_time: (Math.random() * 0.5 + 0.1).toFixed(3),
            timestamp: new Date().toISOString()
        };
        
        this.displayResults(results);
    }

    generateMockPredictions(type) {
        if (type === 'image') {
            // Mock image classification results
            const classes = ['cat', 'dog', 'bird', 'car', 'person', 'building'];
            return classes.map(cls => ({
                class: cls,
                confidence: (Math.random() * 0.4 + 0.1).toFixed(3)
            })).sort((a, b) => b.confidence - a.confidence).slice(0, 3);
        } else if (type === 'audio') {
            // Mock audio classification results
            const classes = ['speech', 'music', 'noise', 'silence', 'conversation'];
            return classes.map(cls => ({
                class: cls,
                confidence: (Math.random() * 0.4 + 0.1).toFixed(3)
            })).sort((a, b) => b.confidence - a.confidence).slice(0, 3);
        } else {
            // Mock regression results
            return Array.from({length: 5}, (_, i) => ({
                sample: i + 1,
                prediction: (Math.random() * 10).toFixed(2)
            }));
        }
    }

    displayResults(results) {
        const container = document.getElementById('results-container');
        
        if (results.type === 'image') {
            container.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Image Classification Results</h6>
                        <div class="mb-3">
                            <strong>Model:</strong> ${results.model}<br>
                            <strong>File:</strong> ${results.filename}<br>
                            <strong>Confidence:</strong> ${(results.confidence * 100).toFixed(1)}%<br>
                            <strong>Processing Time:</strong> ${results.processing_time}s
                        </div>
                        <h6>Top Predictions:</h6>
                        ${results.predictions.map(pred => `
                            <div class="d-flex justify-content-between mb-1">
                                <span>${pred.class}</span>
                                <span class="badge bg-primary">${(pred.confidence * 100).toFixed(1)}%</span>
                            </div>
                        `).join('')}
                    </div>
                    <div class="col-md-6">
                        <div class="text-center">
                            <i class="fas fa-check-circle fa-3x text-success mb-2"></i>
                            <p class="text-success">Inference completed successfully!</p>
                        </div>
                    </div>
                </div>
            `;
        } else if (results.type === 'audio') {
            container.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Audio Classification Results</h6>
                        <div class="mb-3">
                            <strong>Model:</strong> ${results.model}<br>
                            <strong>File:</strong> ${results.filename}<br>
                            <strong>Confidence:</strong> ${(results.confidence * 100).toFixed(1)}%<br>
                            <strong>Processing Time:</strong> ${results.processing_time}s
                        </div>
                        <h6>Top Predictions:</h6>
                        ${results.predictions.map(pred => `
                            <div class="d-flex justify-content-between mb-1">
                                <span>${pred.class}</span>
                                <span class="badge bg-success">${(pred.confidence * 100).toFixed(1)}%</span>
                            </div>
                        `).join('')}
                    </div>
                    <div class="col-md-6">
                        <div class="text-center">
                            <i class="fas fa-check-circle fa-3x text-success mb-2"></i>
                            <p class="text-success">Inference completed successfully!</p>
                        </div>
                    </div>
                </div>
            `;
        } else {
            // Data test results or API results
            if (results.data_shape) {
                // API test results
                container.innerHTML = `
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Model Test Results</h6>
                            <div class="mb-3">
                                <strong>Data Shape:</strong> ${results.data_shape}<br>
                                <strong>Accuracy:</strong> ${(results.accuracy * 100).toFixed(1)}%<br>
                                <strong>Loss:</strong> ${results.loss}<br>
                                <strong>Test Samples:</strong> ${results.test_samples}
                            </div>
                            <h6>Sample Predictions:</h6>
                            ${results.predictions.map(pred => `
                                <div class="d-flex justify-content-between mb-1">
                                    <span>Sample ${pred.sample || 'N/A'}</span>
                                    <span class="badge bg-info">${pred.prediction || pred}</span>
                                </div>
                            `).join('')}
                        </div>
                        <div class="col-md-6">
                            <h6>Performance Metrics</h6>
                            <div class="mb-3">
                                <strong>Inference Time:</strong> ${results.model_performance.inference_time}s<br>
                                <strong>Memory Usage:</strong> ${results.model_performance.memory_usage}
                            </div>
                            <div class="text-center">
                                <i class="fas fa-chart-line fa-2x text-info mb-2"></i>
                                <p class="text-info">Test completed successfully!</p>
                            </div>
                        </div>
                    </div>
                `;
            } else {
                // Simulated data test results
                container.innerHTML = `
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Data Test Results</h6>
                            <div class="mb-3">
                                <strong>Model:</strong> ${results.model}<br>
                                <strong>File:</strong> ${results.filename}<br>
                                <strong>Processing Time:</strong> ${results.processing_time}s
                            </div>
                            <h6>Test Results:</h6>
                            ${results.predictions.map(pred => `
                                <div class="d-flex justify-content-between mb-1">
                                    <span>Sample ${pred.sample}</span>
                                    <span class="badge bg-warning">${pred.prediction}</span>
                                </div>
                            `).join('')}
                        </div>
                        <div class="col-md-6">
                            <div class="text-center">
                                <i class="fas fa-check-circle fa-3x text-success mb-2"></i>
                                <p class="text-success">Test completed successfully!</p>
                            </div>
                        </div>
                    </div>
                `;
            }
        }
    }

    showLoading(message) {
        const container = document.getElementById('results-container');
        container.innerHTML = `
            <div class="text-center">
                <i class="fas fa-spinner fa-spin fa-3x text-primary mb-3"></i>
                <h5>${message}</h5>
                <p class="text-muted">Please wait...</p>
            </div>
        `;
    }

    showSuccess(message) {
        // Create a temporary success message
        const alert = document.createElement('div');
        alert.className = 'alert alert-success alert-dismissible fade show position-fixed';
        alert.style.cssText = 'top: 20px; right: 20px; z-index: 9999;';
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(alert);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 3000);
    }

    showError(message) {
        // Create a temporary error message
        const alert = document.createElement('div');
        alert.className = 'alert alert-danger alert-dismissible fade show position-fixed';
        alert.style.cssText = 'top: 20px; right: 20px; z-index: 9999;';
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Global inference manager instance
const inference = new InferenceManager(); 