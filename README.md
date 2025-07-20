# AI Manager - Training Monitoring Platform

A comprehensive AI training monitoring platform similar to Weights & Biases, built with FastAPI, Dash, and modern web technologies.

## Features

### 🎯 Core Features
- **Real-time Training Monitoring**: Live tracking of training metrics with WebSocket connections
- **Interactive Dashboards**: Beautiful, responsive dashboards with Plotly charts
- **Experiment Tracking**: Compare multiple training runs with detailed metrics
- **Model Versioning**: Track model versions, configurations, and artifacts
- **Configuration Management**: Save and version training configurations
- **Real-time Logging**: Stream training logs with search and filtering

### 📊 Visualization Features
- **Training Curves**: Loss, accuracy, and custom metrics over time
- **Confusion Matrices**: For classification tasks
- **Parameter Distributions**: Histograms and box plots
- **System Metrics**: GPU usage, memory consumption, CPU utilization
- **Custom Plots**: Support for any custom visualization

### 🔧 Technical Features
- **RESTful API**: Complete API for integration with any ML framework
- **WebSocket Support**: Real-time updates during training
- **Database Backend**: SQLAlchemy with PostgreSQL support
- **Authentication**: JWT-based authentication system
- **File Storage**: Artifact and model file management
- **Background Tasks**: Celery for async processing

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-manager

# Create virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate\
```
or 
```
source ~/venv/bin/activate
```

# Install dependencies
`pip install -r requirements.txt`

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
alembic upgrade head

# Run the application

## Option 1: HTTP (Development)
```bash
python -m uvicorn app.main:app --reload
```

## Option 2: HTTPS (Development with SSL)
```bash
# First, set up SSL certificates (one-time setup)
python setup_https.py

# Then run with HTTPS
python run_https.py
```

## Option 3: Using Makefile (Recommended)
```bash
# Install dependencies and initialize database
make setup

# Start development server (HTTP)
make dev

# Or run with Docker
make docker-run
```

## Option 4: Manual HTTPS Command
```bash
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
```

### Usage

1. **Start the application** (choose one option):

   **Option A: HTTP (Simple)**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

   **Option B: HTTPS (Secure)**
   ```bash
   # One-time setup
   python setup_https.py
   
   # Run with HTTPS
   python run_https.py
   ```

   **Option C: Using Makefile (Recommended)**
   ```bash
   make setup    # Install and initialize
   make dev      # Start development server
   ```

2. **Access the dashboard**:
   - **HTTP**: http://localhost:8000
   - **HTTPS**: https://localhost:8000 (may show security warning for self-signed cert)
   - **API Documentation**: http://localhost:8000/docs or https://localhost:8000/docs

3. **Integrate with your training**:
   ```python
   from ai_manager import AIManager
   
   # Initialize the manager
   manager = AIManager(project_name="my_project")
   
   # Start a new run
   with manager.run() as run:
       # Log metrics
       run.log({"loss": 0.5, "accuracy": 0.85})
       
       # Log parameters
       run.config.update({"learning_rate": 0.001, "batch_size": 32})
   ```

## Project Structure

```
ai-manager/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py              # Configuration settings
│   ├── database.py            # Database models and connection
│   ├── api/                   # API routes
│   ├── core/                  # Core functionality
│   ├── models/                # Database models
│   ├── schemas/               # Pydantic schemas
│   ├── services/              # Business logic
│   └── utils/                 # Utility functions
├── frontend/                  # Dashboard frontend
├── tests/                     # Test suite
├── alembic/                   # Database migrations
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## API Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/register` - User registration
- `POST /auth/refresh` - Refresh JWT token

### Projects
- `GET /projects` - List projects
- `POST /projects` - Create project
- `GET /projects/{project_id}` - Get project details
- `PUT /projects/{project_id}` - Update project
- `DELETE /projects/{project_id}` - Delete project

### Runs
- `GET /projects/{project_id}/runs` - List runs
- `POST /projects/{project_id}/runs` - Create run
- `GET /runs/{run_id}` - Get run details
- `PUT /runs/{run_id}` - Update run
- `DELETE /runs/{run_id}` - Delete run

### Metrics
- `POST /runs/{run_id}/metrics` - Log metrics
- `GET /runs/{run_id}/metrics` - Get run metrics
- `GET /runs/{run_id}/metrics/history` - Get metric history

### Artifacts
- `POST /runs/{run_id}/artifacts` - Upload artifact
- `GET /runs/{run_id}/artifacts` - List artifacts
- `GET /artifacts/{artifact_id}` - Download artifact

## Configuration

Create a `.env` file with the following variables:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost/ai_manager

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis (for background tasks)
REDIS_URL=redis://localhost:6379

# File Storage
STORAGE_PATH=./storage

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

## Development

### Available Scripts and Tools

**Launch Scripts:**
- `run_https.py` - Launch server with HTTPS (requires SSL setup)
- `setup_https.py` - Generate SSL certificates for HTTPS development

**Makefile Commands:**
```bash
make help      # Show all available commands
make setup     # Install dependencies and initialize database
make dev       # Start development server (HTTP)
make test      # Run tests
make lint      # Run linting
make format    # Format code
make docker-run # Run with Docker Compose
make clean     # Clean up generated files
```

**Performance Analysis:**
- `performance_analysis.py` - Analyze training performance
- `quick_performance_demo.py` - Quick demo of performance features
- `targeted_cleanup.py` - Clean up specific training runs

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black app/
isort app/
```

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head
```

## Training Performance Monitoring

### 🎯 Viewing Training Performance

#### **1. Real-time Training Dashboard**

**Start the AI Manager server:**
```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Access the web dashboard:**
- **URL:** http://localhost:8000
- **Login:** Username: `admin`, Password: `admin123`

#### **2. Viewing Training Curves**

**In the web dashboard:**
1. **Navigate to Projects** → Select your project
2. **Click on a Run** → View detailed metrics
3. **Real-time Charts** → See training curves updating live
4. **Metric Comparison** → Compare multiple runs side-by-side

**Key metrics to monitor:**
- **Training Loss** → Should decrease over time
- **Validation Loss** → Should decrease (watch for overfitting)
- **Training Accuracy** → Should increase
- **Validation Accuracy** → Should increase (indicates generalization)

#### **3. API-based Performance Analysis**

**Using the performance analysis script:**
```bash
# Run the analysis tool
python performance_analysis.py

# Options available:
# 1. Show available runs
# 2. Analyze specific run
# 3. Compare multiple runs
# 4. Plot training curves
```

**Quick performance demo:**
```bash
python quick_performance_demo.py
```

### 🔄 Resetting and Managing Training

#### **1. Reset Training Runs**

**Delete specific runs:**
```bash
# Using the cleanup script
python targeted_cleanup.py

# Options:
# - Delete all test/debug runs
# - Delete specific projects
# - Keep productive runs
```

**Manual deletion via API:**
```bash
# Get authentication token
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"

# Delete a run (replace {run_id} with actual ID)
curl -X DELETE "http://localhost:8000/runs/{run_id}" \
  -H "Authorization: Bearer {your_token}"

# Delete a project (deletes all associated runs)
curl -X DELETE "http://localhost:8000/projects/{project_id}" \
  -H "Authorization: Bearer {your_token}"
```

#### **2. Start Fresh Training**

**Clear all data and start over:**
```bash
# Stop the server (Ctrl+C)
# Delete the database
rm ai_manager.db

# Restart the server (creates fresh database)
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Reset specific project:**
1. Go to web dashboard
2. Navigate to Projects
3. Select project to reset
4. Delete all runs in the project
5. Start new training runs

### 📊 Understanding Training Curves

#### **1. Healthy Training Patterns**

**Good training curves show:**
- **Loss decreasing** → Model is learning
- **Accuracy increasing** → Performance improving
- **Validation metrics following training** → Good generalization
- **Smooth curves** → Stable learning

**Warning signs:**
- **Validation loss increasing** → Overfitting
- **Training loss not decreasing** → Learning rate too low
- **Gap between train/validation** → Overfitting
- **Oscillating curves** → Learning rate too high

#### **2. Comparing Different Runs**

**In the web dashboard:**
1. **Select multiple runs** → Compare configurations
2. **Overlay charts** → See performance differences
3. **Parameter comparison** → Understand what works best
4. **Export results** → Save for external analysis

**Key comparisons:**
- **Learning rate effects** → Higher vs lower rates
- **Batch size impact** → Memory vs convergence speed
- **Architecture changes** → Model performance differences
- **Data augmentation** → Generalization improvements

### 🚀 Best Practices

#### **1. Training Setup**
```python
# Example training integration
from ai_manager import AIManager

# Initialize with project
manager = AIManager(project_name="image_classifier")

# Start training run
with manager.run(name="experiment_1") as run:
    # Log hyperparameters
    run.config.update({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    })
    
    # Training loop
    for epoch in range(100):
        # Your training code here
        train_loss = train_epoch()
        val_loss = validate()
        
        # Log metrics
        run.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })
```

#### **2. Monitoring Checklist**
- [ ] **Check training curves** every few epochs
- [ ] **Monitor validation metrics** for overfitting
- [ ] **Compare with baseline** runs
- [ ] **Save best models** as artifacts
- [ ] **Document hyperparameters** that work well
- [ ] **Export results** for reporting

#### **3. Troubleshooting Common Issues**

**Training not improving:**
- Increase learning rate
- Check data quality
- Verify model architecture
- Monitor gradient flow

**Overfitting:**
- Add regularization
- Reduce model complexity
- Use data augmentation
- Early stopping

**Unstable training:**
- Reduce learning rate
- Increase batch size
- Check data normalization
- Monitor gradient clipping

### 📈 Advanced Features

#### **1. Custom Metrics**
```python
# Log custom metrics
run.log({
    "custom_metric": value,
    "f1_score": f1_score,
    "precision": precision,
    "recall": recall
})
```

#### **2. Model Artifacts**
```python
# Save trained model
run.save_artifact("model.pth", model_state_dict)

# Save training plots
run.save_artifact("training_curves.png", plot_data)
```

#### **3. Experiment Tracking**
```python
# Tag runs for organization
run.tags = ["baseline", "high_lr", "experiment"]

# Add descriptions
run.description = "Testing higher learning rate for faster convergence"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the GNU General Public License v2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Weights & Biases (wandb.ai)
- Built with FastAPI, Dash, and Plotly
- Uses modern web technologies for real-time monitoring