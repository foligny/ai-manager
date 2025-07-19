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
python -m uvicorn app.main:app --reload
```

### Usage

1. **Start the application**:
   ```bash
   python -m uvicorn app.main:app --reload
   ```

2. **Access the dashboard**:
   - Web Dashboard: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

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