# AI Manager API Documentation

## Overview

AI Manager provides a comprehensive REST API for tracking machine learning experiments, logging metrics, and managing training runs. The API is built with FastAPI and provides automatic OpenAPI documentation.

## Base URL

```
http://localhost:8000
```

## Authentication

The API uses JWT (JSON Web Tokens) for authentication. Most endpoints require authentication.

### Getting a Token

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your_username&password=your_password"
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

### Using the Token

Include the token in the Authorization header:
```
Authorization: Bearer <your_token>
```

## Endpoints

### Authentication

#### POST /auth/register
Register a new user.

**Request Body:**
```json
{
  "email": "user@example.com",
  "username": "username",
  "password": "password"
}
```

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "username",
  "is_active": true,
  "is_superuser": false,
  "created_at": "2023-01-01T00:00:00",
  "updated_at": "2023-01-01T00:00:00"
}
```

#### POST /auth/login
Login and get access token.

**Request Body:**
```
username=your_username&password=your_password
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

#### GET /auth/me
Get current user information.

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "username",
  "is_active": true,
  "is_superuser": false,
  "created_at": "2023-01-01T00:00:00",
  "updated_at": "2023-01-01T00:00:00"
}
```

### Projects

#### GET /projects/
List all projects for the current user.

**Query Parameters:**
- `skip` (int): Number of projects to skip (default: 0)
- `limit` (int): Maximum number of projects to return (default: 100)

**Response:**
```json
[
  {
    "id": 1,
    "name": "My Project",
    "description": "A machine learning project",
    "owner_id": 1,
    "is_public": false,
    "created_at": "2023-01-01T00:00:00",
    "updated_at": "2023-01-01T00:00:00"
  }
]
```

#### POST /projects/
Create a new project.

**Request Body:**
```json
{
  "name": "My Project",
  "description": "A machine learning project",
  "is_public": false
}
```

**Response:**
```json
{
  "id": 1,
  "name": "My Project",
  "description": "A machine learning project",
  "owner_id": 1,
  "is_public": false,
  "created_at": "2023-01-01T00:00:00",
  "updated_at": "2023-01-01T00:00:00"
}
```

#### GET /projects/{project_id}
Get a specific project.

**Response:**
```json
{
  "id": 1,
  "name": "My Project",
  "description": "A machine learning project",
  "owner_id": 1,
  "is_public": false,
  "created_at": "2023-01-01T00:00:00",
  "updated_at": "2023-01-01T00:00:00"
}
```

#### PUT /projects/{project_id}
Update a project.

**Request Body:**
```json
{
  "name": "Updated Project Name",
  "description": "Updated description"
}
```

#### DELETE /projects/{project_id}
Delete a project.

### Runs

#### GET /runs/
List all runs for the current user.

**Query Parameters:**
- `project_id` (int): Filter by project ID
- `skip` (int): Number of runs to skip (default: 0)
- `limit` (int): Maximum number of runs to return (default: 100)

**Response:**
```json
[
  {
    "id": 1,
    "name": "Training Run 1",
    "status": "running",
    "started_at": "2023-01-01T00:00:00",
    "ended_at": null,
    "tags": ["demo", "test"]
  }
]
```

#### POST /runs/
Create a new run.

**Query Parameters:**
- `project_id` (int): ID of the project

**Request Body:**
```json
{
  "name": "Training Run 1",
  "config": {
    "learning_rate": 0.001,
    "batch_size": 32
  },
  "tags": ["demo", "test"]
}
```

**Response:**
```json
{
  "id": 1,
  "name": "Training Run 1",
  "project_id": 1,
  "status": "running",
  "config": {
    "learning_rate": 0.001,
    "batch_size": 32
  },
  "tags": ["demo", "test"],
  "started_at": "2023-01-01T00:00:00",
  "ended_at": null,
  "created_at": "2023-01-01T00:00:00",
  "updated_at": "2023-01-01T00:00:00"
}
```

#### GET /runs/{run_id}
Get a specific run.

#### PUT /runs/{run_id}
Update a run.

**Request Body:**
```json
{
  "status": "completed",
  "ended_at": "2023-01-01T01:00:00"
}
```

#### DELETE /runs/{run_id}
Delete a run.

### Metrics

#### POST /metrics/{run_id}
Log a single metric.

**Request Body:**
```json
{
  "name": "loss",
  "value": 0.5,
  "step": 1
}
```

**Response:**
```json
{
  "id": 1,
  "run_id": 1,
  "name": "loss",
  "value": 0.5,
  "step": 1,
  "timestamp": "2023-01-01T00:00:00"
}
```

#### POST /metrics/{run_id}/batch
Log multiple metrics at once.

**Request Body:**
```json
{
  "metrics": {
    "loss": 0.5,
    "accuracy": 0.85,
    "learning_rate": 0.001
  },
  "step": 1
}
```

**Response:**
```json
{
  "message": "Logged 3 metrics successfully"
}
```

#### GET /metrics/{run_id}
Get metrics for a run.

**Query Parameters:**
- `metric_name` (str): Filter by metric name
- `skip` (int): Number of metrics to skip (default: 0)
- `limit` (int): Maximum number of metrics to return (default: 1000)

**Response:**
```json
[
  {
    "id": 1,
    "run_id": 1,
    "name": "loss",
    "value": 0.5,
    "step": 1,
    "timestamp": "2023-01-01T00:00:00"
  }
]
```

#### GET /metrics/{run_id}/history/{metric_name}
Get metric history for a specific metric.

**Response:**
```json
{
  "name": "loss",
  "values": [0.5, 0.4, 0.3],
  "steps": [1, 2, 3],
  "timestamps": [
    "2023-01-01T00:00:00",
    "2023-01-01T00:01:00",
    "2023-01-01T00:02:00"
  ]
}
```

#### GET /metrics/{run_id}/summary
Get summary statistics for all metrics in a run.

**Response:**
```json
[
  {
    "name": "loss",
    "current_value": 0.3,
    "min_value": 0.1,
    "max_value": 0.8,
    "mean_value": 0.4,
    "total_points": 100
  }
]
```

## WebSocket API

### WebSocket Connection

Connect to the WebSocket endpoint for real-time metric updates:

```
ws://localhost:8000/ws/{run_id}
```

**Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/1');

ws.onopen = function(event) {
    console.log('Connected to WebSocket');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received metric update:', data);
};

ws.onclose = function(event) {
    console.log('WebSocket connection closed');
};
```

## Error Responses

The API returns standard HTTP status codes and error messages:

### 400 Bad Request
```json
{
  "detail": "Validation error"
}
```

### 401 Unauthorized
```json
{
  "detail": "Could not validate credentials"
}
```

### 403 Forbidden
```json
{
  "detail": "Not enough permissions"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse. Limits are:
- 100 requests per minute per user
- 1000 requests per hour per user

## Pagination

List endpoints support pagination using `skip` and `limit` parameters:

```
GET /projects/?skip=0&limit=10
```

## Filtering

Some endpoints support filtering:

```
GET /runs/?project_id=1
GET /metrics/1/?metric_name=loss
```

## Sorting

Metrics are automatically sorted by timestamp in ascending order.

## Examples

### Complete Training Workflow

1. **Create a project:**
```bash
curl -X POST "http://localhost:8000/projects/" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "My ML Project", "description": "Image classification"}'
```

2. **Start a training run:**
```bash
curl -X POST "http://localhost:8000/runs/?project_id=1" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "ResNet Training", "config": {"lr": 0.001}}'
```

3. **Log metrics during training:**
```bash
curl -X POST "http://localhost:8000/metrics/1/batch" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"metrics": {"loss": 0.5, "accuracy": 0.85}, "step": 1}'
```

4. **Complete the run:**
```bash
curl -X PUT "http://localhost:8000/runs/1" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"status": "completed"}'
```

### Python Client Example

```python
from ai_manager import AIManager

# Initialize client
manager = AIManager(
    project_name="my_project",
    api_url="http://localhost:8000",
    username="user",
    password="pass"
)

# Start training run
with manager.run(name="training_run") as run:
    for epoch in range(100):
        # Log metrics
        run.log({
            "loss": 0.5,
            "accuracy": 0.85
        }, step=epoch)
```

## SDKs and Libraries

- **Python**: Built-in client library (`ai_manager`)
- **JavaScript**: Available via npm (`ai-manager-js`)
- **R**: Available via CRAN (`aiManager`)

## Support

For API support and questions:
- Email: support@ai-manager.com
- Documentation: https://docs.ai-manager.com
- GitHub Issues: https://github.com/ai-manager/ai-manager/issues 