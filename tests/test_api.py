"""
API tests for AI Manager.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import Base, get_db
from app.core.auth import get_password_hash
from app.database import User


# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Set up test database."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_user():
    """Create a test user."""
    db = TestingSessionLocal()
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=get_password_hash("testpassword")
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    return user


@pytest.fixture
def auth_headers(test_user):
    """Get authentication headers."""
    response = client.post("/auth/login", data={
        "username": "testuser",
        "password": "testpassword"
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestAuth:
    """Test authentication endpoints."""
    
    def test_register(self):
        """Test user registration."""
        response = client.post("/auth/register", json={
            "email": "new@example.com",
            "username": "newuser",
            "password": "newpassword"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "new@example.com"
        assert data["username"] == "newuser"
    
    def test_login(self, test_user):
        """Test user login."""
        response = client.post("/auth/login", data={
            "username": "testuser",
            "password": "testpassword"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials."""
        response = client.post("/auth/login", data={
            "username": "wronguser",
            "password": "wrongpassword"
        })
        assert response.status_code == 401
    
    def test_me(self, auth_headers):
        """Test getting current user."""
        response = client.get("/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"


class TestProjects:
    """Test project endpoints."""
    
    def test_create_project(self, auth_headers):
        """Test creating a project."""
        response = client.post("/projects/", json={
            "name": "Test Project",
            "description": "A test project",
            "is_public": False
        }, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Project"
        assert data["description"] == "A test project"
    
    def test_list_projects(self, auth_headers):
        """Test listing projects."""
        # Create a project first
        client.post("/projects/", json={
            "name": "Test Project",
            "description": "A test project"
        }, headers=auth_headers)
        
        response = client.get("/projects/", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "Test Project"
    
    def test_get_project(self, auth_headers):
        """Test getting a specific project."""
        # Create a project first
        create_response = client.post("/projects/", json={
            "name": "Test Project",
            "description": "A test project"
        }, headers=auth_headers)
        project_id = create_response.json()["id"]
        
        response = client.get(f"/projects/{project_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Project"
    
    def test_update_project(self, auth_headers):
        """Test updating a project."""
        # Create a project first
        create_response = client.post("/projects/", json={
            "name": "Test Project",
            "description": "A test project"
        }, headers=auth_headers)
        project_id = create_response.json()["id"]
        
        response = client.put(f"/projects/{project_id}", json={
            "name": "Updated Project",
            "description": "Updated description"
        }, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Project"
        assert data["description"] == "Updated description"
    
    def test_delete_project(self, auth_headers):
        """Test deleting a project."""
        # Create a project first
        create_response = client.post("/projects/", json={
            "name": "Test Project",
            "description": "A test project"
        }, headers=auth_headers)
        project_id = create_response.json()["id"]
        
        response = client.delete(f"/projects/{project_id}", headers=auth_headers)
        assert response.status_code == 200
        
        # Verify project is deleted
        get_response = client.get(f"/projects/{project_id}", headers=auth_headers)
        assert get_response.status_code == 404


class TestRuns:
    """Test run endpoints."""
    
    def test_create_run(self, auth_headers):
        """Test creating a run."""
        # Create a project first
        project_response = client.post("/projects/", json={
            "name": "Test Project",
            "description": "A test project"
        }, headers=auth_headers)
        project_id = project_response.json()["id"]
        
        response = client.post("/runs/", json={
            "name": "Test Run",
            "config": {"learning_rate": 0.001},
            "tags": ["test", "demo"]
        }, params={"project_id": project_id}, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Run"
        assert data["config"]["learning_rate"] == 0.001
    
    def test_list_runs(self, auth_headers):
        """Test listing runs."""
        # Create a project and run first
        project_response = client.post("/projects/", json={
            "name": "Test Project",
            "description": "A test project"
        }, headers=auth_headers)
        project_id = project_response.json()["id"]
        
        client.post("/runs/", json={
            "name": "Test Run",
            "config": {"learning_rate": 0.001}
        }, params={"project_id": project_id}, headers=auth_headers)
        
        response = client.get("/runs/", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "Test Run"
    
    def test_get_run(self, auth_headers):
        """Test getting a specific run."""
        # Create a project and run first
        project_response = client.post("/projects/", json={
            "name": "Test Project",
            "description": "A test project"
        }, headers=auth_headers)
        project_id = project_response.json()["id"]
        
        run_response = client.post("/runs/", json={
            "name": "Test Run",
            "config": {"learning_rate": 0.001}
        }, params={"project_id": project_id}, headers=auth_headers)
        run_id = run_response.json()["id"]
        
        response = client.get(f"/runs/{run_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Run"


class TestMetrics:
    """Test metrics endpoints."""
    
    def test_log_metric(self, auth_headers):
        """Test logging a metric."""
        # Create a project and run first
        project_response = client.post("/projects/", json={
            "name": "Test Project",
            "description": "A test project"
        }, headers=auth_headers)
        project_id = project_response.json()["id"]
        
        run_response = client.post("/runs/", json={
            "name": "Test Run",
            "config": {"learning_rate": 0.001}
        }, params={"project_id": project_id}, headers=auth_headers)
        run_id = run_response.json()["id"]
        
        response = client.post(f"/metrics/{run_id}", json={
            "name": "loss",
            "value": 0.5,
            "step": 1
        }, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "loss"
        assert data["value"] == 0.5
    
    def test_log_metrics_batch(self, auth_headers):
        """Test logging multiple metrics."""
        # Create a project and run first
        project_response = client.post("/projects/", json={
            "name": "Test Project",
            "description": "A test project"
        }, headers=auth_headers)
        project_id = project_response.json()["id"]
        
        run_response = client.post("/runs/", json={
            "name": "Test Run",
            "config": {"learning_rate": 0.001}
        }, params={"project_id": project_id}, headers=auth_headers)
        run_id = run_response.json()["id"]
        
        response = client.post(f"/metrics/{run_id}/batch", json={
            "metrics": {
                "loss": 0.5,
                "accuracy": 0.85
            },
            "step": 1
        }, headers=auth_headers)
        assert response.status_code == 200
    
    def test_get_run_metrics(self, auth_headers):
        """Test getting metrics for a run."""
        # Create a project and run first
        project_response = client.post("/projects/", json={
            "name": "Test Project",
            "description": "A test project"
        }, headers=auth_headers)
        project_id = project_response.json()["id"]
        
        run_response = client.post("/runs/", json={
            "name": "Test Run",
            "config": {"learning_rate": 0.001}
        }, params={"project_id": project_id}, headers=auth_headers)
        run_id = run_response.json()["id"]
        
        # Log some metrics
        client.post(f"/metrics/{run_id}", json={
            "name": "loss",
            "value": 0.5,
            "step": 1
        }, headers=auth_headers)
        
        response = client.get(f"/metrics/{run_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "loss" 