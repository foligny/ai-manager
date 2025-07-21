"""
Simple integration tests for project model management functionality.
"""

import pytest
import json
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db, User, Project, ProjectModel, Base, engine


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def test_db():
    """Create test database."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    yield
    # Clean up
    Base.metadata.drop_all(bind=engine)


def test_project_edit_modal_present(client):
    """Test that project edit modal is present in dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "projectModal" in response.text
    assert "Edit Project" in response.text
    assert "Model Management" in response.text
    assert "Add Model from Available Models" in response.text
    assert "Import from Hugging Face" in response.text


def test_dashboard_edit_buttons_present(client):
    """Test that edit buttons are present in project list."""
    response = client.get("/")
    assert response.status_code == 200
    assert "fas fa-edit" in response.text


def test_api_endpoints_exist(client):
    """Test that the API endpoints exist and return proper responses."""
    # Test projects endpoint
    response = client.get("/projects/")
    # Should return 401 (unauthorized) not 404 (not found)
    assert response.status_code == 401
    
    # Test models endpoint
    response = client.get("/models/list")
    # Should return 401 (unauthorized) not 404 (not found)
    assert response.status_code == 401


def test_huggingface_import_endpoint_exists(client):
    """Test that the Hugging Face import endpoint exists."""
    response = client.post("/models/import-huggingface", json={})
    # Should return 401 (unauthorized) not 404 (not found)
    assert response.status_code == 401


def test_project_model_endpoints_exist(client):
    """Test that project model management endpoints exist."""
    # Test get project models
    response = client.get("/projects/1/models")
    # Should return 401 (unauthorized) not 404 (not found)
    assert response.status_code == 401
    
    # Test assign model to project
    response = client.post("/projects/1/models", json={})
    # Should return 401 (unauthorized) not 404 (not found)
    assert response.status_code == 401
    
    # Test remove model from project
    response = client.delete("/projects/1/models/test_model.pth")
    # Should return 401 (unauthorized) not 404 (not found)
    assert response.status_code == 401


def test_frontend_javascript_functions_exist(client):
    """Test that the frontend JavaScript functions are loaded."""
    response = client.get("/static/js/dashboard.js")
    assert response.status_code == 200
    
    js_content = response.text
    assert "editProject" in js_content
    assert "saveProject" in js_content
    assert "addModelToProject" in js_content
    assert "removeModelFromProject" in js_content
    assert "importFromHuggingFace" in js_content


def test_database_models_exist():
    """Test that the database models are properly defined."""
    # Test that ProjectModel exists and has required fields
    assert hasattr(ProjectModel, '__tablename__')
    assert ProjectModel.__tablename__ == "project_models"
    
    # Test that Project has the assigned_models relationship
    assert hasattr(Project, 'assigned_models')


def test_api_schemas_exist():
    """Test that the API schemas are properly defined."""
    from app.schemas.project import ProjectUpdate
    
    # Test that ProjectUpdate schema exists and has required fields
    assert hasattr(ProjectUpdate, '__fields__')
    assert 'name' in ProjectUpdate.__fields__
    assert 'description' in ProjectUpdate.__fields__
    assert 'tags' in ProjectUpdate.__fields__


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 