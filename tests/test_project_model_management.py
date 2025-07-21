"""
Unit tests for project model management functionality.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.database import get_db, User, Project, ProjectModel
from app.api.auth import create_access_token


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def test_user():
    """Create a test user."""
    return {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "is_active": True
    }


@pytest.fixture
def test_project():
    """Create a test project."""
    return {
        "id": 1,
        "name": "Test Project",
        "description": "A test project",
        "owner_id": 1,
        "is_public": False,
        "tags": ["test", "demo"]
    }


@pytest.fixture
def auth_headers(test_user):
    """Create authentication headers."""
    token = create_access_token(data={"sub": test_user["username"]})
    return {"Authorization": f"Bearer {token}"}


class TestProjectModelManagement:
    """Test cases for project model management."""

    def test_update_project_success(self, client, auth_headers, test_project):
        """Test successful project update."""
        # Mock database session
        with patch('app.api.projects.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            
            # Mock project query
            mock_project = MagicMock()
            mock_project.id = test_project["id"]
            mock_project.owner_id = test_project["owner_id"]
            mock_project.name = test_project["name"]
            mock_project.description = test_project["description"]
            mock_project.tags = test_project["tags"]
            
            mock_db.query.return_value.filter.return_value.first.return_value = mock_project
            
            # Test data
            update_data = {
                "name": "Updated Project Name",
                "description": "Updated description",
                "tags": ["updated", "test", "new"]
            }
            
            # Make request
            response = client.put(
                f"/projects/{test_project['id']}",
                headers=auth_headers,
                json=update_data
            )
            
            # Assertions
            assert response.status_code == 200
            result = response.json()
            assert result["name"] == update_data["name"]
            assert result["description"] == update_data["description"]
            assert result["tags"] == update_data["tags"]
            
            # Verify database was called
            mock_db.commit.assert_called_once()
            mock_db.refresh.assert_called_once()

    def test_update_project_not_found(self, client, auth_headers):
        """Test project update when project doesn't exist."""
        with patch('app.api.projects.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            response = client.put(
                "/projects/999",
                headers=auth_headers,
                json={"name": "Test"}
            )
            
            assert response.status_code == 404
            assert "Project not found" in response.json()["detail"]

    def test_get_project_models_success(self, client, auth_headers, test_project):
        """Test getting models assigned to a project."""
        with patch('app.api.projects.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            
            # Mock project
            mock_project = MagicMock()
            mock_project.id = test_project["id"]
            mock_project.owner_id = test_project["owner_id"]
            mock_db.query.return_value.filter.return_value.first.return_value = mock_project
            
            # Mock assigned models
            mock_models = [
                MagicMock(
                    id=1,
                    model_name="test_model.pth",
                    model_path="/models/test_model.pth",
                    model_type="text",
                    model_capabilities=["text_classification"]
                ),
                MagicMock(
                    id=2,
                    model_name="image_model.pth",
                    model_path="/models/image_model.pth",
                    model_type="image",
                    model_capabilities=["image_classification"]
                )
            ]
            mock_db.query.return_value.filter.return_value.all.return_value = mock_models
            
            response = client.get(
                f"/projects/{test_project['id']}/models",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            result = response.json()
            assert len(result) == 2
            assert result[0]["model_name"] == "test_model.pth"
            assert result[1]["model_name"] == "image_model.pth"

    def test_assign_model_to_project_success(self, client, auth_headers, test_project):
        """Test successfully assigning a model to a project."""
        with patch('app.api.projects.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            
            # Mock project
            mock_project = MagicMock()
            mock_project.id = test_project["id"]
            mock_project.owner_id = test_project["owner_id"]
            mock_db.query.return_value.filter.return_value.first.return_value = mock_project
            
            # Mock no existing model assignment
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            # Mock new project model
            mock_project_model = MagicMock(
                id=1,
                project_id=test_project["id"],
                model_name="new_model.pth",
                model_path="/models/new_model.pth",
                model_type="text",
                model_capabilities=["text_generation"]
            )
            
            model_data = {
                "model_name": "new_model.pth",
                "model_path": "/models/new_model.pth",
                "model_type": "text",
                "model_capabilities": ["text_generation"]
            }
            
            response = client.post(
                f"/projects/{test_project['id']}/models",
                headers=auth_headers,
                json=model_data
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result["model_name"] == "new_model.pth"
            assert result["model_type"] == "text"
            
            # Verify database operations
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
            mock_db.refresh.assert_called_once()

    def test_assign_model_already_exists(self, client, auth_headers, test_project):
        """Test assigning a model that's already assigned to the project."""
        with patch('app.api.projects.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            
            # Mock project
            mock_project = MagicMock()
            mock_project.id = test_project["id"]
            mock_project.owner_id = test_project["owner_id"]
            mock_db.query.return_value.filter.return_value.first.return_value = mock_project
            
            # Mock existing model assignment
            mock_existing_model = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = mock_existing_model
            
            model_data = {
                "model_name": "existing_model.pth",
                "model_path": "/models/existing_model.pth",
                "model_type": "text"
            }
            
            response = client.post(
                f"/projects/{test_project['id']}/models",
                headers=auth_headers,
                json=model_data
            )
            
            assert response.status_code == 400
            assert "already assigned" in response.json()["detail"]

    def test_remove_model_from_project_success(self, client, auth_headers, test_project):
        """Test successfully removing a model from a project."""
        with patch('app.api.projects.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            
            # Mock project
            mock_project = MagicMock()
            mock_project.id = test_project["id"]
            mock_project.owner_id = test_project["owner_id"]
            mock_db.query.return_value.filter.return_value.first.return_value = mock_project
            
            # Mock project model to be removed
            mock_project_model = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = mock_project_model
            
            response = client.delete(
                f"/projects/{test_project['id']}/models/test_model.pth",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            assert "removed from project successfully" in response.json()["message"]
            
            # Verify database operations
            mock_db.delete.assert_called_once_with(mock_project_model)
            mock_db.commit.assert_called_once()

    def test_remove_model_not_found(self, client, auth_headers, test_project):
        """Test removing a model that doesn't exist in the project."""
        with patch('app.api.projects.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            
            # Mock project
            mock_project = MagicMock()
            mock_project.id = test_project["id"]
            mock_project.owner_id = test_project["owner_id"]
            mock_db.query.return_value.filter.return_value.first.return_value = mock_project
            
            # Mock no project model found
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            response = client.delete(
                f"/projects/{test_project['id']}/models/nonexistent_model.pth",
                headers=auth_headers
            )
            
            assert response.status_code == 404
            assert "not found in project" in response.json()["detail"]

    @patch('app.api.models.AutoModel.from_pretrained')
    @patch('app.api.models.AutoTokenizer.from_pretrained')
    def test_import_huggingface_model_success(self, mock_tokenizer, mock_model, client, auth_headers):
        """Test successful Hugging Face model import."""
        with patch('app.api.models.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            
            # Mock model and tokenizer
            mock_model_instance = MagicMock()
            mock_tokenizer_instance = MagicMock()
            mock_model.return_value = mock_model_instance
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            # Mock file operations
            with patch('os.path.exists', return_value=False), \
                 patch('os.makedirs'), \
                 patch('torch.save'), \
                 patch('app.api.models.model_analyzer') as mock_analyzer:
                
                mock_analyzer.analyze_model.return_value = {
                    "capabilities": ["text_classification"],
                    "type": "text"
                }
                
                import_data = {
                    "model_name": "facebook/wav2vec2-base",
                    "project_id": 1
                }
                
                response = client.post(
                    "/models/import-huggingface",
                    headers=auth_headers,
                    json=import_data
                )
                
                assert response.status_code == 200
                result = response.json()
                assert result["model_name"] == "facebook_wav2vec2-base.pth"
                assert result["status"] == "imported"
                assert "text_classification" in result["capabilities"]

    def test_import_huggingface_model_already_exists(self, client, auth_headers):
        """Test importing a Hugging Face model that already exists."""
        with patch('app.api.models.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            
            # Mock file already exists
            with patch('os.path.exists', return_value=True), \
                 patch('os.path.getsize', return_value=1024):
                
                import_data = {
                    "model_name": "facebook/wav2vec2-base",
                    "project_id": 1
                }
                
                response = client.post(
                    "/models/import-huggingface",
                    headers=auth_headers,
                    json=import_data
                )
                
                assert response.status_code == 200
                result = response.json()
                assert result["status"] == "already_exists"

    def test_import_huggingface_missing_model_name(self, client, auth_headers):
        """Test importing Hugging Face model without model name."""
        import_data = {
            "project_id": 1
        }
        
        response = client.post(
            "/models/import-huggingface",
            headers=auth_headers,
            json=import_data
        )
        
        assert response.status_code == 400
        assert "Model name is required" in response.json()["detail"]


class TestFrontendIntegration:
    """Test frontend integration with backend APIs."""

    def test_dashboard_project_edit_modal(self, client):
        """Test that project edit modal loads correctly."""
        response = client.get("/")
        assert response.status_code == 200
        assert "projectModal" in response.text
        assert "Edit Project" in response.text
        assert "Model Management" in response.text

    def test_dashboard_edit_button_present(self, client):
        """Test that edit buttons are present in project list."""
        response = client.get("/")
        assert response.status_code == 200
        assert "fas fa-edit" in response.text

    def test_dashboard_model_management_section(self, client):
        """Test that model management section is present."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Add Model from Available Models" in response.text
        assert "Import from Hugging Face" in response.text


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_unauthorized_access(self, client):
        """Test accessing protected endpoints without authentication."""
        response = client.put("/projects/1", json={"name": "Test"})
        assert response.status_code == 401

    def test_project_not_owned(self, client, auth_headers):
        """Test accessing a project owned by another user."""
        with patch('app.api.projects.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            
            # Mock project owned by different user
            mock_project = MagicMock()
            mock_project.owner_id = 999  # Different user
            mock_db.query.return_value.filter.return_value.first.return_value = mock_project
            
            response = client.put(
                "/projects/1",
                headers=auth_headers,
                json={"name": "Test"}
            )
            
            assert response.status_code == 403
            assert "Not enough permissions" in response.json()["detail"]

    def test_invalid_project_data(self, client, auth_headers):
        """Test updating project with invalid data."""
        with patch('app.api.projects.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            
            # Mock project
            mock_project = MagicMock()
            mock_project.id = 1
            mock_project.owner_id = 1
            mock_db.query.return_value.filter.return_value.first.return_value = mock_project
            
            # Test with invalid data
            invalid_data = {
                "name": "",  # Empty name
                "tags": "not_a_list"  # Invalid tags format
            }
            
            response = client.put(
                "/projects/1",
                headers=auth_headers,
                json=invalid_data
            )
            
            # Should handle gracefully or return validation error
            assert response.status_code in [400, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 