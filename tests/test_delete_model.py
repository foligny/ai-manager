"""
Unit tests for model deletion functionality.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.api.models import delete_model


class TestDeleteModel:
    """Test cases for model deletion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_model_name = "test_model.pth"
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    @patch('app.api.models.os.path.exists')
    @patch('app.api.models.os.remove')
    @patch('app.api.models.os.path.join')
    @patch('app.api.models.os.getcwd')
    async def test_delete_model_success(self, mock_getcwd, mock_join, mock_remove, mock_exists):
        """Test successful model deletion."""
        # Mock file operations
        mock_getcwd.return_value = self.temp_dir
        mock_join.return_value = os.path.join(self.temp_dir, self.test_model_name)
        mock_exists.return_value = True
        
        # Mock user and database
        mock_user = Mock()
        mock_db = Mock()
        
        # Test the delete function
        result = await delete_model(self.test_model_name, mock_user, mock_db)
        
        # Verify the result
        assert result["message"] == f"Model {self.test_model_name} deleted successfully"
        assert result["deleted_model"] == self.test_model_name
        
        # Verify file operations were called
        mock_exists.assert_called_once()
        mock_remove.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.api.models.os.path.exists')
    @patch('app.api.models.os.path.join')
    @patch('app.api.models.os.getcwd')
    async def test_delete_model_not_found(self, mock_getcwd, mock_join, mock_exists):
        """Test deletion of non-existent model."""
        # Mock file operations
        mock_getcwd.return_value = self.temp_dir
        mock_join.return_value = os.path.join(self.temp_dir, self.test_model_name)
        mock_exists.return_value = False
        
        # Mock user and database
        mock_user = Mock()
        mock_db = Mock()
        
        # Test that it raises HTTPException
        with pytest.raises(Exception) as exc_info:
            await delete_model(self.test_model_name, mock_user, mock_db)
        
        # Verify the error message
        assert "not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('app.api.models.os.path.exists')
    @patch('app.api.models.os.remove')
    @patch('app.api.models.os.path.join')
    @patch('app.api.models.os.getcwd')
    async def test_delete_model_with_directory(self, mock_getcwd, mock_join, mock_remove, mock_exists):
        """Test deletion of model with associated directory."""
        # Mock file operations
        mock_getcwd.return_value = self.temp_dir
        mock_join.return_value = os.path.join(self.temp_dir, self.test_model_name)
        mock_exists.return_value = True
        
        # Mock user and database
        mock_user = Mock()
        mock_db = Mock()
        
        # Mock shutil.rmtree and os.path.isdir
        with patch('app.api.models.shutil.rmtree') as mock_rmtree:
            with patch('app.api.models.os.path.isdir') as mock_isdir:
                mock_isdir.return_value = True
                
                # Test the delete function
                result = await delete_model(self.test_model_name, mock_user, mock_db)
                
                # Verify the result
                assert result["message"] == f"Model {self.test_model_name} deleted successfully"
                
                # Verify both file and directory were deleted
                mock_remove.assert_called_once()
                mock_rmtree.assert_called_once()


class TestDeleteModelAPI:
    """Integration tests for delete model API endpoint."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_model_name = "test_model.pth"
        
    @patch('app.api.models.os.path.exists')
    @patch('app.api.models.os.remove')
    @patch('app.api.models.os.path.join')
    @patch('app.api.models.os.getcwd')
    def test_delete_model_api_success(self, mock_getcwd, mock_join, mock_remove, mock_exists):
        """Test successful model deletion via API."""
        # Mock file operations
        mock_getcwd.return_value = "/tmp"
        mock_join.return_value = "/tmp/test_model.pth"
        mock_exists.return_value = True
        
        # Mock authentication and database
        with patch('app.api.models.get_current_user') as mock_get_user, \
             patch('app.api.models.get_db') as mock_get_db:
            
            mock_user = Mock()
            mock_get_user.return_value = mock_user
            
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            # Test the API endpoint
            response = self.client.delete(f"/models/{self.test_model_name}")
            
            # Verify the response
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == f"Model {self.test_model_name} deleted successfully"
            assert data["deleted_model"] == self.test_model_name
    
    @patch('app.api.models.os.path.exists')
    @patch('app.api.models.os.path.join')
    @patch('app.api.models.os.getcwd')
    def test_delete_model_api_not_found(self, mock_getcwd, mock_join, mock_exists):
        """Test deletion of non-existent model via API."""
        # Mock file operations
        mock_getcwd.return_value = "/tmp"
        mock_join.return_value = "/tmp/test_model.pth"
        mock_exists.return_value = False
        
        # Mock authentication and database
        with patch('app.api.models.get_current_user') as mock_get_user, \
             patch('app.api.models.get_db') as mock_get_db:
            
            mock_user = Mock()
            mock_get_user.return_value = mock_user
            
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            # Test the API endpoint
            response = self.client.delete(f"/models/{self.test_model_name}")
            
            # Verify the response
            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"]


if __name__ == "__main__":
    pytest.main([__file__]) 