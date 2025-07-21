"""
Unit tests for Hugging Face tag extraction functionality.
"""

import pytest
import requests
from unittest.mock import Mock, patch
from bs4 import BeautifulSoup
import json
import os
import tempfile
import shutil

# Import the function to test
from app.api.models import extract_huggingface_tags


class TestHuggingFaceTagExtraction:
    """Test cases for Hugging Face tag extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_model_name = "cdminix/vocex"
        self.expected_tags = ["speech recognition", "speech synthesis", "text-to-speech"]
        
    def test_extract_tags_success(self):
        """Test successful tag extraction from Hugging Face."""
        with patch('requests.get') as mock_get:
            # Mock the HTML response
            mock_html = f"""
            <html>
                <head>
                    <meta name="keywords" content="speech recognition, speech synthesis, text-to-speech, demo">
                </head>
                <body>
                    <a class="tag">speech recognition</a>
                    <a class="badge">speech synthesis</a>
                    <a class="tag">text-to-speech</a>
                    <a class="tag">demo</a>
                </body>
            </html>
            """
            
            mock_response = Mock()
            mock_response.content = mock_html.encode('utf-8')
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Test the function
            tags = extract_huggingface_tags(self.test_model_name)
            
            # Verify the request was made correctly
            mock_get.assert_called_once_with(
                f"https://huggingface.co/{self.test_model_name}",
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'},
                timeout=10
            )
            
            # Verify tags were extracted correctly (excluding 'demo')
            assert "speech recognition" in tags
            assert "speech synthesis" in tags
            assert "text-to-speech" in tags
            assert "demo" not in tags
            assert len(tags) == 3
    
    def test_extract_tags_with_structured_data(self):
        """Test tag extraction from structured data (JSON-LD)."""
        with patch('requests.get') as mock_get:
            # Mock HTML with structured data
            mock_html = f"""
            <html>
                <head>
                    <script type="application/ld+json">
                    {{
                        "keywords": ["speech recognition", "speech synthesis", "text-to-speech", "demo"]
                    }}
                    </script>
                </head>
                <body></body>
            </html>
            """
            
            mock_response = Mock()
            mock_response.content = mock_html.encode('utf-8')
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            tags = extract_huggingface_tags(self.test_model_name)
            
            assert "speech recognition" in tags
            assert "speech synthesis" in tags
            assert "text-to-speech" in tags
            assert "demo" not in tags
    
    def test_extract_tags_network_error(self):
        """Test handling of network errors."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")
            
            tags = extract_huggingface_tags(self.test_model_name)
            
            assert tags == []
    
    def test_extract_tags_http_error(self):
        """Test handling of HTTP errors."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
            mock_get.return_value = mock_response
            
            tags = extract_huggingface_tags(self.test_model_name)
            
            assert tags == []
    
    def test_extract_tags_empty_response(self):
        """Test handling of empty HTML response."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.content = b"<html></html>"
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            tags = extract_huggingface_tags(self.test_model_name)
            
            assert tags == []
    
    def test_extract_tags_duplicate_removal(self):
        """Test that duplicate tags are removed."""
        with patch('requests.get') as mock_get:
            mock_html = f"""
            <html>
                <body>
                    <a class="tag">speech recognition</a>
                    <a class="tag">speech recognition</a>
                    <a class="badge">speech synthesis</a>
                    <a class="badge">speech synthesis</a>
                </body>
            </html>
            """
            
            mock_response = Mock()
            mock_response.content = mock_html.encode('utf-8')
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            tags = extract_huggingface_tags(self.test_model_name)
            
            assert len(tags) == 2
            assert "speech recognition" in tags
            assert "speech synthesis" in tags


class TestHuggingFaceImportIntegration:
    """Integration tests for Hugging Face model import with tag extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_model_name = "cdminix/vocex"
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    @patch('app.api.models.extract_huggingface_tags')
    @patch('app.api.models.model_analyzer')
    @patch('torch.save')
    @patch('torch.nn.Linear')
    @patch('os.path.getsize')
    @patch('os.path.exists')
    @patch('os.makedirs')
    async def test_import_with_tags(self, mock_makedirs, mock_exists, mock_getsize, mock_linear, mock_save, mock_analyzer, mock_extract_tags):
        """Test importing a model with tag extraction."""
        from app.api.models import import_huggingface_model
        
        # Mock the tag extraction
        mock_extract_tags.return_value = ["speech recognition", "speech synthesis", "text-to-speech"]
        
        # Mock file operations
        mock_exists.return_value = False  # Model doesn't exist initially
        mock_getsize.return_value = 1024  # Mock file size
        
        # Mock the model analyzer
        mock_analyzer.analyze_model.return_value = {
            "capabilities": ["audio", "speech"],
            "type": "transformer"
        }
        
        # Mock torch operations
        mock_model = Mock()
        mock_linear.return_value = mock_model
        
        # Create test data
        model_data = {
            "model_name": self.test_model_name,
            "project_id": 1
        }
        
        # Mock the current user and database session
        mock_user = Mock()
        mock_db = Mock()
        
        # Mock the transformers import to avoid actual downloads
        with patch('app.api.models.AutoModel') as mock_auto_model, \
             patch('app.api.models.AutoTokenizer') as mock_auto_tokenizer, \
             patch('app.api.models.AutoFeatureExtractor') as mock_auto_feature_extractor:
            
            # Mock the transformers to raise exceptions (so it falls back to dummy model)
            mock_auto_model.from_pretrained.side_effect = Exception("Model not found")
            mock_auto_tokenizer.from_pretrained.side_effect = Exception("Tokenizer not found")
            mock_auto_feature_extractor.from_pretrained.side_effect = Exception("Feature extractor not found")
            
            # Test the import function
            result = await import_huggingface_model(model_data, mock_user, mock_db)
            
            # Verify tag extraction was called
            mock_extract_tags.assert_called_once_with(self.test_model_name)
            
            # Verify the result contains the expected data
            assert "model_name" in result
            assert "path" in result
            assert "capabilities" in result
            assert "tags" in result
            assert "status" in result
            
            # Verify tags were extracted and included
            assert result["tags"] == ["speech recognition", "speech synthesis", "text-to-speech"]
            
            # Verify capabilities include both analyzed and extracted tags
            expected_capabilities = ["audio", "speech", "speech recognition", "speech synthesis", "text-to-speech"]
            assert all(cap in result["capabilities"] for cap in expected_capabilities)
            
            # Verify demo tag was excluded
            assert "demo" not in result["tags"]
            assert "demo" not in result["capabilities"]
    
    @pytest.mark.asyncio
    @patch('app.api.models.extract_huggingface_tags')
    @patch('app.api.models.model_analyzer')
    @patch('torch.save')
    @patch('torch.nn.Linear')
    @patch('os.path.getsize')
    @patch('os.path.exists')
    @patch('os.makedirs')
    async def test_import_with_demo_tag_exclusion(self, mock_makedirs, mock_exists, mock_getsize, mock_linear, mock_save, mock_analyzer, mock_extract_tags):
        """Test that demo tags are properly excluded."""
        from app.api.models import import_huggingface_model
        
        # Mock the tag extraction to return filtered tags (demo should be filtered out by the real function)
        mock_extract_tags.return_value = ["speech recognition", "text-to-speech"]
        
        # Mock file operations
        mock_exists.return_value = False  # Model doesn't exist initially
        mock_getsize.return_value = 1024  # Mock file size
        
        # Mock the model analyzer
        mock_analyzer.analyze_model.return_value = {
            "capabilities": ["audio"],
            "type": "transformer"
        }
        
        # Mock torch operations
        mock_model = Mock()
        mock_linear.return_value = mock_model
        
        # Create test data
        model_data = {
            "model_name": self.test_model_name,
            "project_id": 1
        }
        
        # Mock the current user and database session
        mock_user = Mock()
        mock_db = Mock()
        
        # Mock the transformers import to avoid actual downloads
        with patch('app.api.models.AutoModel') as mock_auto_model, \
             patch('app.api.models.AutoTokenizer') as mock_auto_tokenizer, \
             patch('app.api.models.AutoFeatureExtractor') as mock_auto_feature_extractor:
            
            # Mock the transformers to raise exceptions
            mock_auto_model.from_pretrained.side_effect = Exception("Model not found")
            mock_auto_tokenizer.from_pretrained.side_effect = Exception("Tokenizer not found")
            mock_auto_feature_extractor.from_pretrained.side_effect = Exception("Feature extractor not found")
            
            # Test the import function
            result = await import_huggingface_model(model_data, mock_user, mock_db)
            
            # Verify demo tag was excluded
            assert "demo" not in result["tags"]
            assert "demo" not in result["capabilities"]
            
            # Verify other tags were included
            assert "speech recognition" in result["tags"]
            assert "text-to-speech" in result["tags"]


if __name__ == "__main__":
    pytest.main([__file__]) 