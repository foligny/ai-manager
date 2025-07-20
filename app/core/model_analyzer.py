"""
Real AI Model Analysis Module

This module provides actual AI analysis of model files to determine their capabilities,
instead of relying on filename patterns.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
import json
from transformers import AutoConfig, AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """Analyzes model files to determine their actual capabilities."""
    
    def __init__(self):
        self.model_cache = {}
        self.config_cache = {}
    
    def analyze_model(self, model_path: str) -> Dict[str, Any]:
        """
        Analyze a model file to determine its capabilities.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary containing model capabilities and metadata
        """
        try:
            # Try to load as PyTorch model
            if model_path.endswith(('.pth', '.pt')):
                return self._analyze_pytorch_model(model_path)
            elif model_path.endswith('.pkl'):
                return self._analyze_pickle_model(model_path)
            else:
                return self._get_generic_capabilities(model_path)
                
        except Exception as e:
            logger.warning(f"Error analyzing model {model_path}: {e}")
            return self._get_fallback_capabilities(model_path)
    
    def _analyze_pytorch_model(self, model_path: str) -> Dict[str, Any]:
        """Analyze a PyTorch model file."""
        try:
            # Try to load the model
            model_data = torch.load(model_path, map_location='cpu')
            
            # Analyze model structure
            capabilities = []
            model_type = "unknown"
            input_types = []
            output_types = []
            
            # Check if it's a state dict or full model
            if isinstance(model_data, dict):
                # State dict analysis
                layer_names = list(model_data.keys())
                
                # Analyze based on layer names
                if any('conv' in name.lower() for name in layer_names):
                    capabilities.extend(['image', 'image_classification'])
                    input_types.append('image')
                    model_type = "cnn"
                
                if any('lstm' in name.lower() or 'rnn' in name.lower() for name in layer_names):
                    capabilities.extend(['text', 'sequence'])
                    input_types.append('text')
                    model_type = "rnn"
                
                if any('transformer' in name.lower() or 'attention' in name.lower() for name in layer_names):
                    capabilities.extend(['text', 'transformer'])
                    input_types.append('text')
                    model_type = "transformer"
                
                if any('wav' in name.lower() or 'audio' in name.lower() for name in layer_names):
                    capabilities.extend(['audio', 'speech_recognition'])
                    input_types.append('audio')
                    model_type = "audio"
                
                # Check for specific model patterns
                if any('resnet' in name.lower() for name in layer_names):
                    capabilities.extend(['image', 'image_classification'])
                    model_type = "resnet"
                
                if any('bert' in name.lower() for name in layer_names):
                    capabilities.extend(['text', 'text_classification', 'sentiment_analysis'])
                    model_type = "bert"
                
                if any('wav2vec' in name.lower() for name in layer_names):
                    capabilities.extend(['audio', 'speech_recognition'])
                    model_type = "wav2vec"
                
            else:
                # Full model analysis
                model = model_data
                if hasattr(model, 'modules'):
                    for module in model.modules():
                        if isinstance(module, nn.Conv2d):
                            capabilities.extend(['image', 'image_classification'])
                            input_types.append('image')
                            model_type = "cnn"
                        elif isinstance(module, nn.LSTM):
                            capabilities.extend(['text', 'sequence'])
                            input_types.append('text')
                            model_type = "lstm"
                        elif isinstance(module, nn.Transformer):
                            capabilities.extend(['text', 'transformer'])
                            input_types.append('text')
                            model_type = "transformer"
            
            # If no specific capabilities detected, try to infer from model size
            if not capabilities:
                file_size = os.path.getsize(model_path)
                if file_size > 100 * 1024 * 1024:  # > 100MB
                    capabilities = ['large_model', 'generic']
                else:
                    capabilities = ['generic']
            
            return {
                "capabilities": list(set(capabilities)),
                "model_type": model_type,
                "input_types": list(set(input_types)),
                "output_types": output_types,
                "file_size": os.path.getsize(model_path),
                "analysis_method": "pytorch_analysis"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing PyTorch model {model_path}: {e}")
            return self._get_fallback_capabilities(model_path)
    
    def _analyze_pickle_model(self, model_path: str) -> Dict[str, Any]:
        """Analyze a pickle model file."""
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Analyze pickle model
            capabilities = ['generic']
            model_type = "pickle"
            
            if hasattr(model_data, 'predict'):
                capabilities.append('prediction')
            
            if hasattr(model_data, 'fit'):
                capabilities.append('training')
            
            return {
                "capabilities": capabilities,
                "model_type": model_type,
                "input_types": ['generic'],
                "output_types": ['prediction'],
                "file_size": os.path.getsize(model_path),
                "analysis_method": "pickle_analysis"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pickle model {model_path}: {e}")
            return self._get_fallback_capabilities(model_path)
    
    def _get_fallback_capabilities(self, model_path: str) -> Dict[str, Any]:
        """Get fallback capabilities when analysis fails."""
        file_size = os.path.getsize(model_path)
        
        # Infer from file size
        if file_size < 1024:  # < 1KB
            return {
                "capabilities": ['test', 'small'],
                "model_type": "test",
                "input_types": ['generic'],
                "output_types": ['test'],
                "file_size": file_size,
                "analysis_method": "fallback_size"
            }
        elif file_size > 100 * 1024 * 1024:  # > 100MB
            return {
                "capabilities": ['large_model', 'generic'],
                "model_type": "large",
                "input_types": ['generic'],
                "output_types": ['prediction'],
                "file_size": file_size,
                "analysis_method": "fallback_size"
            }
        else:
            return {
                "capabilities": ['generic'],
                "model_type": "generic",
                "input_types": ['generic'],
                "output_types": ['prediction'],
                "file_size": file_size,
                "analysis_method": "fallback_generic"
            }
    
    def _get_generic_capabilities(self, model_path: str) -> Dict[str, Any]:
        """Get generic capabilities for unknown file types."""
        return {
            "capabilities": ['generic'],
            "model_type": "unknown",
            "input_types": ['generic'],
            "output_types": ['prediction'],
            "file_size": os.path.getsize(model_path),
            "analysis_method": "generic"
        }
    
    def get_supported_formats(self, capabilities: List[str]) -> List[str]:
        """Get supported file formats based on model capabilities."""
        formats = []
        
        if 'image' in capabilities or 'image_classification' in capabilities:
            formats.extend(['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
        
        if 'text' in capabilities or 'text_classification' in capabilities:
            formats.extend(['txt', 'json', 'csv'])
        
        if 'audio' in capabilities or 'speech_recognition' in capabilities:
            formats.extend(['wav', 'mp3', 'flac', 'm4a'])
        
        if 'generic' in capabilities:
            formats.extend(['csv', 'json', 'npy', 'txt'])
        
        return list(set(formats))  # Remove duplicates


# Global analyzer instance
model_analyzer = ModelAnalyzer() 