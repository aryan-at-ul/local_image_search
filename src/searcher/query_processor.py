# query_processor.py
"""
Process text and image queries for CLIP-based search.
"""
import os
import torch
import numpy as np
import clip
from PIL import Image
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple

from ..config import CLIP_MODEL
from ..indexer.image_processor import load_image, process_query_image

class QueryProcessor:
    """
    Process text and image queries for CLIP-based search.
    """
    def __init__(
        self,
        model_name: str = CLIP_MODEL,
        device: str = None
    ):
        """
        Initialize the QueryProcessor.
        
        Args:
            model_name: Name of the CLIP model
            device: Device to use for inference ("cuda" or "cpu")
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        self.embedding_dim = self.model.visual.output_dim
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
    def process_text_query(self, text: str) -> np.ndarray:
        """
        Process a text query.
        
        Args:
            text: Text query
            
        Returns:
            Embedding of the text query
        """
        # Tokenize the text
        text_tokens = clip.tokenize([text], truncate=True).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            
        # Normalize embedding
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        text_embedding = text_features.cpu().detach().numpy().astype(np.float32)
        
        return text_embedding
        
    def process_image_query(
        self, 
        image_path: Union[str, Path]
    ) -> Optional[np.ndarray]:
        """
        Process an image query.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Embedding of the image query or None if processing failed
        """
        # Process the image
        image_tensor = process_query_image(image_path, self.preprocess)
        
        if image_tensor is None:
            return None
            
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            
        # Normalize embedding
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        image_embedding = image_features.cpu().detach().numpy().astype(np.float32)
        
        return image_embedding