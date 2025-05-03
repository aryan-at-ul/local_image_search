# embedder.py
"""
CLIP embedding generation for images.
"""
import os
import torch
import numpy as np
import clip
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
from tqdm import tqdm
import pandas as pd

from ..config import CLIP_MODEL, BATCH_SIZE, IMAGE_EMBEDDING_DIR, METADATA_DIR
from .image_processor import get_clip_preprocess, process_batch
from ..utils.file_utils import save_metadata

class ClipEmbedder:
    """
    Generate CLIP embeddings for images.
    """
    def __init__(
        self,
        model_name: str = CLIP_MODEL,
        device: str = None,
        batch_size: int = BATCH_SIZE
    ):
        """
        Initialize the ClipEmbedder.
        
        Args:
            model_name: Name of the CLIP model
            device: Device to use for inference ("cuda" or "cpu")
            batch_size: Batch size for processing images
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        self.batch_size = batch_size
        self.embedding_dim = self.model.visual.output_dim
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
    def generate_embeddings(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path] = IMAGE_EMBEDDING_DIR,
        metadata_dir: Union[str, Path] = METADATA_DIR,
        metadata_filename: str = "metadata.parquet"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for a list of images.
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save embeddings
            metadata_dir: Directory to save metadata
            metadata_filename: Filename for metadata
            
        Returns:
            Tuple of (embeddings array, list of valid image paths)
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
            
        if isinstance(metadata_dir, str):
            metadata_dir = Path(metadata_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images in batches
        processed_data = process_batch(
            image_paths=image_paths,
            preprocess=self.preprocess,
            batch_size=self.batch_size
        )
        
        # Get valid image paths
        valid_paths = processed_data['valid_paths']
        
        if not valid_paths:
            print("No valid images found.")
            return np.array([]), []
        
        # Generate embeddings
        all_embeddings = []
        
        for batch_tensor in tqdm(processed_data['processed_images'], desc="Generating embeddings"):
            # Move batch to device
            batch_tensor = batch_tensor.to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                batch_embedding = self.model.encode_image(batch_tensor)
                
            # Normalize embeddings
            batch_embedding /= batch_embedding.norm(dim=-1, keepdim=True)
            
            # Move to CPU and convert to numpy
            batch_embedding = batch_embedding.cpu().detach().numpy()
            
            all_embeddings.append(batch_embedding)
        
        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        
        # Save metadata
        metadata_path = metadata_dir / metadata_filename
        save_metadata(
            image_paths=[Path(p) for p in valid_paths],
            output_file=metadata_path
        )
        
        # Save embeddings
        embedding_path = output_dir / "embeddings.npy"
        np.save(embedding_path, embeddings)
        
        print(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
        print(f"Embeddings saved to {embedding_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return embeddings, valid_paths