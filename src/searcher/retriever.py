# retriever.py
"""
Retrieve images from FAISS index based on query embeddings.
"""
import os
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional

from ..config import INDEX_PATH, INDEX_INFOS_PATH, NUM_RESULTS
from ..utils.file_utils import load_index_info, load_metadata

class ImageRetriever:
    """
    Retrieve images from FAISS index based on query embeddings.
    """
    def __init__(
        self,
        index_path: Union[str, Path] = INDEX_PATH,
        info_path: Union[str, Path] = INDEX_INFOS_PATH
    ):
        """
        Initialize the ImageRetriever.
        
        Args:
            index_path: Path to the FAISS index
            info_path: Path to the index info file
        """
        if isinstance(index_path, str):
            index_path = Path(index_path)
            
        if isinstance(info_path, str):
            info_path = Path(info_path)
            
        if not index_path.exists():
            raise FileNotFoundError(f"Index file '{index_path}' does not exist")
            
        if not info_path.exists():
            raise FileNotFoundError(f"Index info file '{info_path}' does not exist")
            
        # Load index info
        self.index_info = load_index_info(info_path)
        self.metadata_path = Path(self.index_info['metadata_path'])
        
        # Load index
        print(f"Loading index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        print(f"Index loaded with {self.index.ntotal} vectors")
        
        # Load metadata
        self.metadata = load_metadata(self.metadata_path)
        print(f"Metadata loaded with {len(self.metadata)} entries")
        
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = NUM_RESULTS
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Search for images similar to the query embedding.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        # Ensure query is the right shape and type
        if len(query_embedding.shape) == 2 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding[0]
            
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Get metadata for the results
        result_metadata = self.metadata.iloc[indices[0]]
        
        return distances, indices, result_metadata
        
    def get_result_paths(self, indices: np.ndarray) -> List[Path]:
        """
        Get the file paths for the given indices.
        
        Args:
            indices: Indices of the results
            
        Returns:
            List of file paths
        """
        result_metadata = self.metadata.iloc[indices]
        return [Path(p) for p in result_metadata['image_path']]
        
    def format_results(
        self,
        distances: np.ndarray,
        indices: np.ndarray
    ) -> List[Dict]:
        """
        Format search results.
        
        Args:
            distances: Distances from the query
            indices: Indices of the results
            
        Returns:
            List of dictionaries with result information
        """
        results = []
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.metadata):  # Valid index
                image_path = self.metadata.iloc[idx]['image_path']
                
                # Get additional metadata
                file_name = self.metadata.iloc[idx]['file_name']
                folder_name = self.metadata.iloc[idx]['folder_name']
                
                result = {
                    'rank': i + 1,
                    'similarity': float(dist),
                    'image_path': image_path,
                    'file_name': file_name,
                    'folder_name': folder_name
                }
                
                results.append(result)
                
        return results