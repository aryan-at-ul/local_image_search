# index_builder.py
"""
Build FAISS index for fast similarity search.
"""
import os
import numpy as np
import faiss
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, List
import time

from ..config import (
    INDEX_PATH, 
    INDEX_INFOS_PATH, 
    INDEX_TYPE, 
    METRIC_TYPE,
    MAX_INDEX_MEMORY_USAGE,
    MAX_INDEX_QUERY_TIME_MS
)
from ..utils.file_utils import save_index_info

class IndexBuilder:
    """
    Build and manage FAISS index for fast similarity search.
    """
    def __init__(
        self,
        index_type: str = INDEX_TYPE,
        metric_type: str = METRIC_TYPE,
        max_index_memory_usage: str = MAX_INDEX_MEMORY_USAGE,
        max_index_query_time_ms: float = MAX_INDEX_QUERY_TIME_MS
    ):
        """
        Initialize the IndexBuilder.
        
        Args:
            index_type: Type of the FAISS index
            metric_type: Metric type for the index ("inner_product" or "L2")
            max_index_memory_usage: Maximum memory usage for the index
            max_index_query_time_ms: Maximum query time in milliseconds
        """
        self.index_type = index_type
        self.metric_type = metric_type
        self.max_index_memory_usage = max_index_memory_usage
        self.max_index_query_time_ms = max_index_query_time_ms
        
    def create_index(
        self,
        embeddings: np.ndarray,
        metadata_path: Union[str, Path],
        output_path: Union[str, Path] = INDEX_PATH,
        info_path: Union[str, Path] = INDEX_INFOS_PATH
    ) -> faiss.Index:
        """
        Create a FAISS index from embeddings.
        
        Args:
            embeddings: Array of embeddings
            metadata_path: Path to the metadata file
            output_path: Path to save the index
            info_path: Path to save the index info
            
        Returns:
            FAISS index
        """
        print(f"Creating {self.index_type} index with {self.metric_type} metric...")
        
        if isinstance(output_path, str):
            output_path = Path(output_path)
            
        if isinstance(info_path, str):
            info_path = Path(info_path)
            
        # Make sure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
            
        # Get embedding dimension
        num_vectors, dimension = embeddings.shape
        
        # Prepare for metric type
        if self.metric_type == "inner_product":
            faiss_metric = faiss.METRIC_INNER_PRODUCT
            # Ensure vectors are normalized for inner product
            faiss.normalize_L2(embeddings)
        else:  # Default to L2
            faiss_metric = faiss.METRIC_L2
        
        # Create a simpler, more reliable index for saving
        # If the index_type starts with HNSW, we'll use a flat index instead
        # to avoid the saving issues with HNSW
        if self.index_type.startswith("HNSW"):
            print("Note: Using IndexFlatIP/L2 instead of HNSW for reliable saving")
            if faiss_metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(dimension)
            else:
                index = faiss.IndexFlatL2(dimension)
        elif self.index_type.startswith("IVF"):
            # For IVF-type indices, we need to train on the data
            parts = self.index_type.split(",")
            nlist = int(parts[0][3:])  # Extract number after "IVF"
            if len(parts) > 1:
                encoding = parts[1]
                if encoding == "Flat":
                    quantizer = faiss.IndexFlatIP(dimension) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dimension)
                    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss_metric)
                elif encoding == "PQ":
                    # PQ encoding requires an additional parameter M
                    M = 8  # Default value, can be changed
                    quantizer = faiss.IndexFlatIP(dimension) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dimension)
                    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, M, 8)
            else:
                # Default to IVF with flat encoding
                quantizer = faiss.IndexFlatIP(dimension) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss_metric)
                
            # Train the index
            print(f"Training IVF index with {num_vectors} vectors...")
            index.train(embeddings)
        else:
            # Default to flat index
            print(f"Using index type: Flat")
            index = faiss.IndexFlatIP(dimension) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dimension)
            
        # Add vectors to the index
        print(f"Adding {num_vectors} vectors to the index...")
        index.add(embeddings)
        
        # For IVF indices, tune the number of probes for the desired search time
        if hasattr(index, 'nprobe'):
            self._tune_nprobe(index, embeddings)
        
        # Save the index
        print(f"Saving index to {output_path}...")
        try:
            faiss.write_index(index, str(output_path))
            print("Index saved successfully!")
        except RuntimeError as e:
            print(f"Error saving index: {e}")
            print("Trying alternative saving method...")
            
            # Try with a simpler index type as fallback
            if not isinstance(index, (faiss.IndexFlat, faiss.IndexFlatIP, faiss.IndexFlatL2)):
                print("Converting to a simple flat index for saving...")
                flat_index = faiss.IndexFlatIP(dimension) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dimension)
                flat_index.add(embeddings)
                faiss.write_index(flat_index, str(output_path))
                print("Saved using a simple flat index instead.")
                index = flat_index
        
        # Save index info
        print(f"Saving index info to {info_path}...")
        save_index_info(
            index_path=output_path,
            info_path=info_path,
            metadata_path=metadata_path,
            num_vectors=num_vectors,
            dimension=dimension,
            index_type="Flat" if self.index_type.startswith("HNSW") else self.index_type,  # Use actual index type saved
            metric_type=self.metric_type,
            extra_info={
                'max_index_memory_usage': self.max_index_memory_usage,
                'max_index_query_time_ms': self.max_index_query_time_ms,
                'original_index_type': self.index_type  # Store the originally requested index type
            }
        )
        
        return index
        
    def _tune_nprobe(self, index: faiss.Index, embeddings: np.ndarray) -> None:
        """
        Tune the nprobe parameter for IVF indices.
        
        Args:
            index: FAISS IVF index
            embeddings: Sample embeddings for tuning
        """
        print("Tuning nprobe parameter...")
        
        # Get a sample of embeddings for tuning
        sample_size = min(1000, embeddings.shape[0])
        sample_indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
        sample = embeddings[sample_indices]
        
        # Start with nprobe = 1
        index.nprobe = 1
        
        # Measure query time
        start_time = time.time()
        _, _ = index.search(sample, 10)
        end_time = time.time()
        
        query_time_ms = (end_time - start_time) / sample_size * 1000
        
        # Increase nprobe until query time exceeds the limit
        while query_time_ms < self.max_index_query_time_ms and index.nprobe < index.nlist:
            index.nprobe += 1
            
            start_time = time.time()
            _, _ = index.search(sample, 10)
            end_time = time.time()
            
            query_time_ms = (end_time - start_time) / sample_size * 1000
            
        # Back off by one step to ensure we stay within the limit
        index.nprobe = max(1, index.nprobe - 1)
        
        print(f"Tuned nprobe: {index.nprobe}, estimated query time: {query_time_ms:.2f} ms")