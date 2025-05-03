# file_utils.py
"""
Utility functions for file and directory operations.
"""
import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Optional

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

def get_image_paths(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    Get all image paths in a directory.
    
    Args:
        directory: Directory path to search for images
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of Path objects for all images in the directory
    """
    if isinstance(directory, str):
        directory = Path(directory)
        
    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist")
        
    image_paths = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    image_paths.append(file_path)
    else:
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(file_path)
                
    return image_paths

def save_metadata(image_paths: List[Path], output_file: Union[str, Path]) -> None:
    """
    Save image metadata to a parquet file.
    
    Args:
        image_paths: List of image paths
        output_file: Path to save the metadata
    """
    if isinstance(output_file, str):
        output_file = Path(output_file)
        
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'image_path': [str(path) for path in image_paths],
        'folder_name': [path.parent.name for path in image_paths],
        'file_name': [path.name for path in image_paths],
    }
    
    df = pd.DataFrame(metadata)
    df.to_parquet(output_file)
    
def load_metadata(metadata_file: Union[str, Path]) -> pd.DataFrame:
    """
    Load image metadata from a parquet file.
    
    Args:
        metadata_file: Path to the metadata file
        
    Returns:
        DataFrame containing image metadata
    """
    if isinstance(metadata_file, str):
        metadata_file = Path(metadata_file)
        
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file '{metadata_file}' does not exist")
        
    return pd.read_parquet(metadata_file)

def save_index_info(
    index_path: Union[str, Path],
    info_path: Union[str, Path],
    metadata_path: Union[str, Path],
    num_vectors: int,
    dimension: int,
    index_type: str,
    metric_type: str,
    extra_info: Optional[Dict] = None
) -> None:
    """
    Save index information to a JSON file.
    
    Args:
        index_path: Path to the index file
        info_path: Path to save the info file
        metadata_path: Path to the metadata file
        num_vectors: Number of vectors in the index
        dimension: Dimension of the vectors
        index_type: Type of the index
        metric_type: Metric type used by the index
        extra_info: Additional information to save
    """
    if isinstance(info_path, str):
        info_path = Path(info_path)
        
    info_path.parent.mkdir(parents=True, exist_ok=True)
    
    info = {
        'index_path': str(index_path),
        'metadata_path': str(metadata_path),
        'num_vectors': num_vectors,
        'dimension': dimension,
        'index_type': index_type,
        'metric_type': metric_type,
        'created_at': pd.Timestamp.now().isoformat(),
    }
    
    if extra_info:
        info.update(extra_info)
        
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
        
def load_index_info(info_path: Union[str, Path]) -> Dict:
    """
    Load index information from a JSON file.
    
    Args:
        info_path: Path to the info file
        
    Returns:
        Dictionary containing index information
    """
    if isinstance(info_path, str):
        info_path = Path(info_path)
        
    if not info_path.exists():
        raise FileNotFoundError(f"Info file '{info_path}' does not exist")
        
    with open(info_path, 'r') as f:
        return json.load(f)