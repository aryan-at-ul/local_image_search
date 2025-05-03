# image_processor.py
"""
Image processing utilities for loading and preprocessing images.
"""
import os
import torch
from PIL import Image
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
from tqdm import tqdm
import clip

from ..config import IMAGE_SIZE

def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)
        
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def get_clip_preprocess(model_name: str = None) -> callable:
    """
    Get the preprocessing transform for the CLIP model.
    
    Args:
        model_name: Name of the CLIP model
        
    Returns:
        Preprocessing transform function
    """
    if model_name:
        _, preprocess = clip.load(model_name, device="cpu", jit=False)
    else:
        # Return a standard preprocessing pipeline if model is not loaded
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                 (0.26862954, 0.26130258, 0.27577711))
        ])
    
    return preprocess

def process_batch(
    image_paths: List[Union[str, Path]], 
    preprocess: callable,
    batch_size: int = 32
) -> Dict[str, Union[List[torch.Tensor], List[str], List[int]]]:
    """
    Process a batch of images.
    
    Args:
        image_paths: List of image paths
        preprocess: Preprocessing function
        batch_size: Number of images to process at once
        
    Returns:
        Dictionary with processed images and metadata
    """
    results = {
        'processed_images': [],
        'valid_paths': [],
        'valid_indices': []
    }
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = []
        
        for j, path in enumerate(batch_paths):
            img = load_image(path)
            
            if img is not None:
                try:
                    tensor = preprocess(img)
                    batch_tensors.append(tensor)
                    results['valid_paths'].append(str(path))
                    results['valid_indices'].append(i + j)
                except Exception as e:
                    print(f"Error preprocessing image {path}: {e}")
        
        if batch_tensors:
            batch_tensor = torch.stack(batch_tensors)
            results['processed_images'].append(batch_tensor)
    
    return results

def process_query_image(
    image_path: Union[str, Path], 
    preprocess: callable
) -> Optional[torch.Tensor]:
    """
    Process a query image.
    
    Args:
        image_path: Path to the query image
        preprocess: Preprocessing function
        
    Returns:
        Preprocessed image tensor or None if processing failed
    """
    img = load_image(image_path)
    
    if img is not None:
        try:
            tensor = preprocess(img)
            return torch.unsqueeze(tensor, 0)
        except Exception as e:
            print(f"Error preprocessing query image {image_path}: {e}")
    
    return None