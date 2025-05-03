# config.py
"""
Configuration settings for the local image search application.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "/media/annatar/OLDHDD/flikr_caption_dataset/archive/Images"
MODELS_DIR = BASE_DIR / "models"

# CLIP model settings
CLIP_MODEL = "ViT-B/32"  # Options: "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101", etc.
IMAGE_SIZE = 224  # Size to resize images to before embedding

# Index settings
INDEX_NAME = "image_search.index"
INDEX_PATH = MODELS_DIR / INDEX_NAME
INDEX_INFOS_PATH = MODELS_DIR / "index_infos.json"
INDEX_TYPE = "HNSW32"  # Options: "Flat", "HNSW32", "IVF4096,Flat", etc.
METRIC_TYPE = "inner_product"  # Options: "inner_product", "L2"
MAX_INDEX_MEMORY_USAGE = "4GB"
MAX_INDEX_QUERY_TIME_MS = 10

# Embedding settings
EMBEDDING_DIR = DATA_DIR / "embeddings"
METADATA_DIR = EMBEDDING_DIR / "metadata"
IMAGE_EMBEDDING_DIR = EMBEDDING_DIR / "img_emb"
BATCH_SIZE = 32

# Search settings
NUM_RESULTS = 5

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, EMBEDDING_DIR, METADATA_DIR, IMAGE_EMBEDDING_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)