"""
Simple Flask server to serve the index.html file and handle API requests.
"""
import os
import tempfile
from pathlib import Path
import argparse
import torch
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# Import the local image search modules
from src.config import CLIP_MODEL
from src.utils.file_utils import get_image_paths
from src.indexer.embedder import ClipEmbedder
from src.indexer.index_builder import IndexBuilder
from src.searcher.query_processor import QueryProcessor
from src.searcher.retriever import ImageRetriever

# Define the path to your index.html file
INDEX_HTML_PATH = "/media/annatar/OLDHDD/local_image_search/index.html"

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

@app.route('/')
def index():
    """Serve the main index.html file."""
    return send_file(INDEX_HTML_PATH)

@app.route('/api/status')
def status():
    """Return status information."""
    try:
        # Check if index exists
        from src.config import INDEX_PATH
        index_exists = Path(INDEX_PATH).exists()
        
        return jsonify({
            "status": "running",
            "index_exists": index_exists,
            "device": device
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/index', methods=['POST'])
def index_directory():
    """Index a directory of images."""
    try:
        data = request.get_json()
        
        if not data or 'directory' not in data:
            return jsonify({"status": "error", "message": "No directory specified"})
        
        directory = data['directory']
        recursive = data.get('recursive', True)
        
        if not Path(directory).exists():
            return jsonify({"status": "error", "message": f"Directory not found: {directory}"})
        
        # Get image paths
        image_paths = get_image_paths(directory, recursive=recursive)
        
        if not image_paths:
            return jsonify({"status": "error", "message": "No images found in directory"})
        
        print(f"Found {len(image_paths)} images, starting indexing process...")
        
        # Generate embeddings
        embedder = ClipEmbedder(model_name=CLIP_MODEL, device=device)
        embeddings, valid_paths = embedder.generate_embeddings(image_paths=image_paths)
        
        if len(valid_paths) == 0:
            return jsonify({"status": "error", "message": "No valid images found"})
        
        # Get metadata path
        metadata_path = getattr(embedder, 'metadata_path', None)
        if metadata_path is None:
            # Fallback to default metadata path
            from src.config import METADATA_DIR
            metadata_path = Path(METADATA_DIR) / "metadata.parquet"
        
        # Build index
        index_builder = IndexBuilder()
        index = index_builder.create_index(
            embeddings=embeddings,
            metadata_path=metadata_path
        )
        
        return jsonify({
            "status": "success",
            "message": f"Successfully indexed {len(valid_paths)} images"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": f"Error during indexing: {str(e)}"
        })

@app.route('/api/search', methods=['POST'])
def search():
    """Search for images using text or image query."""
    try:
        # Check content type to handle both JSON and form data properly
        content_type = request.content_type
        
        # Text search (JSON)
        if content_type and 'application/json' in content_type:
            # Text query
            if not request.json or 'text_query' not in request.json:
                return jsonify({
                    "status": "error", 
                    "message": "Missing text_query in JSON data"
                })
                
            text_query = request.json['text_query']
            num_results = request.json.get('num_results', 10)
            
            # Process query
            processor = QueryProcessor(model_name=CLIP_MODEL, device=device)
            query_embedding = processor.process_text_query(text_query)
            
        # Image search (multipart form)
        elif content_type and 'multipart/form-data' in content_type:
            # Image query
            if 'image' not in request.files:
                return jsonify({
                    "status": "error", 
                    "message": "No image file provided"
                })
                
            image_file = request.files['image']
            num_results = int(request.form.get('num_results', 10))
            
            # Create uploads directory if it doesn't exist
            Path("uploads").mkdir(exist_ok=True)
            
            # Save the uploaded image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir='uploads') as temp_file:
                image_path = temp_file.name
                image_file.save(image_path)
            
            # Process query
            processor = QueryProcessor(model_name=CLIP_MODEL, device=device)
            query_embedding = processor.process_image_query(image_path)
            
            # Clean up the temporary file after processing
            try:
                os.unlink(image_path)
            except:
                # Non-critical error, we can continue
                pass
        else:
            return jsonify({
                "status": "error", 
                "message": f"Unsupported content type: {content_type}. Use application/json for text search or multipart/form-data for image search."
            })
        
        if query_embedding is None:
            return jsonify({
                "status": "error", 
                "message": "Failed to process query"
            })
        
        # Search
        retriever = ImageRetriever()
        distances, indices, _ = retriever.search(query_embedding, k=num_results)
        
        # Format results
        results = retriever.format_results(distances, indices)
        
        return jsonify({
            "status": "success",
            "results": results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": f"Error during search: {str(e)}"
        })

@app.route('/image/<path:filename>')
def serve_image(filename):
    """
    Serve an image file directly from the filesystem.
    This handles paths correctly whether they come in with or without leading slash.
    """
    # Remove any leading slash for consistency
    if filename.startswith('/'):
        filename = filename[1:]
    
    # Now handle the path properly
    # For absolute paths like /media/annatar/...
    if filename.startswith('media/'):
        # Split into directory and actual filename
        directory = os.path.dirname(f"/{filename}")
        basename = os.path.basename(filename)
        return send_from_directory(directory, basename)
    else:
        # Split the path for relative paths
        directory = os.path.dirname(filename)
        basename = os.path.basename(filename)
        if directory:
            return send_from_directory(directory, basename)
        else:
            return send_from_directory('.', basename)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start the image search server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Create the uploads directory if it doesn't exist
    Path("uploads").mkdir(exist_ok=True)
    
    # Start server
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"Serving HTML file from: {INDEX_HTML_PATH}")
    app.run(host=args.host, port=args.port, debug=args.debug)