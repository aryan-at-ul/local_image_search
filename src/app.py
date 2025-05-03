"""
Main application entry point for the local image search pipeline.
"""
import os
import argparse
from pathlib import Path
from typing import Union, List, Dict, Optional

from .config import CLIP_MODEL, NUM_RESULTS
from .utils.file_utils import get_image_paths
from .indexer.embedder import ClipEmbedder
from .indexer.index_builder import IndexBuilder
from .searcher.query_processor import QueryProcessor
from .searcher.retriever import ImageRetriever
from .utils.visualizer import display_query_results, create_html_gallery

def index_images(
    directory: Union[str, Path],
    recursive: bool = True,
    model_name: str = CLIP_MODEL,
    device: Optional[str] = None,
    verbose: bool = True
) -> None:
    """
    Index images in a directory.
    
    Args:
        directory: Directory containing images
        recursive: Whether to search recursively
        model_name: Name of the CLIP model
        device: Device to use for inference
        verbose: Whether to print verbose output
    """
    if verbose:
        print(f"Indexing images in {directory}...")
    
    # Get image paths
    image_paths = get_image_paths(directory, recursive=recursive)
    if verbose:
        print(f"Found {len(image_paths)} images")
    
    if not image_paths:
        print("No images found. Exiting.")
        return
    
    # Generate embeddings
    embedder = ClipEmbedder(model_name=model_name, device=device)
    embeddings, valid_paths = embedder.generate_embeddings(image_paths=image_paths)
    
    if len(valid_paths) == 0:
        print("No valid images found. Exiting.")
        return
    
    # Get metadata path
    metadata_path = getattr(embedder, 'metadata_path', None)
    if metadata_path is None:
        # Fallback to default metadata path
        from .config import METADATA_DIR
        metadata_path = Path(METADATA_DIR) / "metadata.parquet"
    
    # Build index
    index_builder = IndexBuilder()
    index = index_builder.create_index(
        embeddings=embeddings,
        metadata_path=metadata_path
    )
    
    if verbose:
        print("Indexing complete!")
    
def search_images(
    query: str,
    is_image_query: bool = False,
    model_name: str = CLIP_MODEL,
    device: Optional[str] = None,
    num_results: int = NUM_RESULTS,
    display_results: bool = True,
    save_html: bool = False,
    html_output: Optional[Union[str, Path]] = None,
    headless: bool = False,
    verbose: bool = True
) -> List[Dict]:
    """
    Search for images similar to the query.
    
    Args:
        query: Query text or image path
        is_image_query: Whether the query is an image path
        model_name: Name of the CLIP model
        device: Device to use for inference
        num_results: Number of results to return
        display_results: Whether to display the results
        save_html: Whether to save the results as an HTML gallery
        html_output: Path to save the HTML gallery
        headless: Whether to avoid using matplotlib for display
        verbose: Whether to print verbose output
        
    Returns:
        List of dictionaries with result information
    """
    # Process the query
    processor = QueryProcessor(model_name=model_name, device=device)
    
    if is_image_query:
        # Query is an image path
        query_embedding = processor.process_image_query(query)
        query_display = f"Image: {query}"
    else:
        # Query is text
        query_embedding = processor.process_text_query(query)
        query_display = f"Text: {query}"
    
    if query_embedding is None:
        print("Failed to process query. Exiting.")
        return []
    
    # Search the index
    retriever = ImageRetriever()
    distances, indices, _ = retriever.search(query_embedding, k=num_results)
    
    # Format the results
    results = retriever.format_results(distances, indices)
    
    # Display the results
    if display_results:
        display_query_results(query_display, results)
    
    # Save as HTML gallery
    if save_html:
        if html_output is None:
            html_output = Path("search_results.html")
        create_html_gallery(query_display, results, html_output)
    
    return results

def main():
    """
    Parse command line arguments and run the application.
    """
    parser = argparse.ArgumentParser(description="Local Image Search")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Index parser
    index_parser = subparsers.add_parser("index", help="Index images in a directory")
    index_parser.add_argument("directory", type=str, help="Directory containing images")
    index_parser.add_argument("--recursive", action="store_true", help="Search recursively")
    index_parser.add_argument("--model", type=str, default=CLIP_MODEL, help="CLIP model name")
    index_parser.add_argument("--device", type=str, help="Device to use (cuda or cpu)")
    index_parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    # Search parser
    search_parser = subparsers.add_parser("search", help="Search for images")
    search_parser.add_argument("query", type=str, help="Query text or image path")
    search_parser.add_argument("--image", action="store_true", help="Query is an image path")
    search_parser.add_argument("--model", type=str, default=CLIP_MODEL, help="CLIP model name")
    search_parser.add_argument("--device", type=str, help="Device to use (cuda or cpu)")
    search_parser.add_argument("--results", type=int, default=NUM_RESULTS, help="Number of results")
    search_parser.add_argument("--no-display", action="store_true", help="Do not display results")
    search_parser.add_argument("--save-html", action="store_true", help="Save results as HTML")
    search_parser.add_argument("--html-output", type=str, help="Path to save HTML gallery")
    search_parser.add_argument("--headless", action="store_true", help="Run in headless mode (no matplotlib)")
    search_parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if args.command == "index":
        index_images(
            directory=args.directory,
            recursive=args.recursive,
            model_name=args.model,
            device=args.device,
            verbose=not args.quiet
        )
    elif args.command == "search":
        search_images(
            query=args.query,
            is_image_query=args.image,
            model_name=args.model,
            device=args.device,
            num_results=args.results,
            display_results=not args.no_display,
            save_html=args.save_html,
            html_output=args.html_output,
            headless=args.headless,
            verbose=not args.quiet
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()