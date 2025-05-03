# visualizer.py
"""
Utility functions for visualizing search results.
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Union, Optional

def display_image(image_path: Union[str, Path]) -> None:
    """
    Display an image.
    
    Args:
        image_path: Path to the image file
    """
    try:
        print(f"Image: {image_path}")
        # Skip actual display which requires matplotlib
    except Exception as e:
        print(f"Error displaying image {image_path}: {e}")

def display_query_results(
    query: str,
    results: List[Dict],
    max_results: int = 5,
    cols: int = 3
) -> None:
    """
    Display the query and search results.
    
    Args:
        query: Query text or image path
        results: List of result dictionaries
        max_results: Maximum number of results to display
        cols: Number of columns in the grid
    """
    # Limit the number of results
    results = results[:max_results]
    
    print("\n===== Search Results =====")
    print(f"Query: {query}")
    print("==========================")
    
    for i, result in enumerate(results):
        print(f"\n[{i+1}] Similarity: {result['similarity']:.4f}")
        print(f"    Path: {result['image_path']}")
        print(f"    File: {result['file_name']}")

def create_html_gallery(
    query: str,
    results: List[Dict],
    output_path: Union[str, Path],
    max_results: int = 20
) -> None:
    """
    Create an HTML gallery of search results.
    
    Args:
        query: Query text or image path
        results: List of result dictionaries
        output_path: Path to save the HTML file
        max_results: Maximum number of results to include
    """
    if isinstance(output_path, str):
        output_path = Path(output_path)
        
    # Limit the number of results
    results = results[:max_results]
    
    # Create the HTML content
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Search Results for: {query}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            grid-gap: 20px;
            margin-top: 20px;
        }}
        .item {{
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            transition: transform 0.3s ease;
        }}
        .item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }}
        .item img {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            display: block;
        }}
        .item .info {{
            padding: 10px;
        }}
        .item .similarity {{
            font-weight: bold;
            color: #0066cc;
        }}
        .item .path {{
            color: #666;
            font-size: 0.8em;
            word-break: break-all;
        }}
    </style>
</head>
<body>
    <h1>Search Results for: {query}</h1>
    <div class="gallery">
"""
    
    # Add the results
    for result in results:
        html += f"""
        <div class="item">
            <img src="file://{result['image_path']}" alt="{result['file_name']}">
            <div class="info">
                <div class="similarity">Similarity: {result['similarity']:.3f}</div>
                <div class="path">{result['image_path']}</div>
            </div>
        </div>
"""
    
    # Close the HTML
    html += """
    </div>
</body>
</html>
"""
    
    # Write the HTML to the file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
        
    print(f"HTML gallery saved to {output_path}")
    
    # Try to open the HTML file in the default browser
    try:
        import webbrowser
        webbrowser.open(f"file://{output_path.absolute()}")
        print("Opening gallery in web browser...")
    except Exception as e:
        print(f"Note: Could not automatically open gallery: {e}")
        print(f"You can manually open it at: file://{output_path.absolute()}")