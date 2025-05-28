import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import clip
import torch
from PIL import Image
import base64
from io import BytesIO
import requests
from dataclasses import dataclass
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    id: str
    score: float
    payload: Dict[str, Any]
    content_type: str  # 'text' or 'image'
    embedding: Optional[np.ndarray] = None

class InMemoryMultimodalSearch:
    """
    Here I use In-memory multimodal search system for text and image embeddings as it
    works directly with JSON files without database dependencies
    """
    
    def __init__(self, 
                 text_model_name: str = "all-MiniLM-L6-v2",
                 clip_model_name: str = "ViT-B/32",
                 cache_embeddings: bool = True):
        """
        Initialize the in-memory multimodal search system
        
        Args:
            text_model_name: Sentence transformer model for text embeddings
            clip_model_name: CLIP model for image embeddings
            cache_embeddings: Whether to cache embeddings to disk
        """

        logger.info("Loading text embedding model...")
        self.text_model = SentenceTransformer(text_model_name)
        
        logger.info("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        
        # Storage for embeddings and metadata
        self.text_embeddings = []
        self.text_metadata = []
        self.image_embeddings = []
        self.image_metadata = []
        
        # Caching
        self.cache_embeddings = cache_embeddings
        self.cache_dir = "embedding_cache"
        if cache_embeddings and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Vector dimensions
        self.text_dim = self.text_model.get_sentence_embedding_dimension()
        self.image_dim = 512  # CLIP ViT-B/32 output dimension
        
        logger.info(f"Text embedding dimension: {self.text_dim}")
        logger.info(f"Image embedding dimension: {self.image_dim}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using sentence transformer"""
        return self.text_model.encode(text, convert_to_numpy=True)
    
    def encode_image_from_base64(self, base64_data: str) -> Optional[np.ndarray]:
        """Encode image from base64 data using CLIP"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            
            # Preprocess and encode
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None
    
    def encode_image_from_url(self, url: str) -> Optional[np.ndarray]:
        """Encode image from URL using CLIP"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error encoding image from URL {url}: {e}")
            return None
    
    def load_cached_embeddings(self, cache_file: str) -> Optional[Tuple[List, List]]:
        """Load cached embeddings from disk"""
        if not self.cache_embeddings or not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                return data['embeddings'], data['metadata']
        except Exception as e:
            logger.warning(f"Could not load cache {cache_file}: {e}")
            return None
    
    def save_cached_embeddings(self, cache_file: str, embeddings: List, metadata: List):
        """Save embeddings to disk cache"""
        if not self.cache_embeddings:
            return
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'metadata': metadata,
                    'timestamp': datetime.now().isoformat()
                }, f)
            logger.info(f"Saved embeddings cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not save cache {cache_file}: {e}")
    
    def process_batch_data(self, batch_file_path: str, images_file_path: Optional[str] = None) -> None:
        """
        Process the batch_issues.json file and create embeddings in memory
        
        Args:
            batch_file_path: Path to the batch_issues.json file
            images_file_path: Optional path to separate images JSON file
        """
        logger.info(f"Processing batch data from {batch_file_path}")
        
        # Check for cached embeddings
        text_cache_file = os.path.join(self.cache_dir, "text_embeddings.pkl")
        image_cache_file = os.path.join(self.cache_dir, "image_embeddings.pkl")
        
        cached_text = self.load_cached_embeddings(text_cache_file)
        cached_images = self.load_cached_embeddings(image_cache_file)
        
        if cached_text and cached_images:
            logger.info("Loading embeddings from cache...")
            self.text_embeddings, self.text_metadata = cached_text
            self.image_embeddings, self.image_metadata = cached_images
            logger.info(f"Loaded {len(self.text_embeddings)} text and {len(self.image_embeddings)} image embeddings from cache")
            return
        
        try:
            with open(batch_file_path, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            # Load separate images file if provided
            images_data = {}
            if images_file_path and os.path.exists(images_file_path):
                with open(images_file_path, 'r', encoding='utf-8') as f:
                    images_data = json.load(f)
                logger.info(f"Loaded images data from {images_file_path}")
            
            text_embeddings = []
            text_metadata = []
            image_embeddings = []
            image_metadata = []
            
            for batch_issue in batch_data:
                batch_id = batch_issue['id']
                batch_title = batch_issue['title']
                
                # Process articles (text content)
                for i, article in enumerate(batch_issue.get('articles', [])):
                    article_text = f"{article['title']} {article['content']}"
                    text_embedding = self.encode_text(article_text)
                    
                    if text_embedding is not None:
                        point_id = f"{batch_id}_article_{i}"
                        
                        text_embeddings.append(text_embedding)
                        text_metadata.append({
                            'id': point_id,
                            'batch_id': batch_id,
                            'batch_title': batch_title,
                            'article_title': article['title'],
                            'content': article['content'],
                            'content_type': 'article',
                            'url': batch_issue.get('url'),
                            'published_date': batch_issue.get('published_date'),
                            'scraped_at': batch_issue.get('scraped_at'),
                            'word_count': batch_issue.get('word_count', 0)
                        })
                
                # Process images from batch_issues.json
                for i, image_data in enumerate(batch_issue.get('images', [])):
                    self._process_single_image(
                        image_data, batch_id, batch_title, batch_issue,
                        i, image_embeddings, image_metadata
                    )
                
                # Process images from separate images file
                if batch_id in images_data:
                    for i, image_data in enumerate(images_data[batch_id]):
                        self._process_single_image(
                            image_data, batch_id, batch_title, batch_issue,
                            i + 1000,  # Offset to avoid ID conflicts
                            image_embeddings, image_metadata
                        )
            
            # Store in memory
            self.text_embeddings = text_embeddings
            self.text_metadata = text_metadata
            self.image_embeddings = image_embeddings
            self.image_metadata = image_metadata
            
            # Cache embeddings
            self.save_cached_embeddings(text_cache_file, text_embeddings, text_metadata)
            self.save_cached_embeddings(image_cache_file, image_embeddings, image_metadata)
            
            logger.info(f"Processing complete. Created {len(text_embeddings)} text and {len(image_embeddings)} image embeddings")
            
        except Exception as e:
            logger.error(f"Error processing batch data: {e}")
            raise
    
    def _process_single_image(self, image_data: Dict, batch_id: str, batch_title: str, 
                            batch_issue: Dict, index: int, 
                            image_embeddings: List, image_metadata: List):
        """Process a single image and add to collections"""
        image_embedding = None
        
        # Try to encode from base64 data first
        if image_data.get('data'):
            image_embedding = self.encode_image_from_base64(image_data['data'])
        
        # Fallback to URL if base64 fails
        if image_embedding is None and image_data.get('url'):
            image_embedding = self.encode_image_from_url(image_data['url'])
        
        if image_embedding is not None:
            point_id = f"{batch_id}_image_{index}"
            
            image_embeddings.append(image_embedding)
            image_metadata.append({
                'id': point_id,
                'batch_id': batch_id,
                'batch_title': batch_title,
                'image_url': image_data.get('url'),
                'image_alt': image_data.get('alt'),
                'image_caption': image_data.get('caption'),
                'content_type': 'image',
                'url': batch_issue.get('url'),
                'published_date': batch_issue.get('published_date'),
                'scraped_at': batch_issue.get('scraped_at')
            })
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search_text(self, 
                    query: str, 
                    limit: int = 10,
                    batch_id_filter: Optional[str] = None,
                    min_score: float = 0.0) -> List[SearchResult]:
        """
        Search for similar text content
        
        Args:
            query: Search query
            limit: Number of results to return
            batch_id_filter: Optional filter by batch ID
            min_score: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects
        """
        if not self.text_embeddings:
            logger.warning("No text embeddings available. Run process_batch_data first.")
            return []
        
        try:
            query_embedding = self.encode_text(query)
            results = []
            
            for i, (embedding, metadata) in enumerate(zip(self.text_embeddings, self.text_metadata)):
                # Apply batch filter if specified
                if batch_id_filter and metadata['batch_id'] != batch_id_filter:
                    continue
                
                score = self.cosine_similarity(query_embedding, embedding)
                
                if score >= min_score:
                    results.append(SearchResult(
                        id=metadata['id'],
                        score=score,
                        payload=metadata,
                        content_type='text',
                        embedding=embedding
                    ))
            
            # Sort by score and return top results
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching text: {e}")
            return []
    
    def search_images(self, 
                      query: str, 
                      limit: int = 10,
                      batch_id_filter: Optional[str] = None,
                      min_score: float = 0.0) -> List[SearchResult]:
        """
        Search for images using text query (CLIP text-to-image search)
        
        Args:
            query: Text query to search for similar images
            limit: Number of results to return
            batch_id_filter: Optional filter by batch ID
            min_score: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects
        """
        if not self.image_embeddings:
            logger.warning("No image embeddings available. Run process_batch_data first.")
            return []
        
        try:
            # Encode text query using CLIP
            text_input = clip.tokenize([query]).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_input)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            query_embedding = text_features.cpu().numpy().flatten()
            results = []
            
            for i, (embedding, metadata) in enumerate(zip(self.image_embeddings, self.image_metadata)):
                # Apply batch filter if specified
                if batch_id_filter and metadata['batch_id'] != batch_id_filter:
                    continue
                
                score = self.cosine_similarity(query_embedding, embedding)
                
                if score >= min_score:
                    results.append(SearchResult(
                        id=metadata['id'],
                        score=score,
                        payload=metadata,
                        content_type='image',
                        embedding=embedding
                    ))
            
            # Sort by score and return top results
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching images: {e}")
            return []
    
    def multimodal_search(self, 
                         query: str, 
                         limit: int = 10,
                         text_weight: float = 0.7,
                         image_weight: float = 0.3,
                         batch_id_filter: Optional[str] = None,
                         min_score: float = 0.0) -> List[SearchResult]:
        """
        Combined multimodal search across text and images
        
        Args:
            query: Search query
            limit: Total number of results to return
            text_weight: Weight for text results (0-1)
            image_weight: Weight for image results (0-1)
            batch_id_filter: Optional filter by batch ID
            min_score: Minimum weighted score threshold
            
        Returns:
            Combined and ranked list of SearchResult objects
        """
        text_limit = max(1, int(limit * 2))  # Get more to allow for reranking
        image_limit = max(1, int(limit * 2))
        
        text_results = self.search_text(query, text_limit, batch_id_filter, 0.0)
        image_results = self.search_images(query, image_limit, batch_id_filter, 0.0)
        
        # Adjust scores by weights
        for result in text_results:
            result.score *= text_weight
        
        for result in image_results:
            result.score *= image_weight
        
        # Combine and filter by minimum score
        all_results = []
        for result in text_results + image_results:
            if result.score >= min_score:
                all_results.append(result)
        
        # Sort by weighted score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:limit]
    
    def get_similar_items(self, 
                         item_id: str, 
                         limit: int = 10,
                         content_type: Optional[str] = None) -> List[SearchResult]:
        """
        Find items similar to a given item by ID
        
        Args:
            item_id: ID of the reference item
            limit: Number of similar items to return
            content_type: 'text' or 'image' to search within same type only
            
        Returns:
            List of similar SearchResult objects
        """
        # Find the reference item
        ref_embedding = None
        ref_metadata = None
        ref_type = None
        
        # Search in text embeddings
        for embedding, metadata in zip(self.text_embeddings, self.text_metadata):
            if metadata['id'] == item_id:
                ref_embedding = embedding
                ref_metadata = metadata
                ref_type = 'text'
                break
        
        # Search in image embeddings if not found
        if ref_embedding is None:
            for embedding, metadata in zip(self.image_embeddings, self.image_metadata):
                if metadata['id'] == item_id:
                    ref_embedding = embedding
                    ref_metadata = metadata
                    ref_type = 'image'
                    break
        
        if ref_embedding is None:
            logger.warning(f"Item {item_id} not found")
            return []
        
        results = []
        
        # Search in appropriate collections
        collections_to_search = []
        if content_type is None or content_type == 'text':
            if ref_type == 'text' or content_type is None:
                collections_to_search.append(('text', self.text_embeddings, self.text_metadata))
        
        if content_type is None or content_type == 'image':
            if ref_type == 'image' or content_type is None:
                collections_to_search.append(('image', self.image_embeddings, self.image_metadata))
        
        for coll_type, embeddings, metadata_list in collections_to_search:
            for embedding, metadata in zip(embeddings, metadata_list):
                if metadata['id'] == item_id:  # Skip the reference item itself
                    continue
                
                score = self.cosine_similarity(ref_embedding, embedding)
                results.append(SearchResult(
                    id=metadata['id'],
                    score=score,
                    payload=metadata,
                    content_type=coll_type,
                    embedding=embedding
                ))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded data"""
        return {
            'text_embeddings': len(self.text_embeddings),
            'image_embeddings': len(self.image_embeddings),
            'total_embeddings': len(self.text_embeddings) + len(self.image_embeddings),
            'text_dimension': self.text_dim,
            'image_dimension': self.image_dim,
            'device': self.device,
            'cache_enabled': self.cache_embeddings
        }
    
    def export_embeddings(self, output_file: str): # Export all embeddings and metadata to a file
        data = {
            'text_embeddings': [emb.tolist() for emb in self.text_embeddings],
            'text_metadata': self.text_metadata,
            'image_embeddings': [emb.tolist() for emb in self.image_embeddings],
            'image_metadata': self.image_metadata,
            'stats': self.get_stats(),
            'exported_at': datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported embeddings to {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="In-Memory Multimodal Search System")
    parser.add_argument("--batch-file", required=True, help="Path to batch_issues.json file")
    parser.add_argument("--images-file", help="Path to separate images JSON file")
    parser.add_argument("--search", help="Search query")
    parser.add_argument("--search-type", choices=["text", "image", "multimodal"], 
                       default="multimodal", help="Type of search")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum similarity score")
    parser.add_argument("--export", help="Export embeddings to JSON file")
    parser.add_argument("--similar", help="Find items similar to given ID")
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding cache")
    
    args = parser.parse_args()
    
    # Initialize search system
    search_system = InMemoryMultimodalSearch(cache_embeddings=not args.no_cache)
    
    print("Processing data and creating embeddings...")
    search_system.process_batch_data(args.batch_file, args.images_file)
    
    stats = search_system.get_stats()
    print("\nSystem Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if args.export:
        search_system.export_embeddings(args.export)
        print(f"Embeddings exported to {args.export}")
    
    if args.similar:
        print(f"\nFinding items similar to: {args.similar}")
        results = search_system.get_similar_items(args.similar, args.limit)
        
        print(f"Found {len(results)} similar items:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result.content_type.upper()}] Score: {result.score:.4f}")
            print(f"   ID: {result.id}")
            
            if result.content_type == 'text':
                print(f"   Title: {result.payload.get('article_title', 'N/A')}")
                print(f"   Content: {result.payload.get('content', '')[:150]}...")
            else:
                print(f"   Alt: {result.payload.get('image_alt', 'N/A')}")
                print(f"   Caption: {result.payload.get('image_caption', 'N/A')}")
    
    if args.search:
        print(f"\nSearching for: '{args.search}'")
        print(f"Search type: {args.search_type}")
        print(f"Minimum score: {args.min_score}")
        
        if args.search_type == "text":
            results = search_system.search_text(args.search, args.limit, min_score=args.min_score)
        elif args.search_type == "image":
            results = search_system.search_images(args.search, args.limit, min_score=args.min_score)
        else:
            results = search_system.multimodal_search(args.search, args.limit, min_score=args.min_score)
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result.content_type.upper()}] Score: {result.score:.4f}")
            print(f"   ID: {result.id}")
            
            if result.content_type == 'text':
                print(f"   Title: {result.payload.get('article_title', 'N/A')}")
                print(f"   Content: {result.payload.get('content', '')[:200]}...")
            else:
                print(f"   Alt: {result.payload.get('image_alt', 'N/A')}")
                print(f"   Caption: {result.payload.get('image_caption', 'N/A')}")
                print(f"   URL: {result.payload.get('image_url', 'N/A')}")

if __name__ == "__main__":
    main()