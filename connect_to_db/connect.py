import json
import logging
import pickle
import os
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from tqdm import tqdm


from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    CreateCollection, Filter, FieldCondition, 
    Match, Range
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantPKLUploader:
    """
    Upload embeddings from PKL files to Qdrant vector database
    """
    
    def __init__(self, 
                 qdrant_url: str,
                 qdrant_api_key: str,
                 text_collection: str = "text_embeddings",
                 image_collection: str = "image_embeddings"):
        """
        Initialize the Qdrant uploader
        
        Args:
            qdrant_url: Qdrant cluster URL
            qdrant_api_key: Qdrant API key
            text_collection: Name for text embeddings collection
            image_collection: Name for image embeddings collection
        """
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        self.text_collection = text_collection
        self.image_collection = image_collection
        
        logger.info(f"Connected to Qdrant at {qdrant_url}")
    
    def _generate_uuid_from_string(self, input_string: str) -> str:
        """
        Generate a deterministic UUID from a string using MD5 hash
        This ensures the same string always produces the same UUID
        """
        # Create MD5 hash of the input string
        hash_object = hashlib.md5(input_string.encode())
        hash_hex = hash_object.hexdigest()
        
        # Convert to UUID format (8-4-4-4-12)
        uuid_str = f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
        return uuid_str
    
    def load_pkl_embeddings(self, pkl_file_path: str) -> tuple:
        """
        Load embeddings and metadata from PKL file
        
        Args:
            pkl_file_path: Path to the PKL file
            
        Returns:
            Tuple of (embeddings_list, metadata_list)
        """
        try:
            with open(pkl_file_path, 'rb') as f:
                data = pickle.load(f)
            
            embeddings = data['embeddings']
            metadata = data['metadata']
            
            logger.info(f"Loaded {len(embeddings)} embeddings from {pkl_file_path}")
            logger.info(f"PKL file created at: {data.get('timestamp', 'Unknown')}")
            
            return embeddings, metadata
            
        except Exception as e:
            logger.error(f"Error loading PKL file {pkl_file_path}: {e}")
            raise
    
    def create_collections(self, text_dim: int, image_dim: int):
        """
        Create Qdrant collections for text and image embeddings
        
        Args:
            text_dim: Dimension of text embeddings
            image_dim: Dimension of image embeddings
        """
        try:
            # Create text collection
            self.client.create_collection(
                collection_name=self.text_collection,
                vectors_config=VectorParams(
                    size=text_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created text collection: {self.text_collection} (dim: {text_dim})")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Text collection {self.text_collection} already exists")
            else:
                logger.error(f"Error creating text collection: {e}")
                raise
        
        try:
            # Create image collection
            self.client.create_collection(
                collection_name=self.image_collection,
                vectors_config=VectorParams(
                    size=image_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created image collection: {self.image_collection} (dim: {image_dim})")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Image collection {self.image_collection} already exists")
            else:
                logger.error(f"Error creating image collection: {e}")
                raise
    
    def clear_collections(self):
        """Clear all data from collections (useful for re-upload)"""
        try:
            self.client.delete_collection(self.text_collection)
            logger.info(f"Deleted text collection: {self.text_collection}")
        except Exception as e:
            logger.warning(f"Could not delete text collection: {e}")
        
        try:
            self.client.delete_collection(self.image_collection)
            logger.info(f"Deleted image collection: {self.image_collection}")
        except Exception as e:
            logger.warning(f"Could not delete image collection: {e}")
    
    def upload_embeddings_from_pkl(self, 
                                  text_pkl_path: str,
                                  image_pkl_path: str,
                                  batch_size: int = 100,
                                  clear_existing: bool = False) -> Dict[str, int]:
        """
        Upload embeddings from PKL files to Qdrant
        
        Args:
            text_pkl_path: Path to text embeddings PKL file
            image_pkl_path: Path to image embeddings PKL file
            batch_size: Number of points to upload per batch
            clear_existing: Whether to clear existing collections first
            
        Returns:
            Dictionary with upload statistics
        """
        logger.info("Starting PKL to Qdrant upload...")
        
        # Load embeddings from PKL files
        logger.info("Loading text embeddings from PKL...")
        text_embeddings, text_metadata = self.load_pkl_embeddings(text_pkl_path)
        
        logger.info("Loading image embeddings from PKL...")
        image_embeddings, image_metadata = self.load_pkl_embeddings(image_pkl_path)
        
        # Get dimensions
        text_dim = len(text_embeddings[0]) if text_embeddings else 384
        image_dim = len(image_embeddings[0]) if image_embeddings else 512
        
        # Clear collections if requested
        if clear_existing:
            logger.info("Clearing existing collections...")
            self.clear_collections()
        
        # Create collections
        self.create_collections(text_dim, image_dim)
        
        # Convert to Qdrant points
        text_points = self._convert_to_points(text_embeddings, text_metadata, "text")
        image_points = self._convert_to_points(image_embeddings, image_metadata, "image")
        
        # Upload to Qdrant
        stats = self._upload_points_batched(text_points, image_points, batch_size)
        
        logger.info("Upload completed!")
        return stats
    
    def _convert_to_points(self, embeddings: List, metadata: List, content_type: str) -> List[PointStruct]:
        """Convert embeddings and metadata to Qdrant PointStruct objects"""
        points = []
        
        logger.info(f"Converting {len(embeddings)} {content_type} embeddings to Qdrant points...")
        
        for i, (embedding, meta) in enumerate(tqdm(zip(embeddings, metadata), desc=f"Converting {content_type}")):
            try:
                # Convert numpy array to list if needed
                if isinstance(embedding, np.ndarray):
                    vector = embedding.tolist()
                else:
                    vector = embedding
                
                # Create point ID - convert string ID to UUID
                original_id = meta.get('id', f"{content_type}_{i}")
                
                # Convert string ID to UUID (deterministic)
                if isinstance(original_id, str):
                    point_id = self._generate_uuid_from_string(original_id)
                else:
                    # If it's already an integer, use it directly
                    point_id = original_id
                
                # Add upload timestamp to payload and preserve original ID
                payload = dict(meta)  # Copy original metadata
                payload['uploaded_at'] = datetime.now().isoformat()
                payload['content_type'] = content_type
                payload['original_id'] = original_id  # Store the original string ID for reference
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))
                
            except Exception as e:
                logger.error(f"Error converting {content_type} point {i}: {e}")
                continue
        
        logger.info(f"Successfully converted {len(points)} {content_type} points")
        return points
    
    def _upload_points_batched(self, text_points: List[PointStruct], 
                              image_points: List[PointStruct], 
                              batch_size: int) -> Dict[str, int]:
        """Upload points to Qdrant in batches"""
        stats = {
            'text_uploaded': 0,
            'image_uploaded': 0,
            'text_failed': 0,
            'image_failed': 0,
            'total_text': len(text_points),
            'total_images': len(image_points)
        }
        
        # Upload text points
        if text_points:
            logger.info(f"Uploading {len(text_points)} text points to collection '{self.text_collection}'...")
            
            for i in tqdm(range(0, len(text_points), batch_size), desc="Uploading text"):
                batch = text_points[i:i + batch_size]
                try:
                    self.client.upsert(
                        collection_name=self.text_collection,
                        points=batch
                    )
                    stats['text_uploaded'] += len(batch)
                except Exception as e:
                    logger.error(f"Error uploading text batch {i//batch_size + 1}: {e}")
                    # Log the raw response if available
                    if hasattr(e, 'response'):
                        logger.error(f"Raw response: {e.response}")
                    stats['text_failed'] += len(batch)
        
        # Upload image points
        if image_points:
            logger.info(f"Uploading {len(image_points)} image points to collection '{self.image_collection}'...")
            
            for i in tqdm(range(0, len(image_points), batch_size), desc="Uploading images"):
                batch = image_points[i:i + batch_size]
                try:
                    self.client.upsert(
                        collection_name=self.image_collection,
                        points=batch
                    )
                    stats['image_uploaded'] += len(batch)
                except Exception as e:
                    logger.error(f"Error uploading image batch {i//batch_size + 1}: {e}")
                    # Log the raw response if available
                    if hasattr(e, 'response'):
                        logger.error(f"Raw response: {e.response}")
                    stats['image_failed'] += len(batch)
        
        return stats
    
    def get_collection_info(self):
        """Get information about the collections"""
        try:
            collections_info = {}
            
            # Try to get text collection info
            try:
                text_info = self.client.get_collection(self.text_collection)
                collections_info['text_collection'] = {
                    'name': self.text_collection,
                    'points_count': text_info.points_count,
                    'vector_size': text_info.config.params.vectors.size,
                    'distance': text_info.config.params.vectors.distance.value
                }
            except Exception as e:
                collections_info['text_collection'] = {'error': str(e)}
            
            # Try to get image collection info
            try:
                image_info = self.client.get_collection(self.image_collection)
                collections_info['image_collection'] = {
                    'name': self.image_collection,
                    'points_count': image_info.points_count,
                    'vector_size': image_info.config.params.vectors.size,
                    'distance': image_info.config.params.vectors.distance.value
                }
            except Exception as e:
                collections_info['image_collection'] = {'error': str(e)}
            
            return collections_info
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {'error': str(e)}
    
    def test_search(self, 
                   text_query: str = "artificial intelligence", 
                   image_query: str = "robot", 
                   limit: int = 3):
        """Test search functionality on uploaded data"""
        logger.info("Testing search functionality...")
        
        try:
            # Test text search (using a dummy vector - you'd need your text model for real queries)
            print(f"\n=== TEXT SEARCH TEST ===")
            print(f"Note: This is a random vector test. For real searches, you need your text embedding model.")
            
            try:
                # Get a sample vector dimension from the collection
                text_info = self.client.get_collection(self.text_collection)
                vector_size = text_info.config.params.vectors.size
                
                # Create a random vector for testing (in real use, you'd encode your query)
                import random
                random_vector = [random.random() for _ in range(vector_size)]
                
                # Use the older search method for compatibility
                text_results = self.client.search(
                    collection_name=self.text_collection,
                    query_vector=random_vector,
                    limit=limit
                )
                
                print(f"Found {len(text_results)} text results:")
                for i, result in enumerate(text_results, 1):
                    print(f"{i}. Score: {result.score:.4f}")
                    print(f"   ID: {result.id}")
                    print(f"   Original ID: {result.payload.get('original_id', 'N/A')}")
                    print(f"   Title: {result.payload.get('article_title', 'N/A')}")
                    print(f"   Batch: {result.payload.get('batch_title', 'N/A')}")
                    print()
            except Exception as e:
                print(f"Text search test failed: {e}")
            
            # Test image search
            print(f"\n=== IMAGE SEARCH TEST ===")
            print(f"Note: This is a random vector test. For real searches, you need your CLIP model.")
            
            try:
                # Get a sample vector dimension from the collection
                image_info = self.client.get_collection(self.image_collection)
                vector_size = image_info.config.params.vectors.size
                
                # Create a random vector for testing
                random_vector = [random.random() for _ in range(vector_size)]
                
                image_results = self.client.search(
                    collection_name=self.image_collection,
                    query_vector=random_vector,
                    limit=limit
                )
                
                print(f"Found {len(image_results)} image results:")
                for i, result in enumerate(image_results, 1):
                    print(f"{i}. Score: {result.score:.4f}")
                    print(f"   ID: {result.id}")
                    print(f"   Original ID: {result.payload.get('original_id', 'N/A')}")
                    print(f"   Alt: {result.payload.get('image_alt', 'N/A')}")
                    print(f"   URL: {result.payload.get('image_url', 'N/A')}")
                    print(f"   Batch: {result.payload.get('batch_title', 'N/A')}")
                    print()
            except Exception as e:
                print(f"Image search test failed: {e}")
                
        except Exception as e:
            logger.error(f"Error during test search: {e}")
    
    def validate_upload(self) -> Dict[str, Any]:
        """Validate that the upload was successful"""
        validation_results = {}
        
        try:
            # Get collection info
            info = self.get_collection_info()
            validation_results['collections'] = info
            
            # Check if collections have data
            text_count = info.get('text_collection', {}).get('points_count', 0)
            image_count = info.get('image_collection', {}).get('points_count', 0)
            
            validation_results['success'] = text_count > 0 or image_count > 0
            validation_results['text_points'] = text_count
            validation_results['image_points'] = image_count
            validation_results['total_points'] = text_count + image_count
            
            if validation_results['success']:
                logger.info(f" Upload validation successful!")
                logger.info(f"   Text points: {text_count}")
                logger.info(f"   Image points: {image_count}")
                logger.info(f"   Total points: {text_count + image_count}")
            else:
                logger.warning("⚠️  Upload validation failed - no data found in collections")
            
        except Exception as e:
            validation_results['error'] = str(e)
            validation_results['success'] = False
            logger.error(f" Upload validation error: {e}")
        
        return validation_results

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload embeddings from PKL files to Qdrant")
    parser.add_argument("--qdrant-url", required=True, help="Qdrant cluster URL")
    parser.add_argument("--qdrant-key", required=True, help="Qdrant API key")
    parser.add_argument("--text-pkl", required=True, help="Path to text embeddings PKL file")
    parser.add_argument("--image-pkl", required=True, help="Path to image embeddings PKL file")
    parser.add_argument("--text-collection", default="text_embeddings", help="Text collection name")
    parser.add_argument("--image-collection", default="image_embeddings", help="Image collection name")
    parser.add_argument("--batch-size", type=int, default=100, help="Upload batch size")
    parser.add_argument("--clear", action="store_true", help="Clear existing collections before upload")
    parser.add_argument("--info-only", action="store_true", help="Only show collection info")
    parser.add_argument("--test", action="store_true", help="Run test searches after upload")
    parser.add_argument("--validate", action="store_true", help="Validate upload")
    
    args = parser.parse_args([
    "--qdrant-url", "https://2829a7bb-1713-4e6c-b1b1-d8ffbf38a124.europe-west3-0.gcp.cloud.qdrant.io",
    "--qdrant-key", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.yR7PB8U3c1ctT4OGCGLQnqE7-SRYDeapPDmCrAOozek",
    "--text-pkl", "embedding_cache/text_embeddings.pkl",
    "--image-pkl", "embedding_cache/image_embeddings.pkl",
    "--clear",
    "--validate",
    "--test"
])

    
    # Validate PKL files exist
    if not args.info_only:
        if not os.path.exists(args.text_pkl):
            logger.error(f"Text PKL file not found: {args.text_pkl}")
            return
        
        if not os.path.exists(args.image_pkl):
            logger.error(f"Image PKL file not found: {args.image_pkl}")
            return
    
    # Initialize uploader
    uploader = QdrantPKLUploader(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_key,
        text_collection=args.text_collection,
        image_collection=args.image_collection
    )
    
    # Show collection info only
    if args.info_only:
        info = uploader.get_collection_info()
        print("\n=== COLLECTION INFORMATION ===")
        print(json.dumps(info, indent=2))
        return
    
    try:
        stats = uploader.upload_embeddings_from_pkl(
            text_pkl_path=args.text_pkl,
            image_pkl_path=args.image_pkl,
            batch_size=args.batch_size,
            clear_existing=args.clear
        )
        
        print("\n=== UPLOAD STATISTICS ===")
        print(f"Text embeddings:")
        print(f"  Total: {stats['total_text']}")
        print(f"  Uploaded: {stats['text_uploaded']}")
        print(f"  Failed: {stats['text_failed']}")
        
        print(f"\nImage embeddings:")
        print(f"  Total: {stats['total_images']}")
        print(f"  Uploaded: {stats['image_uploaded']}")
        print(f"  Failed: {stats['image_failed']}")
        
        print(f"\nOverall:")
        print(f"  Total uploaded: {stats['text_uploaded'] + stats['image_uploaded']}")
        print(f"  Total failed: {stats['text_failed'] + stats['image_failed']}")
        
        # Validate upload
        if args.validate:
            print("\n=== VALIDATION ===")
            validation = uploader.validate_upload()
            if validation.get('success'):
                print("  Upload successful!")
            else:
                print("  Upload validation failed")
        
        if args.test:
            uploader.test_search()
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

if __name__ == "__main__":
    main()