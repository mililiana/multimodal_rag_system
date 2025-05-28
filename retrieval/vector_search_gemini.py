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
import os

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Match

# Google Gemini imports  
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# here is my Gemini API Key 
GEMINI_API_KEY = "...." 

@dataclass
class RetrievedResult:
    """Structure for retrieved results"""
    id: str
    score: float
    content_type: str  # 'text' or 'image'
    title: str
    content: str
    image_url: Optional[str] = None
    image_alt: Optional[str] = None
    batch_title: str = ""
    metadata: Dict[str, Any] = None

@dataclass
class RAGResponse:
    """Structure for RAG system response"""
    answer: str
    retrieved_results: List[RetrievedResult]
    query: str
    timestamp: str
    total_results: int
    processing_time: float

class MultimodalRAGSystem:
    """
    Multimodal RAG System that queries Qdrant and generates answers using Google Gemini Flash
    """
    
    def __init__(self, 
                 qdrant_url: str,
                 qdrant_api_key: str,
                 gemini_api_key: str = None,  
                 text_collection: str = "text_embeddings",
                 image_collection: str = "image_embeddings",
                 text_model_name: str = "all-MiniLM-L6-v2",
                 clip_model_name: str = "ViT-B/32",
                 gemini_model_name: str = "gemini-1.5-flash"):
        """
        Initialize the RAG system with Google Gemini Flash
        
        Args:
            qdrant_url: Qdrant cluster URL
            qdrant_api_key: Qdrant API key
            gemini_api_key: Google Gemini API key (optional, uses hardcoded key if not provided)
            text_collection: Name of text embeddings collection
            image_collection: Name of image embeddings collection
            text_model_name: Sentence transformer model name
            clip_model_name: CLIP model name
            gemini_model_name: Gemini model name (gemini-1.5-flash for free tier)
        """
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # Collection names
        self.text_collection = text_collection
        self.image_collection = image_collection
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize embedding models
        logger.info("Loading text embedding model...")
        self.text_model = SentenceTransformer(text_model_name)
        
        logger.info("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        
        # Initialize Google Gemini
        logger.info(f"Initializing Google Gemini: {gemini_model_name}")
        try:
            api_key = gemini_api_key or GEMINI_API_KEY
            
            if api_key == "YOUR_GEMINI_API_KEY_HERE":
                logger.warning("Gemini API key not set! Please update GEMINI_API_KEY in the code")
                raise Exception("API key not configured")
            
            genai.configure(api_key=api_key)
            
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Initialize the model
            self.gemini_model = genai.GenerativeModel(
                model_name=gemini_model_name,
                safety_settings=self.safety_settings
            )
            
            # Test the connection
            test_response = self.gemini_model.generate_content("Test connection")
            logger.info("Gemini model initialized successfully")
            self.has_gemini = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            logger.info("Falling back to text-only generation...")
            self.gemini_model = None
            self.has_gemini = False
        
        logger.info("RAG System initialized successfully")
    
    def encode_text_query(self, query: str) -> np.ndarray:
        """Encode text query for similarity search"""
        return self.text_model.encode(query, convert_to_numpy=True)
    
    def encode_image_query(self, query: str) -> np.ndarray:
        """Encode text query for image search using CLIP"""
        text_input = clip.tokenize([query]).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        return text_features.cpu().numpy().flatten()
    
    def load_image_from_url(self, image_url: str) -> Optional[Image.Image]:
        """Load image from URL"""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image from {image_url}: {e}")
            return None
    
    def search_qdrant_text(self, 
                          query_vector: np.ndarray, 
                          limit: int = 10,
                          score_threshold: float = 0.0,
                          batch_filter: Optional[str] = None) -> List[RetrievedResult]:
        """Search text embeddings in Qdrant"""
        try:
            query_filter = None
            if batch_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="batch_id",
                            match=Match(value=batch_filter)
                        )
                    ]
                )
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.text_collection,
                query_vector=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
            
            # Convert to RetrievedResult objects
            results = []
            for result in search_results:
                payload = result.payload
                
                results.append(RetrievedResult(
                    id=payload.get('original_id', str(result.id)),
                    score=result.score,
                    content_type='text',
                    title=payload.get('article_title', 'No Title'),
                    content=payload.get('content', ''),
                    batch_title=payload.get('batch_title', ''),
                    metadata=payload
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching text in Qdrant: {e}")
            return []
    
    def search_qdrant_images(self, 
                            query_vector: np.ndarray, 
                            limit: int = 10,
                            score_threshold: float = 0.0,
                            batch_filter: Optional[str] = None) -> List[RetrievedResult]:
        """Search image embeddings in Qdrant"""
        try:
            # Prepare filter if needed
            query_filter = None
            if batch_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="batch_id",
                            match=Match(value=batch_filter)
                        )
                    ]
                )
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.image_collection,
                query_vector=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
            
            # Convert to RetrievedResult objects
            results = []
            for result in search_results:
                payload = result.payload
                
                results.append(RetrievedResult(
                    id=payload.get('original_id', str(result.id)),
                    score=result.score,
                    content_type='image',
                    title=payload.get('image_alt', 'Image'),
                    content=payload.get('image_caption', payload.get('image_alt', 'No description')),
                    image_url=payload.get('image_url'),
                    image_alt=payload.get('image_alt'),
                    batch_title=payload.get('batch_title', ''),
                    metadata=payload
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching images in Qdrant: {e}")
            return []
    
    def multimodal_search(self, 
                         query: str, 
                         limit: int = 10,
                         text_weight: float = 0.7,
                         image_weight: float = 0.3,
                         score_threshold: float = 0.1,
                         batch_filter: Optional[str] = None) -> List[RetrievedResult]:
        """Perform multimodal search across text and images"""
        # Encode query for both modalities
        text_query_vector = self.encode_text_query(query)
        image_query_vector = self.encode_image_query(query)
        
        # Search both modalities
        text_limit = max(1, int(limit * 1.5))  # Get more for reranking
        image_limit = max(1, int(limit * 1.5))
        
        text_results = self.search_qdrant_text(
            text_query_vector, text_limit, 0.0, batch_filter
        )
        image_results = self.search_qdrant_images(
            image_query_vector, image_limit, 0.0, batch_filter
        )
        
        # Apply weights and filter by threshold
        weighted_results = []
        
        for result in text_results:
            weighted_score = result.score * text_weight
            if weighted_score >= score_threshold:
                result.score = weighted_score
                weighted_results.append(result)
        
        for result in image_results:
            weighted_score = result.score * image_weight
            if weighted_score >= score_threshold:
                result.score = weighted_score
                weighted_results.append(result)
        
        # Sort by weighted score and return top results
        weighted_results.sort(key=lambda x: x.score, reverse=True)
        return weighted_results[:limit]
    
    def generate_answer_with_gemini(self, 
                                   query: str, 
                                   retrieved_results: List[RetrievedResult],
                                   max_tokens: int = 1000,
                                   temperature: float = 0.3) -> str:
        """
        I am generating answer using Google Gemini Flash based on retrieved context
        
        Args:
            query: Original user query
            retrieved_results: Retrieved results from search
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            
        Returns:
            Generated answer string
        """
        if not self.has_gemini:
            return self.generate_text_answer(query, retrieved_results)
            
        try:
            content_parts = []
            
            system_prompt = """You are a helpful AI assistant that answers questions based on provided context from The Batch newsletter about AI/ML news. 

Provide comprehensive, accurate answers based on the given context. If images are provided, analyze them and incorporate relevant visual information into your response.

Be concise but thorough, and cite specific sources when relevant."""
            
            content_parts.append(system_prompt)
            
            text_context_parts = []
            image_contexts = []
            
            for i, result in enumerate(retrieved_results[:5], 1):  # Limit to top 5 for context
                if result.content_type == 'text':
                    text_context_parts.append(
                        f"[Article {i}] {result.title}\n"
                        f"Content: {result.content[:500]}...\n"
                        f"Source: {result.batch_title}"
                    )
                else:
                    image = None
                    if result.image_url:
                        image = self.load_image_from_url(result.image_url)
                    
                    if image:
                        image_contexts.append({
                            'image': image,
                            'title': result.title,
                            'description': result.content,
                            'source': result.batch_title,
                            'index': i
                        })
                    else:
                        text_context_parts.append(
                            f"[Image {i}] {result.title}\n"
                            f"Description: {result.content}\n"
                            f"Source: {result.batch_title}"
                        )
            
            if text_context_parts:
                text_context = "\n---\n".join(text_context_parts)
                content_parts.append(f"\nText Context:\n{text_context}")
            
            for ctx in image_contexts:
                content_parts.append(f"\n[Image {ctx['index']}] {ctx['title']}")
                content_parts.append(f"Description: {ctx['description']}")
                content_parts.append(f"Source: {ctx['source']}")
                content_parts.append(ctx['image']) 
            
            content_parts.append(f"\nQuestion: {query}")
            content_parts.append("\nPlease provide a comprehensive answer based on the above context:")
            
            # Generate response with Gemini
            response = self.gemini_model.generate_content(
                content_parts,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {e}")
            logger.info("Falling back to text-only generation...")
            return self.generate_text_answer(query, retrieved_results)
    
    def generate_text_answer(self, 
                            query: str, 
                            retrieved_results: List[RetrievedResult]) -> str:
        """
        Simple text-based answer generation as fallback
        """
        # context from retrieved results
        context_parts = []
        
        for i, result in enumerate(retrieved_results[:5], 1):  # Limit to top 5 for context
            if result.content_type == 'text':
                context_parts.append(
                    f"[Article {i}] {result.title}\n"
                    f"Content: {result.content[:500]}...\n"
                    f"Source: {result.batch_title}\n"
                )
            else:
                context_parts.append(
                    f"[Image {i}] {result.title}\n"
                    f"Description: {result.content}\n"
                    f"Alt text: {result.image_alt or 'N/A'}\n"
                    f"Source: {result.batch_title}\n"
                )
        
        context = "\n---\n".join(context_parts)
        
        # Simple template-based response
        answer = f"""Based on the retrieved context from The Batch newsletter:

{context}

Regarding your question: "{query}"

The available information suggests that this topic is covered across multiple sources in the context above. Please refer to the specific articles and images for detailed information."""
        
        return answer
    
    def query(self, 
              query: str, 
              search_type: str = "multimodal",
              limit: int = 10,
              score_threshold: float = 0.1,
              generate_answer: bool = True,
              batch_filter: Optional[str] = None) -> RAGResponse:
        """
        Main query method that orchestrates the RAG pipeline with Google Gemini
        
        Args:
            query: User query
            search_type: 'text', 'image', or 'multimodal'
            limit: Number of results to retrieve
            score_threshold: Minimum similarity score
            generate_answer: Whether to generate LLM answer
            batch_filter: Optional batch ID filter
            
        Returns:
            RAGResponse object
        """
        start_time = datetime.now()
        
        if search_type == "text":
            query_vector = self.encode_text_query(query)
            retrieved_results = self.search_qdrant_text(
                query_vector, limit, score_threshold, batch_filter
            )
        elif search_type == "image":
            query_vector = self.encode_image_query(query)
            retrieved_results = self.search_qdrant_images(
                query_vector, limit, score_threshold, batch_filter
            )
        else:  # multimodal
            retrieved_results = self.multimodal_search(
                query, limit, score_threshold=score_threshold, batch_filter=batch_filter
            )
        
        # Generate answer if requested
        answer = ""
        if generate_answer and retrieved_results:
            answer = self.generate_answer_with_gemini(query, retrieved_results)
        elif not retrieved_results:
            answer = "I couldn't find any relevant information for your query."
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RAGResponse(
            answer=answer,
            retrieved_results=retrieved_results,
            query=query,
            timestamp=datetime.now().isoformat(),
            total_results=len(retrieved_results),
            processing_time=processing_time
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            text_info = self.qdrant_client.get_collection(self.text_collection)
            image_info = self.qdrant_client.get_collection(self.image_collection)
            
            return {
                'text_collection': {
                    'name': self.text_collection,
                    'points': text_info.points_count,
                    'vector_size': text_info.config.params.vectors.size
                },
                'image_collection': {
                    'name': self.image_collection,
                    'points': image_info.points_count,
                    'vector_size': image_info.config.params.vectors.size
                },
                'total_points': text_info.points_count + image_info.points_count,
                'device': self.device,
                'gemini_model': f'Google Gemini Flash ({self.has_gemini})'
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {'error': str(e)}

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal RAG Query System with Google Gemini Flash")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--search-type", choices=["text", "image", "multimodal"], 
                       default="multimodal", help="Type of search")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    parser.add_argument("--threshold", type=float, default=0.1, help="Score threshold")
    parser.add_argument("--no-answer", action="store_true", help="Skip LLM answer generation")
    parser.add_argument("--batch-filter", help="Filter by batch ID")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    
    args = parser.parse_args()

    # MY credentials
    args.qdrant_url = "https://....europe-west3-0.gcp.cloud.qdrant.io"
    args.qdrant_key = "....yR7PB8U3c1ctT4OGCGLQnqE7-SRYDeapPDmCrAOozek"
    
    rag_system = MultimodalRAGSystem(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_key
    )
    
    if args.stats:
        stats = rag_system.get_system_stats()
        print("\nSystem Statistics:")
        print(json.dumps(stats, indent=2))
        return
    
    print(f"\nQuery: {args.query}")
    print(f"Search Type: {args.search_type}")
    print(f"Processing with Google Gemini Flash...")
    
    response = rag_system.query(
        query=args.query,
        search_type=args.search_type,
        limit=args.limit,
        score_threshold=args.threshold,
        generate_answer=not args.no_answer,
        batch_filter=args.batch_filter
    )
    
    print(f"\n" + "="*50)
    print(f"QUERY RESULTS (Google Gemini Flash)")
    print(f"="*50)
    print(f"Processing Time: {response.processing_time:.2f}s")
    print(f"Total Results: {response.total_results}")
    
    if response.answer:
        print(f"\nANSWER:")
        print(f"{response.answer}")
    
    print(f"\nRETRIEVED RESULTS:")
    for i, result in enumerate(response.retrieved_results, 1):
        print(f"\n{i}. [{result.content_type.upper()}] Score: {result.score:.4f}")
        print(f"   Title: {result.title}")
        print(f"   Content: {result.content[:200]}...")
        if result.image_url:
            print(f"   Image URL: {result.image_url}")
        print(f"   Source: {result.batch_title}")

if __name__ == "__main__":
    main()
