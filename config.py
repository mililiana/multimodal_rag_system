# config.py
import os
from typing import Dict, Any

class RAGConfig:
    """Configuration class for Multimodal RAG System"""
    
    # Qdrant Configuration
    QDRANT_URL = "https://....europe-west3-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "..."
    
    # Collection names
    TEXT_COLLECTION = "text_embeddings"
    IMAGE_COLLECTION = "image_embeddings"
    
    # Model configurations
    TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
    CLIP_MODEL_NAME = "ViT-B/32"
    GEMINI_MODEL_NAME = "gemini-1.5-flash"
    
    # Gemini API Key (you can also set this via environment variable)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "...")
    
    # Default search parameters
    DEFAULT_LIMIT = 5
    DEFAULT_SCORE_THRESHOLD = 0.1
    DEFAULT_TEXT_WEIGHT = 0.7
    DEFAULT_IMAGE_WEIGHT = 0.3
    
    # UI Configuration
    PAGE_TITLE = "Multimodal RAG System - The Batch News"
    
    # Safety settings for Gemini
    SAFETY_SETTINGS = {
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
    }
    
    @classmethod
    def get_qdrant_config(cls) -> Dict[str, str]:
        """Get Qdrant configuration"""
        return {
            'url': cls.QDRANT_URL,
            'api_key': cls.QDRANT_API_KEY,
            'text_collection': cls.TEXT_COLLECTION,
            'image_collection': cls.IMAGE_COLLECTION
        }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, str]:
        """Get model configuration"""
        return {
            'text_model_name': cls.TEXT_MODEL_NAME,
            'clip_model_name': cls.CLIP_MODEL_NAME,
            'gemini_model_name': cls.GEMINI_MODEL_NAME,
            'gemini_api_key': cls.GEMINI_API_KEY
        }
    
    @classmethod
    def get_default_search_params(cls) -> Dict[str, Any]:
        """Get default search parameters"""
        return {
            'limit': cls.DEFAULT_LIMIT,
            'score_threshold': cls.DEFAULT_SCORE_THRESHOLD,
            'text_weight': cls.DEFAULT_TEXT_WEIGHT,
            'image_weight': cls.DEFAULT_IMAGE_WEIGHT
        }

# Sample queries for the UI
SAMPLE_QUERIES = [
    "computer vision breakthrough",
    "natural language processing advances",
    "AI ethics and responsible development",
    "machine learning in healthcare",
    "robotics automation",
    "deep learning research",
    "AI safety measures",
    "neural network architectures",
    "data science applications",
    "artificial intelligence trends"
]

# Help texts for UI elements
HELP_TEXTS = {
    'search_type': "Choose the type of search: multimodal (text+images), text only, or images only",
    'limit': "Maximum number of results to retrieve from the database",
    'score_threshold': "Minimum similarity score for results (higher = more relevant)",
    'generate_answer': "Use Google Gemini AI to generate a comprehensive answer based on retrieved results",
    'batch_filter': "Filter results by specific batch ID (leave empty to search all batches)",
    'query_input': "Enter your question or topic of interest. Be specific for better results."
}
