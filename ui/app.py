import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import json
import time
from datetime import datetime
import base64
import logging

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'retrieval')))

from vector_search_gemini import MultimodalRAGSystem, RetrievedResult, RAGResponse

st.set_page_config(
    page_title="Multimodal RAG System - The Batch News",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(css_file):
    try:
        with open(css_file, 'r') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{css_file}' not found. Please ensure it exists in the same directory.")
    except Exception as e:
        st.error(f"Error loading CSS: {str(e)}")

load_css('ui/styles.css')

if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'system_stats' not in st.session_state:
    st.session_state.system_stats = None
if 'initialization_error' not in st.session_state:
    st.session_state.initialization_error = None

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with caching to avoid reinitialization"""
    try:
        # MY Qdrant credentials
        qdrant_url = "https://2829a7bb-1713-4e6c-b1b1-d8ffbf38a124.europe-west3-0.gcp.cloud.qdrant.io"
        qdrant_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.yR7PB8U3c1ctT4OGCGLQnqE7-SRYDeapPDmCrAOozek"

        
        rag_system = MultimodalRAGSystem(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_key,
            text_collection="text_embeddings",
            image_collection="image_embeddings"
        )
        
        stats = rag_system.get_system_stats()
        
        return rag_system, stats, None
        
    except Exception as e:
        error_msg = f"Failed to initialize RAG system: {str(e)}"
        logging.error(error_msg)
        return None, None, error_msg

def load_image_safely(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def display_retrieved_result(result: RetrievedResult, index: int):
    
    if result.content_type == 'text':
        with st.container():
            st.markdown(f'<div class="article-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f'<span class="content-type-badge text-badge">TEXT</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="article-title">{result.title}</div>', unsafe_allow_html=True)
            with col2:
                score_percentage = int(result.score * 100)
                st.markdown(f'<div class="relevance-score">Score: {score_percentage}%</div>', unsafe_allow_html=True)
            
            metadata_text = f"ID: {result.id}"
            if result.batch_title:
                metadata_text += f" | Source: {result.batch_title}"
            st.markdown(f'<div class="article-meta">{metadata_text}</div>', unsafe_allow_html=True)
            
            # Content
            st.write(result.content)
            
            if result.metadata:
                with st.expander("Additional Details"):
                    st.json(result.metadata)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        with st.container():
            st.markdown(f'<div class="image-card">', unsafe_allow_html=True)
            
            # Header with title, type badge, and relevance score
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f'<span class="content-type-badge image-badge">IMAGE</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="image-title">{result.title}</div>', unsafe_allow_html=True)
            with col2:
                score_percentage = int(result.score * 100)
                st.markdown(f'<div class="relevance-score">Score: {score_percentage}%</div>', unsafe_allow_html=True)
            
            # Metadata
            metadata_text = f"ID: {result.id}"
            if result.batch_title:
                metadata_text += f" | Source: {result.batch_title}"
            st.markdown(f'<div class="article-meta">{metadata_text}</div>', unsafe_allow_html=True)
            
            # Image and description in columns
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if result.image_url:
                    try:
                        image = load_image_safely(result.image_url)
                        if image:
                            st.image(image, caption=result.image_alt or "Retrieved Image", use_column_width=True)
                        else:
                            st.error("Failed to load image")
                            st.text(f"URL: {result.image_url}")
                    except Exception as e:
                        st.error(f"Image error: {e}")
                else:
                    st.info("No image URL available")
            
            with col2:
                st.write("**Description:**")
                st.write(result.content)
                
                if result.image_alt and result.image_alt != result.content:
                    st.write("**Alt Text:**")
                    st.write(result.image_alt)
                
                # Image URL for reference
                if result.image_url:
                    st.write("**Image URL:**")
                    st.code(result.image_url, language=None)
            
            # Additional metadata if available
            if result.metadata:
                with st.expander(" Additional Details"):
                    st.json(result.metadata)
            
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header"> Multimodal RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Intelligent News Retrieval from The Batch with Gemini AI</p>', unsafe_allow_html=True)
    
    # Initialize RAG system
    if st.session_state.rag_system is None:
        with st.spinner(" Initializing RAG System (this may take a moment)..."):
            rag_system, stats, error = initialize_rag_system()
            
            if error:
                st.error(f" {error}")
                st.stop()
            else:
                st.session_state.rag_system = rag_system
                st.session_state.system_stats = stats
                st.success(" RAG System initialized successfully!")
    
    with st.sidebar:
        st.header("Configuration")
        
        search_type = st.selectbox(
            "Search Type",
            ["multimodal", "text", "image"],
            help="Choose the type of search to perform"
        )
        
        limit = st.slider("Number of results to retrieve", min_value=1, max_value=20, value=5)
        score_threshold = st.slider("Relevance threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        
        generate_answer = st.checkbox("Generate AI Answer", value=True, help="Use Gemini to generate an answer based on retrieved results")
        
        batch_filter = st.text_input("Batch Filter (optional)", help="Filter results by specific batch ID")
        if not batch_filter.strip():
            batch_filter = None
        
        st.divider()
        
        st.header("System Stats")
        if st.session_state.system_stats:
            stats = st.session_state.system_stats
            
            if 'error' not in stats:
                st.metric("Text Documents", stats.get('text_collection', {}).get('points', 0))
                st.metric("Images", stats.get('image_collection', {}).get('points', 0))
                st.metric("Total Items", stats.get('total_points', 0))
                
                st.write(f"**Device:** {stats.get('device', 'Unknown')}")
                st.write(f"**Model:** {stats.get('gemini_model', 'Unknown')}")
            else:
                st.error("Could not load system stats")
        
        st.metric("Searches Performed", len(st.session_state.search_history))
        
        st.divider()
        
        st.header("Search History")
        if st.session_state.search_history:
            for i, (query, timestamp, result_count) in enumerate(reversed(st.session_state.search_history[-5:])):
                query_display = query[:25] + "..." if len(query) > 25 else query
                if st.button(f"{query_display} ({result_count})", key=f"history_{i}"):
                    st.session_state.current_query = query
                    st.rerun()
        else:
            st.write("No search history yet")
        
        if st.session_state.search_history and st.button("Clear History"):
            st.session_state.search_history = []
            st.rerun()
    
    st.markdown('<div class="query-box">', unsafe_allow_html=True)
    
    # Query input
    st.subheader(" Enter your query")
    
    # Use session state to maintain query
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    query = st.text_input(
        "What would you like to know?",
        value=st.session_state.current_query,
        placeholder="e.g., 'Latest developments in computer vision' or 'AI ethics guidelines'",
        help="Enter your question or topic of interest"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_button = st.button("Search", type="primary")
    with col2:
        clear_button = st.button(" Clear")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle clear button
    if clear_button:
        st.session_state.current_query = ""
        st.rerun()
    
    # Handle search
    if search_button and query.strip() and st.session_state.rag_system:
        st.session_state.current_query = query
        
        # Perform search
        start_time = time.time()
        
        with st.spinner(f" Searching ({search_type}) for relevant content..."):
            try:
                response: RAGResponse = st.session_state.rag_system.query(
                    query=query,
                    search_type=search_type,
                    limit=limit,
                    score_threshold=score_threshold,
                    generate_answer=generate_answer,
                    batch_filter=batch_filter
                )
                
                # Add to search history
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.search_history.append((query, timestamp, response.total_results))
                
                # Display results
                if response.total_results > 0:
                    st.success(f"Found {response.total_results} relevant results in {response.processing_time:.2f}s")
                    
                    # Results summary
                    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Results Found", response.total_results)
                    with col2:
                        if response.retrieved_results:
                            avg_score = np.mean([r.score for r in response.retrieved_results])
                            st.metric("Avg Score", f"{avg_score:.3f}")
                        else:
                            st.metric("Avg Score", "N/A")
                    with col3:
                        text_count = sum(1 for r in response.retrieved_results if r.content_type == 'text')
                        st.metric("Text Results", text_count)
                    with col4:
                        image_count = sum(1 for r in response.retrieved_results if r.content_type == 'image')
                        st.metric("Image Results", image_count)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display AI-generated answer if available
                    if response.answer and response.answer.strip():
                        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                        st.markdown('<div class="answer-title">AI-Generated Answer</div>', unsafe_allow_html=True)
                        st.markdown(response.answer)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Display retrieved results
                    st.subheader(" Retrieved Results")
                    
                    for i, result in enumerate(response.retrieved_results):
                        display_retrieved_result(result, i)
                        
                else:
                    st.warning("No relevant content found. Try adjusting your query or lowering the relevance threshold.")
                    
            except Exception as e:
                st.error(f" Error during search: {str(e)}")
                st.exception(e)
        
    elif search_button and not query.strip():
        st.error("Please enter a search query.")
    elif search_button and not st.session_state.rag_system:
        st.error("RAG system not initialized. Please refresh the page.")
    
    st.divider()
    st.subheader(" Try these sample queries:")
    
    sample_queries = [
        "computer vision breakthrough",
        "natural language processing",
        "AI ethics guidelines",
        "machine learning models",
        "robotics applications"
    ]
    
    cols = st.columns(len(sample_queries))
    for i, sample_query in enumerate(sample_queries):
        with cols[i]:
            if st.button(sample_query, key=f"sample_{i}"):
                st.session_state.current_query = sample_query
                st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p> Multimodal RAG System | Built with Streamlit & Google Gemini</p>
        <p>This system retrieves and analyzes text articles and images from The Batch newsletter</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
