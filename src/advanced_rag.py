#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced RAG System for Schrödinger Suite Documentation
Version: 2.1 (With Comprehensive Logging)
Features:
- Semantic Search with SentenceTransformers (all-mpnet-base-v2)
- Vector Store with FAISS
- Smart Chunking & Metadata Preservation
- Structured Output for LLM Context
- Comprehensive File Logging (rag_execution.log)
"""

import os
import sys
import time
import logging
import glob
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Third-party libraries
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required library. Please run: pip install -r requirements.txt")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configuration ---
CONFIG = {
    "docs_dir": "../data",  # Relative to src/
    "index_dir": "../index",
    "log_file": "../rag_execution.log",
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "chunk_size": 512,      # Characters
    "chunk_overlap": 50,    # Characters
    "top_k": 5,             # Number of chunks to retrieve
    "embedding_dim": 768    # Dimension for all-mpnet-base-v2
}

# --- Logging Setup ---
def setup_logging(log_path: str) -> logging.Logger:
    """
    Configures a logger that writes to both console and a file.
    Ensures logs persist even if the terminal session ends.
    """
    # Resolve absolute path
    abs_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), log_path))
    
    logger = logging.getLogger("SchrodingerRAG")
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # Clear existing handlers to avoid duplicates

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File Handler (Persistent)
    file_handler = logging.FileHandler(abs_log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler (Real-time)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("="*60)
    logger.info("STARTING ADVANCED RAG SYSTEM (v2.1)")
    logger.info(f"Log file saved to: {abs_log_path}")
    logger.info("="*60)
    
    return logger

# Initialize Logger
logger = setup_logging(CONFIG["log_file"])

# --- Core Functions ---

def load_documents(docs_dir: str) -> List[Dict[str, Any]]:
    """
    Loads all markdown/text files from the docs directory.
    Preserves filename and path in metadata.
    """
    documents = []
    abs_docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), docs_dir))
    
    logger.info(f"Scanning directory for documents: {abs_docs_dir}")
    
    if not os.path.exists(abs_docs_dir):
        logger.warning(f"Directory not found: {abs_docs_dir}. Creating empty placeholder.")
        os.makedirs(abs_docs_dir, exist_ok=True)
        # Create a dummy file for testing if directory is empty
        dummy_path = os.path.join(abs_docs_dir, "sample_desmond.md")
        with open(dummy_path, "w", encoding="utf-8") as f:
            f.write("# Desmond Molecular Dynamics\n\nDesmond is a high-performance MD engine.\nKey switch: `-md_step` defines the timestep.\nIf temperature > 300K, use NVT ensemble.")
        logger.info(f"Created sample file for testing: {dummy_path}")
        abs_docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), docs_dir))

    file_patterns = ["*.md", "*.txt", "*.rst"]
    files_found = []
    for pattern in file_patterns:
        files_found.extend(glob.glob(os.path.join(abs_docs_dir, pattern)))

    if not files_found:
        logger.error("No document files (.md, .txt, .rst) found in the data directory.")
        return documents

    logger.info(f"Found {len(files_found)} document files.")

    for filepath in files_found:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Skipping empty file: {filepath}")
                continue

            documents.append({
                "content": content,
                "metadata": {
                    "source": os.path.basename(filepath),
                    "path": filepath,
                    "size_bytes": len(content.encode('utf-8'))
                }
            })
            logger.info(f"Loaded: {os.path.basename(filepath)} ({len(content)} chars)")
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {str(e)}")

    logger.info(f"Successfully loaded {len(documents)} documents into memory.")
    return documents

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits text into overlapping chunks.
    Respects code blocks and paragraphs where possible (simple split for now).
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            # Try to break at a newline or space to avoid cutting words/code mid-stream
            last_newline = text.rfind('\n', start, end)
            last_space = text.rfind(' ', start, end)
            break_point = max(last_newline, last_space)
            
            if break_point > start + (chunk_size // 2): # Only break if we found a good spot
                end = break_point + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= text_len: break # Prevent infinite loop on very short remainders

    return chunks

def create_embeddings_and_index(documents: List[Dict[str, Any]]) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Creates embeddings using SentenceTransformers and builds a FAISS index.
    """
    logger.info(f"Loading embedding model: {CONFIG['model_name']}")
    start_time = time.time()
    
    try:
        model = SentenceTransformer(CONFIG['model_name'])
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded successfully in {load_time:.2f} seconds.")

    all_chunks = []
    all_metadata = []

    logger.info("Starting chunking and embedding process...")
    embed_start = time.time()

    for doc in documents:
        chunks = chunk_text(doc['content'], CONFIG['chunk_size'], CONFIG['chunk_overlap'])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            meta = doc['metadata'].copy()
            meta['chunk_id'] = i
            meta['chunk_preview'] = chunk[:50].replace('\n', ' ')
            all_metadata.append(meta)

    if not all_chunks:
        logger.warning("No chunks created. Index will be empty.")
        return None, []

    logger.info(f"Total chunks created: {len(all_chunks)}")
    logger.info("Generating embeddings (this may take a moment)...")
    
    embeddings = model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
    
    embed_time = time.time() - embed_start
    logger.info(f"Embeddings generated in {embed_time:.2f} seconds. Shape: {embeddings.shape}")

    # Build FAISS Index
    logger.info("Building FAISS Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    logger.info(f"FAISS Index built with {index.ntotal} vectors.")
    
    return index, all_metadata, all_chunks

def search_query(index: Any, metadata: List[Dict], chunks: List[str], query: str, top_k: int) -> List[Dict]:
    """
    Performs semantic search and returns ranked results.
    """
    if index is None or index.ntotal == 0:
        logger.warning("Index is empty. Cannot perform search.")
        return []

    logger.info(f"Processing query: '{query}'")
    model = SentenceTransformer(CONFIG['model_name'])
    
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    D, I = index.search(query_embedding, k=min(top_k, index.ntotal))
    
    results = []
    logger.info(f"Found {len(I[0])} relevant chunks:")
    
    for i, idx in enumerate(I[0]):
        score = float(D[0][i])
        chunk_text = chunks[idx]
        meta = metadata[idx]
        
        result_entry = {
            "rank": i + 1,
            "score": score,
            "source_file": meta['source'],
            "chunk_preview": meta['chunk_preview'],
            "content": chunk_text,
            "full_path": meta['path']
        }
        results.append(result_entry)
        
        logger.info(f"  [{i+1}] Source: {meta['source']} | Score: {score:.4f}")
        logger.info(f"      Preview: {meta['chunk_preview']}...")

    return results

def generate_context_for_llm(results: List[Dict]) -> str:
    """
    Formats search results into a clean context string for the LLM.
    Adheres to 'Fail-Safe' rules: preserves code, numbers, and conditions.
    """
    if not results:
        return "No relevant information found in the documentation."

    context_parts = []
    context_parts.append("--- START OF RETRIEVED CONTEXT ---")
    
    for res in results:
        part = f"""
[Source: {res['source_file']}] (Relevance Score: {res['score']:.4f})
Path: {res['full_path']}
Content:
{res['content']}
------------------------
"""
        context_parts.append(part)
    
    context_parts.append("--- END OF RETRIEVED CONTEXT ---")
    
    full_context = "\n".join(context_parts)
    logger.info("Context generated for LLM consumption.")
    return full_context

# --- Main Execution ---

if __name__ == "__main__":
    try:
        # 1. Load Documents
        docs = load_documents(CONFIG["docs_dir"])
        
        if not docs:
            logger.error("No documents to process. Exiting.")
            sys.exit(0)

        # 2. Create Index
        index_result = create_embeddings_and_index(docs)
        
        if index_result and len(index_result) == 3:
            index, metadata, chunks = index_result
            
            # Save index to disk for persistence (optional but good practice)
            index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), CONFIG["index_dir"], "faiss.index"))
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(index, index_path)
            logger.info(f"FAISS index saved to: {index_path}")

            # 3. Run Test Queries
            test_queries = [
                "How to set timestep in Desmond?",
                "Jaguar DFT convergence switches",
                "Glide docking grid parameters"
            ]

            logger.info("\n--- RUNNING TEST QUERIES ---")
            for q in test_queries:
                logger.info(f"\nQuery: {q}")
                results = search_query(index, metadata, chunks, q, CONFIG["top_k"])
                context = generate_context_for_llm(results)
                # In a real app, you would send 'context' + 'q' to an LLM here.
                logger.info("Context ready for LLM generation.")
                
            logger.info("\n=== EXECUTION COMPLETED SUCCESSFULLY ===")
            logger.info(f"Check '{CONFIG['log_file']}' for the full detailed report.")
            
        else:
            logger.error("Index creation failed.")

    except Exception as e:
        logger.exception(f"Critical error during execution: {str(e)}")
        sys.exit(1)
