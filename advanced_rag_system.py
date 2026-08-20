#!/usr/bin/env python3
"""
Advanced RAG System for Schrödinger Suite Documentation
Optimized for scientific parameters, switches, and simulation configs.
Dependencies: sentence-transformers, faiss-cpu, numpy, markdown
No LangChain dependency for maximum speed and minimal footprint.
"""

import os
import re
import glob
import json
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional

# Try importing required libraries
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Installing required packages...")
    os.system("pip install -q sentence-transformers faiss-cpu numpy")
    from sentence_transformers import SentenceTransformer
    import faiss

class SchrodingerRAG:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", index_path: str = "schrodinger_index.faiss"):
        """
        Initialize the RAG system.
        :param model_name: Embedding model (BGE is excellent for scientific retrieval)
        :param index_path: Path to save/load FAISS index
        """
        self.model_name = model_name
        self.index_path = index_path
        self.chunks_data = []  # Stores {'text': str, 'metadata': dict}
        self.index = None
        self.model = None
        
        print(f"🚀 Initializing Schrödinger RAG with model: {model_name}...")
        self.load_model()

    def load_model(self):
        """Load the embedding model."""
        # Use trust_remote_code=True for some BGE models if needed
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        print(f"✅ Model loaded successfully. Dimension: {self.model.get_sentence_embedding_dimension()}")

    def preprocess_markdown(self, text: str, source_file: str) -> List[str]:
        """
        Preprocess scientific markdown text.
        - Removes excessive whitespace
        - Preserves code blocks and parameter lists
        - Splits by headers and logical breaks
        """
        # Clean up extra newlines but keep structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Split by Markdown headers (##, ###) to keep context together
        # This ensures a specific module (e.g., Desmond) stays together
        sections = re.split(r'(?=^#{1,6}\s)', text, flags=re.MULTILINE)
        
        chunks = []
        for section in sections:
            if not section.strip():
                continue
            
            # If section is still too large (>1024 chars), split by paragraphs
            if len(section) > 1024:
                paragraphs = section.split('\n\n')
                current_chunk = ""
                for para in paragraphs:
                    if len(current_chunk) + len(para) > 1024:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para + "\n\n"
                    else:
                        current_chunk += para + "\n\n"
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(section.strip())
        
        return chunks

    def extract_scientific_params(self, text: str) -> Dict[str, str]:
        """
        Heuristic extraction of scientific parameters and switches.
        Looks for patterns like: -flag value, --option=value, KEYWORD = value
        """
        params = {}
        
        # Pattern for command line switches: -name value or --name=value
        switch_pattern = r'(-{1,2}[\w-]+)\s*[=\s]\s*([^\s,;]+)'
        matches = re.findall(switch_pattern, text)
        for key, val in matches:
            params[key] = val
            
        # Pattern for configuration keywords: KEYWORD value (common in Schrodinger inputs)
        # e.g., "FORCE_FIELD OPLS4"
        kw_pattern = r'\b([A-Z_]{2,})\s+([A-Z0-9_.-]+)\b'
        matches = re.findall(kw_pattern, text)
        for key, val in matches:
            if key not in ['THE', 'AND', 'FOR', 'WITH']: # Filter common words
                params[key] = val
                
        return params

    def ingest_documents(self, docs_folder: str = "docs"):
        """
        Ingest all markdown files from the docs folder.
        """
        if not os.path.exists(docs_folder):
            print(f"⚠️ Folder '{docs_folder}' not found. Creating dummy data for testing...")
            self.create_dummy_schrodinger_data()
            return

        md_files = glob.glob(os.path.join(docs_folder, "*.md"))
        if not md_files:
            print(f"⚠️ No .md files found in '{docs_folder}'.")
            return

        print(f"📂 Found {len(md_files)} documentation files. Processing...")
        
        for file_path in md_files:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                chunks = self.preprocess_markdown(content, filename)
                
                for chunk in chunks:
                    metadata = {
                        "source": filename,
                        "params": self.extract_scientific_params(chunk),
                        "char_count": len(chunk)
                    }
                    self.chunks_data.append({"text": chunk, "metadata": metadata})
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        print(f"✅ Processed {len(self.chunks_data)} chunks.")
        self.build_index()

    def create_dummy_schrodinger_data(self):
        """Create realistic dummy data for Schrödinger Suite if no docs exist yet."""
        print("📝 Generating dummy Schrödinger Suite documentation for testing...")
        
        dummy_docs = [
            {
                "filename": "desmond_md_simulation.md",
                "content": """
# Desmond Molecular Dynamics Module

## Overview
Desmond is a high-performance molecular dynamics (MD) program designed for efficient simulation of biomolecular systems. It utilizes the GROMOS force field and OPLS4.

## Command Line Usage
To run a simulation, use the `desmond` command with specific switches:
```bash
desmond -c input.cfg -o output.dae -cpu 48
```

## Configuration Parameters
Key parameters in the `.cfg` file:
- FORCE_FIELD OPLS4
- INTEGRATOR reversible_reference_propagator
- TIMESTEP 0.002
- TEMPERATURE 300.0
- PRESSURE 1.01325
- ENSEMBLE NPT

## Advanced Switches
-use_gpus true
-gpu_ids 0,1,2,3
-checkpoint_interval 10.0
-max_sim_time 100.0
                """
            },
            {
                "filename": "jaguar_dft_setup.md",
                "content": """
# Jaguar Quantum Chemistry Module

## DFT Calculations
Jaguar provides density functional theory (DFT) capabilities for electronic structure calculations.

## Running Jobs
Submit jobs via the command line:
```bash
jaguar run job.inp -wait -local
```

## Key Keywords
Theory DFT
Functional B3LYP
Basis 6-31G*
Charge 0
Multiplicity 1
Solvent Water
Grid UltraFine

## Memory Settings
-memory 16GB
-nproc 8
                """
            },
            {
                "filename": "glide_docking_protocol.md",
                "content": """
# Glide Molecular Docking

## Docking Protocols
Glide performs ligand docking using XP (Extra Precision) and SP (Standard Precision) modes.

## Execution
glide -HOST localhost:20 -WAIT -JOBNAME dock_run_01

## Input Parameters
LIGAND_FILE ligands.mae
RECEPTOR_FILE protein.mae
GRID_FILE grid.zip
DOCKING_MODE XP
SAMPLE_LIGANDS 1000

## Scoring
Emodel weight 1.0
Gscore weight 0.5
                """
            }
        ]
        
        os.makedirs("docs", exist_ok=True)
        for doc in dummy_docs:
            path = os.path.join("docs", doc["filename"])
            with open(path, 'w') as f:
                f.write(doc["content"])
        
        self.ingest_documents("docs")

    def build_index(self):
        """Build FAISS index from chunks."""
        if not self.chunks_data:
            return

        texts = [chunk['text'] for chunk in self.chunks_data]
        
        print("🧮 Generating embeddings (this may take a moment)...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Normalize embeddings for cosine similarity (by adding to FAISS as L2 on normalized vectors)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print(f"✅ FAISS index built with {self.index.ntotal} vectors.")
        
        # Save index
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path.replace('.faiss', '.json'), 'w') as f:
            json.dump(self.chunks_data, f)
        print(f"💾 Index saved to {self.index_path}")

    def load_index(self):
        """Load existing FAISS index and chunks."""
        if not os.path.exists(self.index_path):
            return False
        
        self.index = faiss.read_index(self.index_path)
        json_path = self.index_path.replace('.faiss', '.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.chunks_data = json.load(f)
            print(f"✅ Loaded existing index with {self.index.ntotal} vectors.")
            return True
        return False

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Perform semantic search.
        Returns top_k relevant chunks with metadata.
        """
        if self.index is None:
            print("❌ Index not loaded. Run ingest_documents first.")
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks_data):
                chunk = self.chunks_data[idx]
                results.append({
                    "score": float(1 / (1 + distances[0][i])), # Simple similarity score conversion
                    "text": chunk['text'],
                    "metadata": chunk['metadata']
                })
        
        return results

    def generate_context_for_llm(self, query: str) -> str:
        """
        Generate a formatted context string for the LLM to answer the user.
        Includes extracted parameters for precision.
        """
        results = self.search(query, top_k=3)
        if not results:
            return "No relevant documentation found."
        
        context_parts = []
        for i, res in enumerate(results):
            meta = res['metadata']
            part = f"""
--- Document {i+1}: {meta['source']} ---
Relevance Score: {res['score']:.2f}
Extracted Parameters: {json.dumps(meta['params']) if meta['params'] else 'None'}
Content:
{res['text']}
"""
            context_parts.append(part)
        
        return "\n".join(context_parts)

if __name__ == "__main__":
    # Initialize System
    rag = SchrodingerRAG(model_name="BAAI/bge-small-en-v1.5")
    
    # Try loading existing index, otherwise ingest docs
    if not rag.load_index():
        rag.ingest_documents("docs")
    
    # Test Queries relevant to Schrödinger Suite
    test_queries = [
        "How to set up an NPT ensemble in Desmond with OPLS4?",
        "What are the command line switches for running Jaguar DFT?",
        "How to configure Glide for Extra Precision (XP) docking?",
        "What is the default timestep for Desmond simulations?"
    ]
    
    print("\n" + "="*50)
    print("🔍 RUNNING TEST QUERIES")
    print("="*50)
    
    for q in test_queries:
        print(f"\n🗣️  User Query: {q}")
        context = rag.generate_context_for_llm(q)
        print("🤖 Retrieved Context for LLM:")
        print(context[:500] + "..." if len(context) > 500 else context) # Truncate for display
        print("-" * 30)

    print("\n✅ System ready. Place your real markdown files in /workspace/docs/ and re-run.")
