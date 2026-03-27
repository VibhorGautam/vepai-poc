"""
Configuration for VEPai RAG chatbot.
Centralizes all tuneable parameters and data sources.
"""

import os

# Embedding model (runs locally, no API key needed)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM settings (OpenAI-compatible, works with Ollama or any provider)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")  # default for local Ollama
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")

# Chunking parameters
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between consecutive chunks

# Retrieval settings
TOP_K = 5                          # number of chunks to retrieve
SEARCH_TYPE = "mmr"                # maximal marginal relevance for diversity
MMR_FETCH_K = 20                   # candidates to consider before MMR filtering
MMR_LAMBDA = 0.7                   # balance between relevance (1.0) and diversity (0.0)

# ChromaDB persistent storage
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "vep_docs"

# VEP documentation sources to ingest
VEP_DOC_URLS = [
    {
        "url": "https://www.ensembl.org/info/docs/tools/vep/index.html",
        "label": "VEP Overview",
    },
    {
        "url": "https://www.ensembl.org/info/docs/tools/vep/script/vep_options.html",
        "label": "Command Line Options",
    },
    {
        "url": "https://www.ensembl.org/info/docs/tools/vep/script/vep_filter.html",
        "label": "Filtering Results",
    },
    {
        "url": "https://www.ensembl.org/info/docs/tools/vep/online/index.html",
        "label": "Web Interface",
    },
    {
        "url": "https://www.ensembl.org/info/docs/tools/vep/vep_formats.html",
        "label": "Input/Output Formats",
    },
]
