# VEPai: RAG Chatbot for Ensembl VEP Documentation

A proof-of-concept Retrieval-Augmented Generation chatbot that answers questions about Ensembl's **Variant Effect Predictor (VEP)** using its official documentation as the knowledge base.

Built as a GSoC 2026 prototype for the EMBL-EBI "Ask VEPai" project.

## Architecture

```
                         VEPai RAG Pipeline

  +-----------+     +-------------+     +------------------+
  |  Ensembl  |     |   Ingestion |     |    ChromaDB      |
  |  VEP Docs | --> |   Pipeline  | --> |  Vector Store    |
  |  (HTML)   |     | (chunk+embed)|    | (persistent)     |
  +-----------+     +-------------+     +--------+---------+
                                                 |
                                                 v
  +-----------+     +-------------+     +------------------+
  |   User    |     |  LangChain  |     |    Retriever     |
  |  Question | --> | RetrievalQA | <-- |  (MMR, top-5)    |
  +-----------+     +------+------+     +------------------+
                           |
                           v
                    +--------------+
                    |  LLM Answer  |
                    | with Sources |
                    +--------------+
```

**Ingestion** (`ingest.py`):
1. Fetches 5 key VEP documentation pages from ensembl.org
2. Extracts article text, strips navigation and boilerplate
3. Splits into ~500 character chunks with 50 char overlap
4. Embeds with `all-MiniLM-L6-v2` (local, no API key needed)
5. Stores persistently in ChromaDB

**Search** (`search.py`):
1. Encodes user query with the same embedding model
2. Retrieves top-5 relevant chunks using MMR (maximal marginal relevance) for diversity
3. Passes context + question to an LLM via LangChain RetrievalQA
4. Returns answer with source page citations

## Setup

```bash
# Clone and enter the project
cd vepai-poc

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### LLM Backend

The chatbot works with any OpenAI-compatible API. Two options:

**Option A: Local with Ollama (free, private)**
```bash
# Install Ollama (https://ollama.ai)
ollama pull mistral
# That's it, defaults point to localhost:11434
```

**Option B: OpenAI API or any compatible provider**
```bash
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-your-key-here"
export LLM_MODEL="gpt-4o-mini"
```

## Usage

### Step 1: Ingest VEP documentation

```bash
python ingest.py
```

This fetches the VEP docs, chunks them, and builds the vector store. Takes about 30 seconds on first run (mostly downloading the embedding model).

### Step 2: Ask questions

```bash
python search.py "How do I run VEP on a VCF file?"
python search.py "What output formats does VEP support?"
python search.py "How do I filter results by consequence type?"
python search.py "What is the default input format for VEP?"
```

Example output:
```
============================================================
ANSWER
============================================================
To run VEP on a VCF file, use the following command:

  ./vep -i input.vcf -o output.txt

VEP natively accepts VCF format as input. You can also specify
the input format explicitly with --format vcf. For compressed
VCF files, VEP handles .vcf.gz directly.

SOURCES:
  - Command Line Options: https://www.ensembl.org/.../vep_options.html
  - Input/Output Formats: https://www.ensembl.org/.../vep_formats.html
```

## Configuration

All parameters live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Local sentence transformer |
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `TOP_K` | 5 | Chunks retrieved per query |
| `SEARCH_TYPE` | mmr | Maximal marginal relevance |
| `LLM_MODEL` | mistral | LLM model name |

## Project Structure

```
vepai-poc/
  config.py          # All tuneable parameters
  ingest.py          # Documentation ingestion pipeline
  search.py          # RAG query pipeline with CLI
  requirements.txt   # Python dependencies
  chroma_db/         # Persistent vector store (created after ingestion)
```

## Future Work

- Add a FastAPI web interface for browser-based querying
- Support additional VEP documentation pages (plugins, cache setup)
- Implement conversation memory for multi-turn Q&A
- Add VCF example parsing to ground answers in real variant data
- Integrate with Ensembl REST API for live variant lookups
