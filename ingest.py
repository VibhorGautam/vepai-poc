"""
VEP Documentation Ingestion Pipeline

Downloads Ensembl VEP documentation pages, extracts meaningful text,
chunks it into overlapping segments, and stores embeddings in ChromaDB.
"""

import sys
import time

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    VEP_DOC_URLS,
)


def fetch_page(url: str) -> str:
    """Fetch a page and return raw HTML. Retries once on failure."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; VEPai-bot/1.0; "
            "GSoC research project)"
        )
    }
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt == 0:
                print(f"  Retry after error: {e}")
                time.sleep(2)
            else:
                raise
    return ""


def extract_text(html: str) -> str:
    """
    Pull meaningful content from an Ensembl documentation page.
    Strips navigation, footers, and script tags to keep only the
    article body text.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove noise elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Ensembl docs typically wrap content in #content or .content
    content_div = (
        soup.find("div", {"id": "content"})
        or soup.find("div", {"class": "content"})
        or soup.find("main")
        or soup.body
    )

    if content_div is None:
        return soup.get_text(separator="\n", strip=True)

    return content_div.get_text(separator="\n", strip=True)


def chunk_documents(texts: list[dict]) -> list[dict]:
    """
    Split extracted text into overlapping chunks.
    Each chunk carries metadata (source URL and label) so we can
    cite sources in answers later.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for doc in texts:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": doc["url"],
                    "label": doc["label"],
                    "chunk_index": i,
                },
            })

    return all_chunks


def build_vectorstore(chunks: list[dict]) -> Chroma:
    """Embed all chunks and persist to ChromaDB."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"Embedding {len(texts)} chunks and storing in ChromaDB...")
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    return vectorstore


def run_ingestion():
    """Main ingestion pipeline: fetch, extract, chunk, embed, store."""
    print("=" * 50)
    print("VEPai Documentation Ingestion")
    print("=" * 50)

    # Step 1: Fetch and extract text from each doc page
    documents = []
    for source in VEP_DOC_URLS:
        print(f"\nFetching: {source['label']}")
        print(f"  URL: {source['url']}")

        html = fetch_page(source["url"])
        text = extract_text(html)

        if len(text) < 100:
            print(f"  WARNING: Very little text extracted ({len(text)} chars)")
        else:
            print(f"  Extracted {len(text)} characters")

        documents.append({
            "url": source["url"],
            "label": source["label"],
            "text": text,
        })

        # Be polite to Ensembl servers
        time.sleep(1)

    # Step 2: Chunk the documents
    chunks = chunk_documents(documents)
    print(f"\nTotal chunks created: {len(chunks)}")

    # Step 3: Embed and store
    vectorstore = build_vectorstore(chunks)

    print(f"\nDone! Vector store saved to: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")

    # Quick sanity check
    results = vectorstore.similarity_search("VEP input format", k=2)
    print(f"\nSanity check (query: 'VEP input format'):")
    for r in results:
        print(f"  - [{r.metadata.get('label')}] {r.page_content[:80]}...")

    return vectorstore


if __name__ == "__main__":
    run_ingestion()
