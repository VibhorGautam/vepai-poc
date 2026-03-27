"""
VEP RAG Query Pipeline

Takes a user question about Ensembl VEP, retrieves relevant documentation
chunks from ChromaDB, and generates an answer with source citations using
an OpenAI-compatible LLM (Ollama, OpenAI, or any compatible API).
"""

import sys
import textwrap

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    MMR_FETCH_K,
    MMR_LAMBDA,
    SEARCH_TYPE,
    TOP_K,
)

# Prompt template that instructs the LLM to use only retrieved context
# and cite sources properly
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=textwrap.dedent("""\
        You are VEPai, an expert assistant for Ensembl's Variant Effect Predictor.
        Answer the question using ONLY the documentation context provided below.
        If the context does not contain enough information, say so honestly.

        When referencing specific details, mention which documentation section
        the information comes from.

        Context:
        {context}

        Question: {question}

        Answer (be specific, include any relevant command examples or parameters):"""),
)


def load_vectorstore() -> Chroma:
    """Load the persisted ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def build_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    """Set up the RetrievalQA chain with MMR-based retrieval."""
    retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": TOP_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        },
    )

    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0.1,  # low temp for factual answers
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )

    return chain


def format_sources(source_docs: list) -> str:
    """Format source documents into a readable citation block."""
    seen = set()
    sources = []
    for doc in source_docs:
        url = doc.metadata.get("source", "unknown")
        label = doc.metadata.get("label", "")
        key = f"{label}|{url}"
        if key not in seen:
            seen.add(key)
            sources.append(f"  - {label}: {url}")
    return "\n".join(sources)


def query(question: str) -> dict:
    """
    Run the full RAG pipeline for a given question.
    Returns the answer text and source citations.
    """
    print(f"Loading vector store from {CHROMA_DIR}...")
    vectorstore = load_vectorstore()

    # Verify we have documents
    collection = vectorstore._collection
    count = collection.count()
    if count == 0:
        return {
            "answer": "No documents found. Please run ingest.py first.",
            "sources": "",
        }

    print(f"Found {count} chunks in the vector store")
    print("Building QA chain...")

    chain = build_qa_chain(vectorstore)

    print(f"Querying: {question}\n")
    result = chain.invoke({"query": question})

    answer = result["result"]
    sources = format_sources(result.get("source_documents", []))

    return {"answer": answer, "sources": sources}


def main():
    """CLI entry point. Pass your question as a command line argument."""
    if len(sys.argv) < 2:
        print("Usage: python search.py \"<your question about VEP>\"")
        print()
        print("Examples:")
        print('  python search.py "How do I run VEP on a VCF file?"')
        print('  python search.py "What output formats does VEP support?"')
        print('  python search.py "How do I filter VEP results by consequence?"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = query(question)

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    if result["sources"]:
        print()
        print("SOURCES:")
        print(result["sources"])


if __name__ == "__main__":
    main()
