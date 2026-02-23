"""
Ingest webpages and a PDF into a Chroma DB using LangChain.

Requires env:
- `OPENAI_API_KEY` (for embeddings)
- optional: `CHROMA_PERSIST_DIR` (directory to persist Chroma DB; default: ./chromadb)

Usage:
    python ingestion_chroma.py

This script:
- Loads the provided URLs (HTML via WebBaseLoader, PDF via PyPDFLoader downloaded locally).
- Splits text into paragraph/character chunks and attaches `source`+`chunk` metadata.
- Creates/uses a Chroma vector store and upserts embeddings.

Note: to load specific pages from a PDF, see the short example at the bottom of this file.
"""

import os
import tempfile
import requests
import logging
from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URLS_HTML = []
URLS_PDF = [
    "https://cepr.org/system/files/publication-files/68579-geneva_11_the_fundamental_principles_of_financial_regulation.pdf"
]


def download_pdf(url: str) -> str:
    """Download a PDF from `url` to a temp file and return its path."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return path


def load_documents(html_urls: List[str], pdf_urls: List[str]):
    docs = []

    if html_urls:
        logger.info("Loading HTML pages via WebBaseLoader...")
        loader = WebBaseLoader(web_paths=html_urls)
        loaded = loader.load()
        # ensure source metadata
        for d, u in zip(loaded, html_urls):
            md = dict(d.metadata or {})
            md.setdefault("source", u)
            d.metadata = md
        docs.extend(loaded)

    start_page = 17
    for url in pdf_urls:
        logger.info("Downloading PDF %s", url)
        path = download_pdf(url)
        loader = PyPDFLoader(path)
        loaded = loader.load()
        for d in loaded:
            if d.metadata.get('page', 0) + 1 >= start_page:
                md = dict(d.metadata or {})
                md["source"] = url
                d.metadata = md
                docs.append(d)
        try:
            os.remove(path)
        except Exception:
            pass
    return docs


def chunk_documents(docs, chunk_size: int = 1000, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # attach chunk id + ensure source metadata
    for i, c in enumerate(chunks):
        md = dict(c.metadata or {})
        src = md.get("source") or md.get("url")
        if src:
            md["source"] = src
        md["chunk"] = i
        c.metadata = md

    logger.info("Created %d chunks", len(chunks))
    return chunks


def main():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required in environment")

    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chromadb")

    logger.info("Loading source documents...")
    docs = load_documents(URLS_HTML, URLS_PDF)
    logger.info("Loaded %d documents", len(docs))

    logger.info("Chunking documents...")
    chunks = chunk_documents(docs)

    logger.info("Creating embeddings and Chroma vectorstore...")
    embeddings = OpenAIEmbeddings()

    # Chroma.from_documents will create a collection and persist (if persist_directory provided)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="regulatory_doc",
    )

    # persist to disk
    try:
        vectordb.persist()
    except Exception:
        # some Chroma builds persist automatically; ignore if persist not available
        pass

    try:
        print("written:", vectordb._collection.count())
    except Exception:
        print("Can't read db._collection.count(); check chromadb version")

    logger.info("Ingestion complete. Chroma persisted at %s", persist_dir)


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Quick answer: loading specific PDF pages with PyPDFLoader
# -----------------------------------------------------------------------------
# Yes — `PyPDFLoader` supports loading specific page ranges. Two common ways:
# 1) Use the loader's `load` with `page_numbers` if supported by your langchain version:
#
#    loader = PyPDFLoader(path)
#    docs = loader.load(page_numbers=[2,3,4])  # load pages 3-5 (0-based index)
#
# 2) Or load the full PDF and slice the returned documents (each item may be a page):
#
#    pages = loader.load()
#    subset = pages[10:20]  # pages 11-20
#
# If you need precise control, you can also use `pypdf` or similar directly to extract
# specific pages to a temporary PDF file and pass that file to `PyPDFLoader`.
# -----------------------------------------------------------------------------
