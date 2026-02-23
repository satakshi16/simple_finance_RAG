"""
Ingest webpages and a PDF into Pinecone using LangChain WebBaseLoader.

Requires env:
- OPENAI_API_KEY
- PINECONE_API_KEY
- PINECONE_ENV
- PINECONE_INDEX (optional; default: finance-qna-index)
"""
import os
import getpass
import tempfile
import requests
import logging
from typing import List

# from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
# from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_core.documents import Document
# from langchain.schema import Document

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URLS_HTML = [
    "https://mycreditunion.gov/brochure-publications/brochure/money-basics-guide-budgeting-and-savings",
    "https://www.pillar.bank/2025/01/07/a-beginners-guide-to-budgeting-and-saving/",
]
URLS_PDF = [
    "https://www.consolidatedcreditsolutions.org/wp-content/uploads/2017/03/budgetingmadeeasy.pdf",
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
        for d, u in zip(loaded, html_urls * 1):  # loader sets source but ensure it
            d.metadata.setdefault("source", getattr(d.metadata, "source", None) or u)
        docs.extend(loaded)

    for url in pdf_urls:
        logger.info("Downloading PDF %s", url)
        path = download_pdf(url)
        loader = PyPDFLoader(path)
        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = url
        docs.extend(loaded)
        try:
            os.remove(path)
        except Exception:
            pass

    if not docs:
        print("No documents were loaded. Exiting.")
        return

    return docs


def chunk_documents(docs, chunk_size: int = 500, chunk_overlap: int = 100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)

    # Ensure each chunk includes the originating URL (source) and a chunk id
    for i, c in enumerate(chunks):
        md = dict(c.metadata or {})
        # prefer 'source' but also accept 'url' if present
        source = md.get("source") or md.get("url")
        if source:
            md["source"] = source
        md["chunk"] = i
        c.metadata = md

    logger.info("Created %d text chunks ready for embedding.", len(chunks))
    # print(chunks)
    return chunks


def init_pinecone(index_name: str, api_key: str, env: str | None = None, dim: int = 1536):
    # Use Pinecone client instance (new SDK)
    pc = Pinecone(api_key=api_key)

    # list_indexes() can return different types depending on SDK version
    try:
        idxs = pc.list_indexes()
        names = idxs.names() if hasattr(idxs, "names") else list(idxs)
    except Exception:
        names = []

    if index_name not in names:
        logger.info("Creating index %s", index_name)
        pc.create_index(name=index_name, dimension=dim, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    return pc


def main():
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")

    logger.info("Loading documents...")
    docs = load_documents(URLS_HTML, URLS_PDF)
    logger.info("Loaded %d documents", len(docs))

    logger.info("Chunking documents...")
    chunks = chunk_documents(docs)
    logger.info("Created %d chunks", len(chunks))

    logger.info("Initializing embeddings and Pinecone...")
    # Initialize embeddings and Pinecone vector store
    index_name = "finance-docs"
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # pc = init_pinecone(index_name, pinecone_api_key)
    vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
)

    # get the index object from client
    # idx = pc.index(index_name)

    # lc_docs = chunks

    logger.info("Upserting to Pinecone index '%s'...", index_name)
    # LC_Pinecone.from_documents(lc_docs, embeddings, index_name=index_name)
    vectorstore.add_documents(chunks)
    logger.info("Ingestion complete. %d vectors upserted.", len(chunks))


if __name__ == "__main__":
    main()