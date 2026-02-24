"""
Centralized configuration for the Financial AI Assistant.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── ChromaDB Settings ──
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chromadb")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "regulatory_doc")

# ── Retriever Settings ──
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "3"))

# ── LLM Settings ──
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# ── Tavily Settings ──
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "3"))
TAVILY_TOPIC = os.getenv("TAVILY_TOPIC", "finance")