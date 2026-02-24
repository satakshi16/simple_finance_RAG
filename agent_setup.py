"""
Agent initialization: tools, LLM, and LangGraph agent assembly.
"""

import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from pydantic import BaseModel, Field

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    RETRIEVER_TOP_K,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TAVILY_MAX_RESULTS,
    TAVILY_TOPIC,
)
from system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ── Structured Output Schema ──────────────────────────────────
class GroundedResponse(BaseModel):
    """Schema to enforce grounded answers with source metadata."""

    answer: str = Field(
        description="The detailed answer based ONLY on the retrieved tool context."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="A list of specific source URLs, document names, or citation IDs used to form the answer.",
    )
    confidence_score: float = Field(
        description="A score from 0.0 to 1.0 indicating how fully the retrieved context answers the question.",
    )


# ── Agent Factory ─────────────────────────────────────────────
def build_agent():
    """
    Construct and return the LangGraph ReAct agent.

    Returns the compiled agent graph that can be invoked with
    ``agent.invoke({"messages": [...]})``
    """

    # --- Vector Store & Retriever ---
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    logger.info("Vector store initialized from %s", CHROMA_PERSIST_DIR)
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

    # --- Knowledge-Base Search Tool ---
    @tool
    def financial_regulation_kb_search(query: str) -> str:
        """
        Use this tool ONLY when the user's question is about
        'The Fundamental Principles of Financial Regulation'.
        Queries the internal Knowledge Base for regulatory principles and frameworks.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the knowledge base."

        formatted_docs = []
        for doc in docs:
            source_name = doc.metadata.get("source", "Unknown Internal Document")
            page_num = doc.metadata.get("page", "N/A")
            formatted_docs.append(
                f"[Source: {source_name}, Page: {page_num}]\nContent: {doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted_docs)

    # --- Web Search Tool ---
    tavily_web_search = TavilySearch(
        max_results=TAVILY_MAX_RESULTS,
        topic=TAVILY_TOPIC,
    )

    tools = [financial_regulation_kb_search, tavily_web_search]

    # --- LLM ---
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    # --- Assemble Agent ---
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=GroundedResponse
    )

    return agent