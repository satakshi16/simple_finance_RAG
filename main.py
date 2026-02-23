from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import MessagesState
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from pydantic import BaseModel, Field

import os
import logging
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

class GroundedResponse(BaseModel):
    """Schema to enforce grounded answers with source metadata."""
    answer: str = Field(
        description="The detailed answer based ONLY on the retrieved tool context."
    )
    sources: list[str] = Field(
        default_factory=list, 
        description="A list of specific source URLs, document names, or citation IDs used to form the answer."
    )
    confidence_score: float = Field(
        description="A score from 0.0 to 1.0 indicating how fully the retrieved context answers the question."
    )

# ==========================================
# 1. Initialize Vector Store (Knowledge Base)
# ==========================================
# Assumes ingestion is already done and DB exists locally
PERSIST_DIR = "./chromadb"
COLLECTION_NAME = "regulatory_doc"
embeddings = OpenAIEmbeddings()

vector_store = Chroma(
collection_name=COLLECTION_NAME,
embedding_function=embeddings,
persist_directory=PERSIST_DIR
)

# Create a retriever from the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
logger.info(f"Retriever created with k=3 for ChromaDB from {PERSIST_DIR}")

# ==========================================
# 2. Define Tools
# ==========================================
@tool
def financial_regulation_kb_search(query: str) -> str:
    """
    Use this tool ONLY when the user's question is about the 'fundamental principles of financial regulation'.
    Queries the internal Knowledge Base for regulatory principles and frameworks.
    """

    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the knowledge base."
    
    # Explicitly bundle metadata with the text so the LLM can extract it
    formatted_docs = []
    for doc in docs:
        # Extract metadata (fallback to "Unknown" if missing)
        source_name = doc.metadata.get("source", "Unknown Internal Document")
        page_num = doc.metadata.get("page", "N/A")
        
        # Combine into a single string for the LLM's context window
        formatted_chunk = f"[Source: {source_name}, Page: {page_num}]\nContent: {doc.page_content}"
        formatted_docs.append(formatted_chunk)
        
    return "\n\n---\n\n".join(formatted_docs)

# Create the Tavily search tool for general finance queries
tavily_web_search = TavilySearch(
    max_results=3,
    topic="finance"
)

# Bundle tools for the agent
tools = [financial_regulation_kb_search]
        #  tavily_web_search]

# ==========================================
# 3. Define the LLM and System Prompt
# ==========================================

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# The state_modifier acts as our router instructions
SYSTEM_PROMPT = """You are a highly capable financial AI assistant. 
You must analyze the user's query and strictly follow these routing rules:

1. THE 2009 REGULATION BOOK (KNOWLEDGE BASE): If the question mentions or relates to the book "The Fundamental Principles of Financial Regulation" (2009, Geneva Reports on the World Economy 11), its authors (Markus Brunnermeier, Andrew Crockett, Charles Goodhart, Avinash Persaud, Hyun Song Shin), or core themes specific to it (e.g., macro-prudential vs. micro-prudential regulation, counter-cyclical capital charges), you MUST use the `financial_regulation_kb_search` tool. Write your answer based ONLY on the retrieved context. Do NOT use the web search tool for this book.

2. GENERAL FINANCE (WEB SEARCH): If the question is about general financial topics, current markets, or other public financial frameworks, but does NOT relate to the 2009 "Fundamental Principles of Financial Regulation" book, you MUST use the `tavily_search_results_json` tool to search the web. Write your answer based ONLY on the web search results.

3. PERSONAL FINANCE (REJECT): If the question is personal (e.g., "should I buy this stock?", "how should I invest my $5000?", "is my portfolio good?"), you MUST NOT use any tools. Immediately reply with exactly: "I am unable to answer personal finance questions. I can only provide publicly available financial information."

Ensure that all informational responses are accurately grounded in the tool outputs. Do not use prior knowledge outside of the tool contexts.
"""

# ==========================================
# 4. Compile the LangGraph Agent
# ==========================================

agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    response_format=GroundedResponse
)

# ==========================================
# 5. Testing the Agent Logic
# ==========================================
if __name__ == "__main__":
    test_queries = [
        # Should route to ChromaDB
        "According to Fundamental Principles of Financial Regulation, in financial regulation, what are the purposes of regulation?",
        # Should route to Tavily
        # "What is Apple's current stock price and recent earnings report?",
        # Should trigger standard rejection string
        "I have $10,000 saved up. Should I invest it in Tesla or an S&P 500 ETF?"
    ]

    for q in test_queries:
        print(f"\n--- Question: {q} ---")
        # Run the graph
        response = agent_executor.invoke({"messages": [("user", q)]})
        final_output = response["messages"]

        # Tool used
        # Find the first assistant message 
        first_ai = next(m for m in final_output if getattr(m, "type", None) == "ai")
        tool_used = first_ai.tool_calls[0]['name'] if first_ai.tool_calls and first_ai.tool_calls[0]['name'] != "GroundedResponse" else "No tool used"
        print(f"Tool used: {tool_used}")

        # Print the answer
        for field_name, value in response["structured_response"]:
            print(f"{field_name}: {value}")