import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from pydantic import BaseModel, Field
import os
import logging
from dotenv import load_dotenv


# ==========================================
# 1. Setup & Configuration
# ==========================================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Financial AI Assistant", page_icon="📈", layout="centered")

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
# 2. Cache Agent Initialization 
# ==========================================
# Using @st.cache_resource prevents reloading ChromaDB and the LLM on every UI interaction
@st.cache_resource
def initialize_agent():
    # Initialize Vector Store
    PERSIST_DIR = "./chromadb"
    COLLECTION_NAME = "regulatory_doc"
    embeddings = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )
    
    logger.info(f"Vector store initialized from {PERSIST_DIR}")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Define KB Tool
    @tool
    def financial_regulation_kb_search(query: str) -> str:
        """
        Use this tool ONLY when the user's question is about the 'fundamental principles of financial regulation'.
        Queries the internal Knowledge Base for regulatory principles and frameworks.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the knowledge base."
        
        formatted_docs = []
        for doc in docs:
            source_name = doc.metadata.get("source", "Unknown Internal Document")
            page_num = doc.metadata.get("page", "N/A")
            formatted_chunk = f"[Source: {source_name}, Page: {page_num}]\nContent: {doc.page_content}"
            formatted_docs.append(formatted_chunk)
            
        return "\n\n---\n\n".join(formatted_docs)

    # Define Web Search Tool
    tavily_web_search = TavilySearch(
        max_results=3,
        topic="finance"
    )

    tools = [financial_regulation_kb_search, tavily_web_search]

    # Define LLM & System Prompt
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    SYSTEM_PROMPT = """You are a highly capable financial AI assistant. 
    You must analyze the user's query and strictly follow these routing rules:

    1. THE 2009 REGULATION BOOK (KNOWLEDGE BASE): If the question mentions or relates to the book "The Fundamental Principles of Financial Regulation", you MUST use the `financial_regulation_kb_search` tool. Write your answer based ONLY on the retrieved context. 
    2. GENERAL FINANCE (WEB SEARCH): If the question is about general financial topics but does NOT relate to the 2009 book, you MUST use the `tavily_search_results_json` tool to search the web. Write your answer based ONLY on the web search results.
    3. PERSONAL FINANCE (REJECT): If the question is personal, immediately reply with exactly: "I am unable to answer personal finance questions. I can only provide publicly available financial information."

    Ensure that all informational responses are accurately grounded in the tool outputs. Do not use prior knowledge outside of the tool contexts.
    """

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=GroundedResponse
    )
    return agent

# Load the agent once per session
agent_executor = initialize_agent()

# ==========================================
# 3. Streamlit UI & Memory Setup
# ==========================================
st.title("📈 Financial AI Assistant")
st.markdown("Ask questions about *The Fundamental Principles of Financial Regulation* or general market finance.")

# Initialize Short-Term Memory
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [] # Stores dicts for Streamlit rendering
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = [] # Stores tuples for the LangChain agent

# Render existing chat history
for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"]["tool"] != "No tool used":
            with st.expander("🔍 View AI Reasoning & Sources"):
                st.write(f"**Tool Invoked:** `{msg['metadata']['tool']}`")
                st.write(f"**Sources:** {', '.join(msg['metadata']['sources']) if msg['metadata']['sources'] else 'None'}")
                st.write(f"**Confidence Score:** {msg['metadata']['confidence']}")

# ==========================================
# 4. Handle User Input
# ==========================================
if user_query := st.chat_input("Ask a financial question..."):
    # 1. Add user message to UI and Memory
    st.session_state.ui_messages.append({"role": "user", "content": user_query})
    st.session_state.agent_memory.append(("user", user_query))
    
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financial data..."):
            
            # Pass the entire conversation history to the agent for short-term memory
            response = agent_executor.invoke({"messages": st.session_state.agent_memory})
            
            final_output = response["messages"]
            first_ai = next((m for m in final_output if getattr(m, "type", None) == "ai"), None)
            
            # Extract Tool Used (Ignoring the structured response schema call)
            tool_used = "No tool used"
            if first_ai and hasattr(first_ai, 'tool_calls') and first_ai.tool_calls:
                real_tools = [tc['name'] for tc in first_ai.tool_calls if tc['name'] != "GroundedResponse"]
                if real_tools:
                    tool_used = real_tools[0]

            # Extract Structured Response (handling different Pydantic parsing behaviors)
            structured = response.get("structured_response")
            answer = "Could not parse a structured answer."
            sources = []
            confidence_score = 0.0

            if structured:
                if hasattr(structured, "model_dump"): 
                    structured_dict = structured.model_dump()
                elif hasattr(structured, "dict"): 
                    structured_dict = structured.dict()
                elif isinstance(structured, dict):
                    structured_dict = structured
                else: 
                    structured_dict = {k: v for k, v in structured}

                answer = structured_dict.get("answer", answer)
                sources = structured_dict.get("sources", [])
                confidence_score = structured_dict.get("confidence_score", 0.0)
            else:
                # Fallback for rejections (e.g., personal finance questions)
                answer = final_output[-1].content

            # Display the answer
            st.markdown(answer)
            
            # Display metadata elegantly inside an expander
            if tool_used != "No tool used":
                with st.expander("🔍 View AI Reasoning & Sources"):
                    st.write(f"**Tool Invoked:** `{tool_used}`")
                    st.write(f"**Sources:** {', '.join(sources) if sources else 'None'}")
                    st.write(f"**Confidence Score:** {confidence_score}")

            # 3. Save AI response to Memory and UI state
            st.session_state.agent_memory.append(("assistant", answer))
            st.session_state.ui_messages.append({
                "role": "assistant", 
                "content": answer,
                "metadata": {
                    "tool": tool_used,
                    "sources": sources,
                    "confidence": confidence_score
                }
            })