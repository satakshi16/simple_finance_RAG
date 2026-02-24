"""
Financial AI Assistant — Streamlit Application

A conversational assistant that answers questions about:
  • "The Fundamental Principles of Financial Regulation" (via ChromaDB KB)
  • General finance topics (via Tavily web search)
  • Casual greetings (no tools invoked)

Includes in-conversation memory so the agent can recall user-provided details.
"""

import json
import streamlit as st
import logging
from langchain_core.messages import HumanMessage, AIMessage

from agent_setup import build_agent

# Logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Config
st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="📈",
    layout="centered",
)


# Cache the agent so it persists across reruns
@st.cache_resource
def get_agent():
    return build_agent()


agent = get_agent()


# Helper: render a single message
def render_message(msg: dict) -> None:
    """
    Render a chat message from session-state dict.
    This is the SINGLE rendering path used both for history replay
    and for the freshly generated message, ensuring expanders always appear.
    """
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        metadata = msg.get("metadata")
        if metadata and metadata.get("tool") != "No tool used":
            with st.expander("🔍 View AI Reasoning & Sources"):
                st.write(f"**Tool Invoked:** `{metadata['tool']}`")
                sources_str = (
                    ", ".join(metadata["sources"]) if metadata["sources"] else "None"
                )
                st.write(f"**Sources:** {sources_str}")
                st.write(f"**Confidence Score:** {metadata['confidence']}")


# Helper: extract response parts from agent output
def parse_agent_response(response: dict) -> dict:
    """
    Pull out the answer text, tool used, sources, and confidence
    from the raw LangGraph agent response.

    Handles multiple LangGraph versions and response shapes via
    three fallback extraction paths.
    """
    final_messages = response.get("messages", [])

    # ---- Debug: log the raw message types so we can diagnose ----
    for i, m in enumerate(final_messages):
        msg_type = getattr(m, "type", type(m).__name__)
        has_tc = bool(getattr(m, "tool_calls", None))
        logger.info("  msg[%d] type=%s  has_tool_calls=%s", i, msg_type, has_tc)

    # ------------------------------------------------------------------
    # 1. Detect which tool was invoked
    #    Scan ALL AI messages (not just the first) because in a ReAct
    #    loop the tool-calling step can appear at any position.
    # ------------------------------------------------------------------
    tool_used = "No tool used"
    for m in final_messages:
        if getattr(m, "type", None) != "ai":
            continue
        for tc in getattr(m, "tool_calls", []) or []:
            name = tc.get("name", "")
            # Skip the structured-output schema call — it is not a "real" tool
            if name and name != "GroundedResponse":
                tool_used = name
                break  # take the first real tool we find
        if tool_used != "No tool used":
            break

    logger.info("Detected tool_used = %s", tool_used)

    # ------------------------------------------------------------------
    # 2. Extract structured response (answer / sources / confidence)
    #    Three extraction paths, tried in order:
    # ------------------------------------------------------------------
    answer = "Could not parse a structured answer."
    sources: list[str] = []
    confidence_score: float = 0.0

    # --- Path A: top-level "structured_response" key (LangGraph ≥ 0.2) ---
    structured = response.get("structured_response")

    if structured:
        logger.info("Path A: Found top-level 'structured_response' key")
        if hasattr(structured, "model_dump"):
            d = structured.model_dump()
        elif hasattr(structured, "dict"):
            d = structured.dict()
        elif isinstance(structured, dict):
            d = structured
        else:
            d = dict(structured)

        answer = d.get("answer", answer)
        sources = d.get("sources", [])
        confidence_score = d.get("confidence_score", 0.0)

    else:
        # --- Path B: structured output encoded as a tool_call named
        #     "GroundedResponse" inside the last AI message --------
        logger.info("Path B: Looking for GroundedResponse in tool_calls")

        for m in reversed(final_messages):
            if getattr(m, "type", None) != "ai":
                continue
            for tc in getattr(m, "tool_calls", []) or []:
                if tc.get("name") == "GroundedResponse":
                    args = tc.get("args", {})
                    answer = args.get("answer", answer)
                    sources = args.get("sources", [])
                    confidence_score = args.get("confidence_score", 0.0)
                    logger.info("  → extracted from GroundedResponse tool_call")
                    break
            if answer != "Could not parse a structured answer.":
                break

        # --- Path C: last AI message content (plain text or JSON) ---
        if answer == "Could not parse a structured answer.":
            logger.info("Path C: Falling back to last AI message content")
            last_ai = None
            for m in reversed(final_messages):
                if getattr(m, "type", None) == "ai" and getattr(m, "content", ""):
                    last_ai = m
                    break

            if last_ai:
                content = last_ai.content
                # Try JSON parse (some versions return JSON string)
                try:
                    parsed_json = json.loads(content)
                    if isinstance(parsed_json, dict) and "answer" in parsed_json:
                        answer = parsed_json["answer"]
                        sources = parsed_json.get("sources", [])
                        confidence_score = parsed_json.get("confidence_score", 0.0)
                        logger.info("  → parsed JSON from AI content")
                    else:
                        answer = content
                except (json.JSONDecodeError, TypeError):
                    # Plain text (greetings, rejections, etc.)
                    answer = content

    logger.info(
        "Final → tool=%s  sources=%s  confidence=%s  answer_len=%d",
        tool_used, sources, confidence_score, len(answer),
    )

    return {
        "answer": answer,
        "tool": tool_used,
        "sources": sources,
        "confidence": confidence_score,
    }


# Session State Initialisation 
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []  # list[dict] for rendering
if "langchain_messages" not in st.session_state:
    st.session_state.langchain_messages = []  # list[HumanMessage|AIMessage]

# UI Header
st.title("📈 Financial AI Assistant")
st.markdown(
    "Ask questions about *The Fundamental Principles of Financial Regulation* "
    "or general market finance."
)

# 
# Render full chat history from session state.
#
# This is the ONLY place previous messages are drawn.  When a
# new query comes in (below), we do NOT render the new message
# inline — instead we append it to session state and call
# st.rerun() so the history loop re-draws everything, including
# the new message WITH its expander.  This guarantees that the
# expander for message N is still visible when message N+1
# arrives, because every message goes through the same
# render_message() path.
# 
for msg in st.session_state.ui_messages:
    render_message(msg)

#  Handle new user input
if user_query := st.chat_input("Ask a financial question..."):

    # 1. Append user message to both stores
    st.session_state.ui_messages.append({"role": "user", "content": user_query})
    st.session_state.langchain_messages.append(HumanMessage(content=user_query))

    # 2. Show a spinner while the agent works
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financial data..."):
            response = agent.invoke(
                {"messages": st.session_state.langchain_messages}
            )
            logger.info("Agent response keys: %s", list(response.keys()))
            parsed = parse_agent_response(response)

    # 3. Persist assistant response to both stores
    st.session_state.langchain_messages.append(
        AIMessage(content=parsed["answer"])
    )
    st.session_state.ui_messages.append(
        {
            "role": "assistant",
            "content": parsed["answer"],
            "metadata": {
                "tool": parsed["tool"],
                "sources": parsed["sources"],
                "confidence": parsed["confidence"],
            },
        }
    )

    # 4. Rerun so the history loop re-draws ALL messages uniformly,
    #    including the brand-new assistant message with its expander.
    st.rerun()
