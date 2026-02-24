# Financial AI Assistant

A conversational AI assistant built with **Streamlit**, **LangChain**, and **LangGraph** that answers financial questions using a combination of a local knowledge base and web search.

---

## Features

| Feature | Description |
|---|---|
| **Knowledge Base Search** | Retrieves answers from a ChromaDB vector store containing *The Fundamental Principles of Financial Regulation* (2009). |
| **Web Search** | Uses Tavily to answer general finance questions with live web results. |
| **Greeting Handling** | Responds to casual greetings without invoking tools or discussing finance. |
| **Personal Finance Rejection** | Politely declines personal investment or portfolio advice. |
| **In-Conversation Memory** | Remembers details the user shares (name, role, etc.) and can recall them later — without hallucinating information that was never provided. |
| **Structured Output** | Every tool-backed answer includes sources and a confidence score (0.0 – 1.0). |
| **Transparent Reasoning** | Expandable "View AI Reasoning & Sources" panel on every tool-assisted response. |

---

## Project Structure

```
financial-ai-assistant/
├── app.py               # Streamlit UI, chat loop, session state management
├── agent_setup.py       # Agent construction: tools, LLM, LangGraph wiring
├── system_prompt.py     # ← EDIT THIS FILE to change agent behaviour
├── config.py            # Environment-driven configuration constants
├── requirements.txt     # Python dependencies
└── chromadb/            # Persisted ChromaDB vector store
```

### Modular Prompt Design

The system prompt lives in **`system_prompt.py`** — a standalone Python file that exports a single string constant (`SYSTEM_PROMPT`). To change the agent's routing rules, tone, or scope, edit only this file. No other code changes are required.

---

## Prerequisites

- **Python 3.10+**
- **OpenAI API key** — for embeddings and the chat model
- **Tavily API key** — for web search ([get one free at tavily.com](https://tavily.com))
- **ChromaDB vector store** — a pre-populated `./chromadb` directory with the regulatory document collection

---

## Setup

### 1. Clone & install dependencies

There were few issues with my uv add, so had to manually install dependencies using pip

### 2. Configure environment variables

Open `.env` and fill in your API keys:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 3. Prepare the vector store

If you don't already have a populated ChromaDB directory, you need to ingest the regulation document first using ingestion_chroma.py
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## Usage Guide

### Query Routing

The agent decides which action to take based on the user's message:

| User Message Type | Agent Action | Tool Used |
|---|---|---|
| Greeting ("Hi", "Hello") | Friendly reply, no tools | None |
| About the 2009 regulation book | Knowledge base search | `financial_regulation_kb_search` |
| General finance question | Web search | `tavily_search` |
| Personal finance advice | Polite rejection | None |
| Off-topic question | Gentle redirect | None |

### In-Conversation Memory

The assistant tracks the full conversation within a session. Examples:

```
User:  My name is Alex and I work at a hedge fund.
Agent: Nice to meet you, Alex! How can I help you today?

... (several messages later) ...

User:  What's my name?
Agent: Your name is Alex, as you mentioned earlier.
```

Memory is **session-scoped** — it resets when the browser tab is closed or the Streamlit app is restarted. The agent will never fabricate details; if you ask for something you never shared, it will tell you so.

### Structured Responses

Every tool-backed answer includes:

- **Answer** — the main response text
- **Sources** — document names, page numbers, or URLs
- **Confidence Score** — 0.0 (no supporting context) to 1.0 (fully supported)

Click the **"🔍 View AI Reasoning & Sources"** expander below any assistant message to inspect these details.

---

## Configuration Reference

All settings can be overridden via environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `TAVILY_API_KEY` | *(required)* | Tavily search API key |
| `CHROMA_PERSIST_DIR` | `./chromadb` | Path to ChromaDB persistence directory |
| `CHROMA_COLLECTION_NAME` | `regulatory_doc` | ChromaDB collection name |
| `RETRIEVER_TOP_K` | `3` | Number of chunks retrieved per KB query |
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI chat model name |
| `LLM_TEMPERATURE` | `0` | LLM temperature (0 = deterministic) |
| `TAVILY_MAX_RESULTS` | `3` | Max web search results per query |
| `TAVILY_TOPIC` | `finance` | Tavily search topic filter |

---

## Updating the System Prompt

To change how the agent behaves — routing rules, tone, new categories, memory instructions — edit **only** `system_prompt.py`:

```python
# system_prompt.py
SYSTEM_PROMPT = """Your updated prompt here..."""
```

Then restart the Streamlit app. No other files need to change.

---
