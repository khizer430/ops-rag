"""
dashboard/app.py
================
Phase 6 of ops-rag: Streamlit chat UI over the FastAPI RAG backend.

Run:
    # terminal 1 (API):
    python -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000

    # terminal 2 (UI):
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# If your API is running elsewhere, set the OPS_RAG_API_URL environment variable
API_URL = os.environ.get("OPS_RAG_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 30.0

st.set_page_config(
    page_title="ops-rag",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <style>
      /* Slightly more breathing room in chat bubbles */
      [data-testid="stChatMessage"] { padding: 1rem; border-radius: 12px; }
      /* Compact source cards */
      .source-card {
          border-left: 3px solid #4f8bf9;
          padding: 0.6rem 0.9rem;
          margin: 0.5rem 0;
          border-radius: 6px;
          background: rgba(79, 139, 249, 0.06);
          font-size: 0.92rem;
      }
      .source-badge {
          display: inline-block;
          padding: 2px 8px;
          border-radius: 10px;
          font-size: 0.75rem;
          font-weight: 600;
          margin-right: 6px;
          background: #4f8bf9;
          color: white;
      }
      .source-badge.sim  { background: #22c55e; }
      .source-badge.page { background: #f59e0b; }
      .source-text {
          margin-top: 6px;
          color: #d1d5db;
          font-family: ui-monospace, "Cascadia Code", Menlo, monospace;
          font-size: 0.82rem;
          white-space: pre-wrap;
          max-height: 180px;
          overflow-y: auto;
          padding: 6px 8px;
          border-radius: 4px;
          background: rgba(0, 0, 0, 0.25);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #

def api_health() -> dict[str, Any] | None:
    """Ping the backend to check if it's alive and what version is loaded."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def call_chat(query: str, top_k: int) -> tuple[dict[str, Any] | None, str | None]:
    """Send a query to the /chat endpoint. Returns (response_json, error_message)."""
    try:
        r = requests.post(
            f"{API_URL}/chat",
            json={"query": query, "top_k": top_k},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 400:
            return None, r.json().get("detail", "Bad request.")
        if r.status_code >= 500:
            return None, f"Backend error ({r.status_code}). Check the API logs."
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the API at {API_URL}. Is uvicorn running?"
    except requests.exceptions.Timeout:
        return None, f"Request timed out after {REQUEST_TIMEOUT:.0f}s."
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error: {exc}"


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #

def render_sources(sources: list[dict[str, Any]]) -> None:
    """One st.expander per answer, each source as a styled HTML card."""
    if not sources:
        st.caption("No sources retrieved for this answer.")
        return

    with st.expander(f"📚 View Sources ({len(sources)})", expanded=False):
        for i, s in enumerate(sources, 1):
            sim = s.get("similarity", 0.0)
            page = s.get("page", "?")
            source_name = s.get("source", "unknown")
            text = (s.get("text") or "").strip()

            st.markdown(
                f"""
                <div class="source-card">
                  <span class="source-badge">#{i}</span>
                  <span class="source-badge page">p.{page}</span>
                  <span class="source-badge sim">{sim:.3f}</span>
                  <strong>{source_name}</strong>
                  <div class="source-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_assistant_message(payload: dict[str, Any], latency_ms: float) -> None:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(payload["answer"])
        st.caption(
            f"⏱ {latency_ms:.0f} ms · backend: `{payload.get('llm_backend', '?')}` · "
            f"sources: {len(payload.get('sources', []))}"
        )
        render_sources(payload.get("sources", []))


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

if "messages" not in st.session_state:
    # Each item: {"role": "user"|"assistant", "content": str, "payload": dict|None, "latency_ms": float}
    st.session_state.messages = []


# --------------------------------------------------------------------------- #
# Sidebar — health + settings
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.title("🤖 ops-rag")
    st.caption("Retrieval-Augmented Generation over technical PDFs")

    health = api_health()
    if health:
        st.success("API online", icon="✅")
        st.metric("Vectors indexed", health.get("vectors", 0))
        st.caption(f"model: `{health.get('model', '?')}`")
        st.caption(f"backend: `{health.get('llm_backend', '?')}`")
    else:
        st.error("API offline", icon="🔌")
        st.caption(f"Expected at `{API_URL}`")
        st.code("python -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000", language="powershell")

    st.divider()
    top_k = st.slider("Top-k chunks to retrieve", min_value=1, max_value=10, value=3,
                      help="Passed to the API on every query.")

    st.divider()
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Phase 6 · Streamlit frontend")


# --------------------------------------------------------------------------- #
# Main — history + input
# --------------------------------------------------------------------------- #

st.title("Ask about your documents")
st.caption(
    "Every answer cites the exact chunk and page it was built from. "
    "Open **View Sources** below any answer to verify."
)

# Re-render history on every Streamlit script rerun.
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg["content"])
    else:
        render_assistant_message(msg["payload"], msg["latency_ms"])

# Chat input.
if prompt := st.chat_input("Ask a question about the ingested PDFs..."):
    
    # 1. Render the user message instantly
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Call the API
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Retrieving context and generating answer..."):
            t0 = time.perf_counter()
            payload, error = call_chat(prompt, top_k)
            latency_ms = (time.perf_counter() - t0) * 1000.0

        if error:
            st.error(error)
            # Don't persist failed turns in history to prevent UI pollution
            st.session_state.messages.pop() 
        else:
            st.markdown(payload["answer"])
            st.caption(
                f"⏱ {latency_ms:.0f} ms · backend: `{payload.get('llm_backend', '?')}` · "
                f"sources: {len(payload.get('sources', []))}"
            )
            render_sources(payload.get("sources", []))

            # Save to history
            st.session_state.messages.append(
                {"role": "assistant", "payload": payload, "latency_ms": latency_ms}
            )