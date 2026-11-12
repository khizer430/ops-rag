"""
src/serving/api.py
==================
Phase 5 of ops-rag: FastAPI backend for the RAG pipeline.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make `src` importable when run via `python -m uvicorn ...`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.embeddings.vector_store import VectorStore  # noqa: E402
from src.generation.rag_pipeline import RAGPipeline, make_llm  # noqa: E402

logger = logging.getLogger("ops_rag.api")


# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=2000, description="User question.")]
    top_k: Annotated[int, Field(ge=1, le=10, default=3, description="Number of chunks to retrieve (default 3).")]


class SourceCitation(BaseModel):
    chunk_id: str
    source: str
    page: int
    chunk_index: int
    similarity: float = Field(description="Cosine similarity in [-1, 1]; higher is better.")
    text: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceCitation]
    llm_backend: str
    prompt: str | None = Field(
        default=None,
        description="Exact prompt sent to the LLM. Only populated when OPS_RAG_DEBUG_PROMPT=1.",
    )


class HealthResponse(BaseModel):
    status: str
    vectors: int
    model: str
    llm_backend: str


# --------------------------------------------------------------------------- #
# Lifespan: load once, share everywhere
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ops-rag API...")
    store = VectorStore()
    if store.count() == 0:
        logger.warning(
            "Vector store is empty -- run `python scripts/build_vector_store.py` "
            "and restart. The API will still respond to /health."
        )

    llm = make_llm()  # reads OPS_RAG_LLM / OLLAMA_* env vars, auto-probes
    pipeline = RAGPipeline(store=store, llm=llm, top_k=3)

    app.state.store = store
    app.state.pipeline = pipeline

    logger.info(
        "API ready: collection=%s vectors=%d model=%s llm=%s",
        store.collection_name, store.count(), store.model_name, pipeline.llm_backend,
    )

    yield

    logger.info("Shutting down ops-rag API.")


# --------------------------------------------------------------------------- #
# App + middleware
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="ops-rag API",
    description="Retrieval-Augmented Generation over technical PDFs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vectors=app.state.store.count(),
        model=app.state.store.model_name,
        llm_backend=app.state.pipeline.llm_backend,
    )


@app.get("/stats")
def stats() -> dict[str, Any]:
    store: VectorStore = app.state.store
    return {
        "collection": store.collection_name,
        "vectors": store.count(),
        "embedding_dim": store.embedding_dim,
        "model": store.model_name,
        "llm_backend": app.state.pipeline.llm_backend,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    pipeline: RAGPipeline = app.state.pipeline
    try:
        resp = pipeline.answer(req.query, top_k=req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline error on query=%r top_k=%d", req.query, req.top_k)
        raise HTTPException(status_code=500, detail="Internal error during RAG.") from exc

    include_prompt = os.environ.get("OPS_RAG_DEBUG_PROMPT") == "1"

    return ChatResponse(
        query=resp.query,
        answer=resp.answer,
        sources=[
            SourceCitation(
                chunk_id=s.chunk_id,
                source=s.source,
                page=s.page,
                chunk_index=s.chunk_index,
                similarity=round(s.similarity, 4),
                text=s.text,
            )
            for s in resp.sources
        ],
        llm_backend=resp.llm_backend,
        prompt=resp.prompt if include_prompt else None,
    )


# --------------------------------------------------------------------------- #
# Dev entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serving.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )