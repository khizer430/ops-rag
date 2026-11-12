"""
src/generation/rag_pipeline.py
==============================
Phase 4 of ops-rag: retrieve -> build prompt -> generate -> cite.

Two LLM backends, one interface:

  MockLLM      Extractive generator. Deterministic. No external service.
               Picks the query-most-relevant sentences from retrieved chunks
               and stitches them with page citations. This is NOT a real
               language model -- it's an honest stand-in so the pipeline is
               runnable offline and the eval story stays reproducible. Its
               README description should say "extractive mock".

  OllamaLLM    Real generative backend over local Ollama (llama3.2 / mistral).
               Same interface; drop-in replacement. Selected automatically if
               Ollama is reachable, or explicitly via OPS_RAG_LLM=ollama.

Env vars
--------
  OPS_RAG_LLM     auto | mock | ollama     (default: auto)
  OLLAMA_HOST     http://localhost:11434   (default)
  OLLAMA_MODEL    llama3.2                 (default)

Usage
-----
  python -m src.generation.rag_pipeline
  python -m src.generation.rag_pipeline --llm mock --query "What is the title?"
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

# Make `src` importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.embeddings.vector_store import (  # noqa: E402
    RetrievedChunk,
    VectorStore,
)

logger = logging.getLogger("ops_rag.rag")


# --------------------------------------------------------------------------- #
# Public response model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class RAGResponse:
    """The full result of one pipeline call, wired straight into the API."""
    query: str
    answer: str
    sources: list[RetrievedChunk]
    prompt: str                 # for debugging / observability
    llm_backend: str            # "mock" or "ollama:<model>"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "prompt": self.prompt,
            "llm_backend": self.llm_backend,
        }


# --------------------------------------------------------------------------- #
# LLM interface + implementations
# --------------------------------------------------------------------------- #

class BaseLLM(Protocol):
    """Minimal interface every backend must implement."""
    name: str
    def generate(self, *, prompt: str, context: list[RetrievedChunk], query: str) -> str: ...


# ---- Extractive mock -------------------------------------------------------- #

# Tiny stopword set -- enough to kill the noise, small enough to stay readable.
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have i in is it its of on or "
    "that the this to was were what which who whom whose why how when where "
    "do does did should shall can could would will you your we our they their".split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text) if t.lower() not in _STOPWORDS}


class MockLLM:
    """
    Extractive stand-in. Scores every sentence in the retrieved chunks by
    Jaccard overlap with the query tokens, returns the top 3 as a paragraph
    with page citations.

    Why Jaccard-overlap rather than embedding similarity?
    It's deterministic, dependency-free, and interpretable: you can reproduce
    any answer by hand. That's exactly what a *test double* should be.
    """
    name = "mock"

    def __init__(self, max_sentences: int = 3, min_score: float = 0.10) -> None:
        self._max_sentences = max_sentences
        self._min_score = min_score

    def generate(self, *, prompt: str, context: list[RetrievedChunk], query: str) -> str:
        if not context:
            return "I don't have enough information in the provided context to answer that."

        q_tokens = _tokens(query)
        if not q_tokens:
            first = _SENTENCE_SPLIT.split(context[0].text.strip())[0]
            return f"{first} [{_cite(context[0])}]"

        scored: list[tuple[float, str, RetrievedChunk]] = []
        for chunk in context:
            for sent in _SENTENCE_SPLIT.split(chunk.text.strip()):
                s = sent.strip()
                if len(s) < 20:
                    continue
                
                # Clean up stray leading page numbers (e.g., "10 d) The...")
                s = re.sub(r"^\d+\s*", "", s)
                
                s_tokens = _tokens(s)
                if not s_tokens:
                    continue
                
                overlap = len(q_tokens & s_tokens) / len(q_tokens)
                if overlap >= 0.2:
                    scored.append((overlap, s, chunk))

        if not scored:
            top = context[0]
            first_sent = next(
                (s.strip() for s in _SENTENCE_SPLIT.split(top.text) if len(s.strip()) >= 20),
                top.text.strip()[:200],
            )
            return f"Weak match — the following may be relevant but should be verified: {first_sent} [{_cite(top)}]"

        # Dedup and take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        picked, seen = [], set()
        for score, sent, chunk in scored:
            if sent in seen:
                continue
            seen.add(sent)
            # Format as a Markdown bullet point with italicized citation
            picked.append(f"- {sent} *[{_cite(chunk)}]*")
            if len(picked) >= self._max_sentences:
                break

        # Return as a bulleted list!
        return "Here is what I found in the document:\n\n" + "\n".join(picked)

def _cite(chunk: RetrievedChunk) -> str:
    return f"{chunk.source} p.{chunk.page}"


# ---- Ollama backend --------------------------------------------------------- #

class OllamaLLM:
    """Thin client for a local Ollama instance. No API key required."""
    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 60.0) -> None:
        import requests  # local import keeps cold-start fast for mock path
        self._requests = requests
        self._host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self._model = model or os.environ.get("OLLAMA_MODEL") or "llama3.2"
        self._timeout = timeout
        self.name = f"ollama:{self._model}"

    def generate(self, *, prompt: str, context: list[RetrievedChunk], query: str) -> str:
        try:
            resp = self._requests.post(
                f"{self._host}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 400},
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = (data.get("response") or "").strip()
            return answer or "(Ollama returned an empty response.)"
        except Exception as exc:  # noqa: BLE001 - caller gets a graceful message
            logger.warning("Ollama call failed: %s -- falling back to extractive answer.", exc)
            return MockLLM().generate(prompt=prompt, context=context, query=query)


def make_llm(backend: str | None = None) -> BaseLLM:
    """
    Resolve backend from arg -> env var -> auto-probe.
    `auto` tries Ollama with a 1.5s probe; if unreachable, returns MockLLM.
    """
    chosen = (backend or os.environ.get("OPS_RAG_LLM") or "auto").lower()

    if chosen == "mock":
        logger.info("LLM backend: MockLLM (extractive)")
        return MockLLM()

    if chosen == "ollama":
        logger.info("LLM backend: OllamaLLM")
        return OllamaLLM()

    if chosen == "auto":
        try:
            import requests
            host = (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
            requests.get(f"{host}/api/tags", timeout=1.5).raise_for_status()
            logger.info("Ollama reachable at %s -- using OllamaLLM", host)
            return OllamaLLM()
        except Exception:  # noqa: BLE001
            logger.info("Ollama not reachable -- using MockLLM (extractive)")
            return MockLLM()

    raise ValueError(f"Unknown LLM backend: {chosen!r} (expected auto|mock|ollama)")


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

_PROMPT_TEMPLATE = (
    "You are a precise assistant answering questions about technical documents. "
    "Use ONLY the context below. If the answer is not in the context, say so. "
    "Cite sources inline as [p.<page>].\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n"
    "Answer:"
)


def build_prompt(query: str, chunks: Iterable[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a citation-tagged context block.
    We use a numbered [p.<page>] tag so the LLM (and the user) can attribute
    every claim back to a specific chunk -- the entire trust story of RAG.
    """
    blocks = []
    for i, c in enumerate(chunks, start=1):
        page_tag = f"p.{c.page}"
        blocks.append(f"[Source {i} | {page_tag} | {c.source}]\n{c.text}")
    context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    return _PROMPT_TEMPLATE.format(context=context, query=query)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

class RAGPipeline:
    """The full RAG step: retrieve -> format prompt -> generate -> cite."""
    def __init__(self, store: VectorStore, llm: BaseLLM | None = None, top_k: int = 3) -> None:
        self._store = store
        self._llm = llm or make_llm()
        self._top_k = top_k
        logger.info("RAGPipeline ready: backend=%s top_k=%d", self._llm.name, top_k)

    @property
    def llm_backend(self) -> str:
        return self._llm.name
    def answer(self, query: str, top_k: int | None = None) -> RAGResponse:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        # Use the request-specific top_k if provided, otherwise use the default
        k = top_k if top_k is not None else self._top_k
        
        sources = self._store.query(query, top_k=k)
        prompt = build_prompt(query, sources)
        text = self._llm.generate(prompt=prompt, context=sources, query=query)

        return RAGResponse(
            query=query,
            answer=text.strip(),
            sources=sources,
            prompt=prompt,
            llm_backend=self._llm.name,
        )

# --------------------------------------------------------------------------- #
# CLI demo
# --------------------------------------------------------------------------- #

def _print_response(r: RAGResponse) -> None:
    print("\n" + "=" * 88)
    print(f"Q: {r.query}")
    print(f"backend: {r.llm_backend}")
    print("-" * 88)
    print(r.answer)
    print("-" * 88)
    print(f"Sources ({len(r.sources)}):")
    for i, s in enumerate(r.sources, start=1):
        preview = s.text.replace("\n", " ").strip()[:100]
        print(f"  {i}. {s.source} p.{s.page}  sim={s.similarity:.3f}  {preview!r}...")
    print("=" * 88)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ops-rag RAG pipeline.")
    parser.add_argument("--query", "-q", action="append", default=None,
                        help="Query. Pass multiple -q for several. Defaults to a small demo set.")
    parser.add_argument("--llm", choices=["auto", "mock", "ollama"], default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    store = VectorStore()
    if store.count() == 0:
        logger.error("Vector store is empty. Run scripts/build_vector_store.py first.")
        return 2

    pipeline = RAGPipeline(store=store, llm=make_llm(args.llm), top_k=args.top_k)

    queries = args.query or [
        "What is the title of this document?",
        "Which organization issued this template?",
        "What is the seed funding for?",
    ]
    for q in queries:
        _print_response(pipeline.answer(q))
    return 0


if __name__ == "__main__":
    sys.exit(main())