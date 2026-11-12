"""
src/embeddings/vector_store.py
==============================
Phase 2 of ops-rag: dense embeddings + persistent vector store + retrieval.

What this module owns
---------------------
1. A single SentenceTransformer instance per VectorStore, used for BOTH
   document embeddings (ingest) and query embeddings (retrieval). This is
   non-negotiable: if doc and query vectors live in different spaces, cosine
   similarity is meaningless.
2. A persistent ChromaDB collection using cosine distance, upsert-safe by
   content-hashed chunk ids (re-ingesting the same PDF overwrites, never
   duplicates).
3. A `query()` method that returns `RetrievedChunk`s enriched with a
   human-interpretable `similarity` score (1 - cosine_distance).

Why we pass embeddings explicitly to Chroma instead of letting it embed:
  * Chroma's default embedder is an ONNX port of all-MiniLM-L6-v2 that does
    NOT normalize vectors. Ours does (normalize_embeddings=True). Mixing the
    two produces subtly wrong distances. Explicit is safe.
  * We already need a PyTorch model for query-time encoding. Reusing it for
    ingest guarantees the two paths are bit-identical.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("ops_rag.vector_store")

# --------------------------------------------------------------------------- #
# Defaults & paths
# --------------------------------------------------------------------------- #

# __file__ = <root>/src/embeddings/vector_store.py  ->  parents[2] = <root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_PERSIST_DIR: Path = PROJECT_ROOT / "data" / "chroma"
DEFAULT_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
DEFAULT_COLLECTION: str = "ops_rag_chunks"
DEFAULT_MODEL: str = "all-MiniLM-L6-v2"

# Every chunk dict coming out of Phase 1 MUST have these keys.
REQUIRED_CHUNK_KEYS: tuple[str, ...] = (
    "chunk_id", "text", "source", "page", "chunk_index",
)


# --------------------------------------------------------------------------- #
# Public data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by `VectorStore.query()` with its similarity score."""

    chunk_id: str
    text: str
    source: str
    page: int
    chunk_index: int
    distance: float        # Chroma cosine distance = 1 - cosine_similarity
    similarity: float      # cosine similarity in [-1, 1]; higher is better

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Chunk loader (shared with the __main__ demo & scripts/build_vector_store.py)
# --------------------------------------------------------------------------- #

def load_chunks_from_json_dir(processed_dir: Path | str = DEFAULT_PROCESSED_DIR) -> list[dict[str, Any]]:
    """Read every `*_chunks.json` under `processed_dir` and flatten to one list."""
    processed_dir = Path(processed_dir)
    if not processed_dir.is_dir():
        logger.error("Processed-chunk directory not found: %s", processed_dir)
        return []

    all_chunks: list[dict[str, Any]] = []
    for path in sorted(processed_dir.glob("*_chunks.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Skipping %s (unreadable): %s", path.name, exc)
            continue
        chunks = payload.get("chunks", [])
        logger.info("Loaded %3d chunks from %s", len(chunks), path.name)
        all_chunks.extend(chunks)

    logger.info("Total chunks loaded: %d", len(all_chunks))
    return all_chunks


# --------------------------------------------------------------------------- #
# VectorStore
# --------------------------------------------------------------------------- #

class VectorStore:
    """
    Thin, opinionated wrapper around a persistent ChromaDB collection.

    Parameters
    ----------
    persist_dir : where Chroma writes its sqlite+parquet files.
    collection_name : Chroma collection name (one collection per logical corpus).
    model_name : SentenceTransformer model id.
    device : "cpu" | "cuda" | None (auto-detect). Defaults to auto.
    """
    def query_mmr(
        self,
        query_text: str,
        top_k: int = 3,
        candidate_pool: int = 20,
        lambda_: float = 0.7,
    ) -> list[RetrievedChunk]:
        """
        Retrieve `top_k` chunks with Maximal Marginal Relevance.

        Balances relevance vs. redundancy: after the first (most relevant)
        chunk is selected, each subsequent pick maximizes
            λ · sim(query, candidate) − (1−λ) · max_sim(candidate, selected)
        which suppresses "hub" chunks (near-duplicates that crowd out the
        actually-informative one).

        Args:
            candidate_pool: over-fetch this many by cosine, then MMR down to top_k.
            lambda_: 1.0 = pure relevance (equivalent to query()); 0.0 = pure diversity.
        """
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError("lambda_ must be in [0, 1]")
        if self.count() == 0:
            return []

        n = min(candidate_pool, self.count())
        q_vec = self._model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)[0]
        raw = self._collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        candidates: list[tuple[str, str, dict, float, list[float]]] = []
        # We re-fetch embeddings for candidates -- cheap for n=20, and avoids
        # depending on Chroma's `include=["embeddings"]` which has varied by version.
        for cid, doc, meta, dist in zip(
            raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            candidates.append((
                str(cid), str(doc), meta or {},
                float(dist),   # will convert to similarity below
                [],            # placeholder; filled after re-embed
            ))

        # Re-embed candidate texts (same normalization -> comparable distances).
        texts = [c[1] for c in candidates]
        cand_embs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

        sims_to_query = [1.0 - c[3] for c in candidates]   # cosine sim
        selected_idx: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(top_k, len(remaining))):
            if not selected_idx:
                best = max(remaining, key=lambda i: sims_to_query[i])
            else:
                import numpy as np
                def mmr_score(i: int) -> float:
                    redundancy = max(
                        float(np.dot(cand_embs[i], cand_embs[j])) for j in selected_idx
                    )
                    return lambda_ * sims_to_query[i] - (1.0 - lambda_) * redundancy
                best = max(remaining, key=mmr_score)
            selected_idx.append(best)
            remaining.remove(best)

        results: list[RetrievedChunk] = []
        for i in selected_idx:
            cid, doc, meta, dist, _ = candidates[i]
            results.append(RetrievedChunk(
                chunk_id=cid, text=doc,
                source=str(meta.get("source", "unknown")),
                page=int(meta.get("page", 0)),
                chunk_index=int(meta.get("chunk_index", -1)),
                distance=dist, similarity=1.0 - dist,
            ))
        return results
    def __init__(
        self,
        persist_dir: Path | str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._model_name = model_name

        logger.info("Loading embedding model '%s' (first run downloads ~90 MB)", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        logger.info(
            "Model ready: dim=%d, max_seq_length=%d tokens",
            getattr(self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension)(),
            self._model.max_seq_length,
        )

        logger.info("Opening ChromaDB at %s (collection='%s')", self._persist_dir, collection_name)
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # <- metric selector, see "Why"
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection. Useful between evaluation runs."""
        logger.warning("Resetting collection '%s'", self._collection_name)
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def add_chunks(
        self,
        chunks: Iterable[dict[str, Any]],
        batch_size: int = 512,
    ) -> int:
        """
        Embed and upsert chunks. Idempotent: same chunk_id -> overwrite.

        Returns the number of chunks upserted (== len(chunks), unless error).
        """
        chunks = list(chunks)
        if not chunks:
            logger.warning("add_chunks called with an empty list; nothing to do")
            return 0

        # --- validate schema up-front so a bad chunk fails loudly, not mid-write ---
        for i, chunk in enumerate(chunks):
            missing = [k for k in REQUIRED_CHUNK_KEYS if k not in chunk]
            if missing:
                raise KeyError(
                    f"Chunk #{i} missing required keys {missing}. "
                    f"Re-run Phase 1 ingestion to regenerate the JSON."
                )

        # --- embed everything in one pass (fastest path in SentenceTransformer) ---
        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks (batch_size=32)...", len(texts))
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 200,
            normalize_embeddings=True,     # <- see "Why normalize" below
            convert_to_numpy=True,
        )
        logger.info("Embeddings computed: shape=%s", embeddings.shape)

        # --- upsert in batches (Chroma's sqlite writer prefers moderate batches) ---
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[c["chunk_id"] for c in batch],
                embeddings=embeddings[start : start + batch_size].tolist(),
                documents=[c["text"] for c in batch],
                metadatas=[
                    {
                        # Chroma metadata values must be str/int/float/bool.
                        "chunk_id":    str(c["chunk_id"]),
                        "source":      str(c["source"]),
                        "page":        int(c["page"]),
                        "chunk_index": int(c["chunk_index"]),
                        "char_count":  int(c.get("char_count", len(c["text"]))),
                    }
                    for c in batch
                ],
            )
            total += len(batch)

        logger.info(
            "Upserted %d chunks -> collection '%s' now holds %d vectors",
            total, self._collection_name, self.count(),
        )
        return total

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def query(self, query_text: str, top_k: int = 3) -> list[RetrievedChunk]:
        """
        Embed `query_text`, run HNSW ANN search, return top_k chunks.
        Results are ordered best -> worst (ascending Chroma distance).
        """
        if not query_text or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")
        if self.count() == 0:
            logger.warning("Collection '%s' is empty; query returns []", self._collection_name)
            return []

        # Encode with the SAME normalization as documents.
        q_vec = self._model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        # Chroma's `n_results` must not exceed collection size.
        n = min(top_k, self.count())
        raw = self._collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        results: list[RetrievedChunk] = []
        # Chroma nests one level for batched queries; we always query with 1.
        for cid, doc, meta, dist in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            meta = meta or {}
            results.append(
                RetrievedChunk(
                    chunk_id=str(cid),
                    text=str(doc),
                    source=str(meta.get("source", "unknown")),
                    page=int(meta.get("page", 0)),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    distance=float(dist),
                    similarity=1.0 - float(dist),   # cosine space -> sim = 1 - dist
                )
            )
        return results


# --------------------------------------------------------------------------- #
# __main__ demo (runs only if executed directly, not on import)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    chunks = load_chunks_from_json_dir(DEFAULT_PROCESSED_DIR)
    if not chunks:
        logger.error("No chunks found in %s — run `python scripts/ingest_pdf.py` first.", DEFAULT_PROCESSED_DIR)
        sys.exit(1)

    store = VectorStore()
    store.add_chunks(chunks)
    print(f"\nCollection now holds {store.count()} vectors "
          f"({store.embedding_dim}-dim, model={store.model_name})\n")

    demo_queries = [
        "What is the title of this document?",
        "Which organization issued this template?",
        "What are the funding areas?",
    ]
    for q in demo_queries:
        print(f"Query: {q!r}")
        for rank, r in enumerate(store.query(q, top_k=3), start=1):
            preview = r.text.replace("\n", " ")[:110]
            print(f"  {rank}. sim={r.similarity:.4f}  p{r.page:<3} {preview!r}")
        print()