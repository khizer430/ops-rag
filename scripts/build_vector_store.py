"""
scripts/build_vector_store.py
=============================
Phase 2 runner: JSON chunks -> embeddings -> ChromaDB.

Idempotent. Safe to re-run after adding more PDFs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make `src` importable when run as `python scripts/build_vector_store.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings.vector_store import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    VectorStore,
    load_chunks_from_json_dir,
)

logger = logging.getLogger("ops_rag.build_store")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    chunks = load_chunks_from_json_dir(DEFAULT_PROCESSED_DIR)
    if not chunks:
        logger.error(
            "No chunks in %s. Run `python scripts/ingest_pdf.py` first.",
            DEFAULT_PROCESSED_DIR,
        )
        return 1

    store = VectorStore()
    store.add_chunks(chunks)
    logger.info(
        "Done. Collection '%s' contains %d vectors.",
        store.collection_name, store.count(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())