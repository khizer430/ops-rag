"""
scripts/evaluate_retrieval.py
=============================
Phase 3 of ops-rag: measure retrieval quality with Hit Rate@k and MRR@k.

Metrics
-------
Hit Rate@k = fraction of queries whose top-k contains at least one correct chunk.
             Answers "can the LLM possibly answer this from what we retrieved?"
MRR@k      = mean of 1/rank of the first correct chunk (0 if none in top-k).
             Answers "how high does the correct chunk rank, on average?"

'Correct' = chunk text contains at least one of the query's `expected_keywords`.

Design notes
------------
- k is read from the QA file's top_k field, NOT hard-coded, so evaluation
  matches whatever Phase 4 actually feeds the LLM.
- QA items where NO chunk anywhere in the corpus matches any keyword are
  flagged as GROUND-TRUTH BUGS, not retrieval misses. That distinction
  prevents the classic "evaluated against impossible questions -> metrics
  look terrible -> wrongly blame the model" trap.
- Per-question diagnostics print the first correct rank and the top-1 preview
  so you can eyeball whether failures are chunking, model, or QA-set issues.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to sys.path so `src.*` imports work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings.vector_store import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    RetrievedChunk,
    VectorStore,
    load_chunks_from_json_dir,
)

logger = logging.getLogger("ops_rag.eval")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QA_FILE = PROJECT_ROOT / "data" / "eval" / "qa_set.json"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class QAResult:
    qa_id: str
    question: str
    expected_keywords: list[str]
    retrieved: list[RetrievedChunk]
    first_correct_rank: int | None      # 1-based; None if missed
    ground_truth_exists: bool           # False -> QA-set bug, not retrieval miss

    @property
    def hit(self) -> bool:
        return self.first_correct_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.first_correct_rank is None else 1.0 / self.first_correct_rank

    def matched_keyword(self) -> str | None:
        """Return the specific keyword that made the top-ranked correct chunk match."""
        if self.first_correct_rank is None:
            return None
        chunk = self.retrieved[self.first_correct_rank - 1]
        for kw in self.expected_keywords:
            if kw.lower() in chunk.text.lower():
                return kw
        return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def chunk_matches_any_keyword(chunk_text: str, keywords: list[str]) -> bool:
    """Case-insensitive substring match against any keyword."""
    text_lc = chunk_text.lower()
    return any(kw.lower() in text_lc for kw in keywords)


def load_qa_set(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"QA set not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for i, item in enumerate(data.get("items", [])):
        missing = [k for k in ("id", "question", "expected_keywords") if k not in item]
        if missing:
            raise KeyError(f"QA item #{i} missing keys: {missing}")
        if not item["expected_keywords"]:
            raise ValueError(f"QA item '{item['id']}' has empty expected_keywords")
    return data


def build_corpus_keyword_index(
    chunks: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """
    Map each keyword -> set of chunk_ids whose text contains it.
    Used to distinguish "QA-set bug" (keyword exists nowhere) from
    "retrieval miss" (keyword exists in corpus but wasn't retrieved).
    """
    index: dict[str, set[str]] = {}
    for chunk in chunks:
        text_lc = chunk["text"].lower()
        # We do a naive full-text scan per chunk; fine for tens of thousands.
        # (If you scale to millions, invert: index tokens first, then lookup.)
        for token in _candidate_keywords_from_chunk(text_lc):
            index.setdefault(token, set()).add(chunk["chunk_id"])
    return index


def _candidate_keywords_from_chunk(text_lc: str) -> list[str]:
    """
    Placeholder hook. We instead check per-keyword membership on demand,
    which is simpler and avoids the false-positive risk of substring indexing.
    """
    return []


def keyword_exists_in_corpus(keyword: str, chunks: list[dict[str, Any]]) -> bool:
    """True if any chunk in the corpus contains this keyword (case-insensitive)."""
    kw = keyword.lower()
    return any(kw in c["text"].lower() for c in chunks)


# --------------------------------------------------------------------------- #
# Evaluation core
# --------------------------------------------------------------------------- #

def evaluate_one(
    store: VectorStore,
    item: dict[str, Any],
    top_k: int,
    corpus_chunks: list[dict[str, Any]],
) -> QAResult:
    q = item["question"]
    keywords: list[str] = item["expected_keywords"]

    retrieved = store.query(q, top_k=top_k)

    # Find the first (highest-ranked) correct chunk.
    first_correct_rank: int | None = None
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk_matches_any_keyword(chunk.text, keywords):
            first_correct_rank = rank
            break

    # Sanity: does ANY keyword from this QA item exist somewhere in the corpus?
    # If not, the item is unanswerable regardless of retrieval quality.
    gt_exists = any(keyword_exists_in_corpus(kw, corpus_chunks) for kw in keywords)

    return QAResult(
        qa_id=item["id"],
        question=q,
        expected_keywords=keywords,
        retrieved=retrieved,
        first_correct_rank=first_correct_rank,
        ground_truth_exists=gt_exists,
    )


def format_results_table(results: list[QAResult], top_k: int) -> str:
    """Human-readable per-question diagnostic."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 96)
    lines.append(f"  PER-QUESTION DIAGNOSTICS  (k={top_k})")
    lines.append("=" * 96)

    for r in results:
        hit_str = "HIT " if r.hit else "MISS"
        rank_str = f"@rank {r.first_correct_rank}" if r.hit else "      "
        gt_str = "" if r.ground_truth_exists else "  [GROUND-TRUTH BUG: keyword nowhere in corpus]"
        lines.append(f"\n[{r.qa_id}] {hit_str} {rank_str:<10} RR={r.reciprocal_rank:.3f}{gt_str}")
        lines.append(f"      Q: {r.question}")

        if r.hit:
            kw = r.matched_keyword() or "?"
            c = r.retrieved[r.first_correct_rank - 1]  # type: ignore[index]
            preview = c.text.replace("\n", " ").strip()[:130]
            lines.append(f"      matched on {kw!r}  |  p{c.page} sim={c.similarity:.3f}")
            lines.append(f"      {preview!r}...")
        else:
            # Show why top-1 didn't match — helps spot chunking/model problems.
            top = r.retrieved[0] if r.retrieved else None
            if top:
                preview = top.text.replace("\n", " ").strip()[:130]
                lines.append(
                    f"      top-1 was p{top.page} sim={top.similarity:.3f}: {preview!r}..."
                )
            lines.append(f"      expected keywords: {r.expected_keywords}")

    return "\n".join(lines)


def print_summary(results: list[QAResult], top_k: int) -> dict[str, float]:
    """Compute and print Hit Rate@k and MRR@k. Return as dict for testing."""
    # We EXCLUDE ground-truth-bug items from the denominator — they aren't
    # retrieval failures, they're QA-set construction failures.
    valid = [r for r in results if r.ground_truth_exists]
    bugs = len(results) - len(valid)

    if not valid:
        print("\nNo valid QA items (all flagged as ground-truth bugs). Fix qa_set.json.")
        return {"hit_rate": 0.0, "mrr": 0.0, "n_valid": 0, "n_bugs": bugs}

    hit_rate = sum(1 for r in valid if r.hit) / len(valid)
    mrr = statistics.fmean(r.reciprocal_rank for r in valid)

    print("\n" + "=" * 96)
    print(f"  SUMMARY  (n={len(valid)} valid, {bugs} ground-truth bug(s) excluded)")
    print("=" * 96)
    print(f"  Hit Rate@{top_k} : {hit_rate:.4f}   ({sum(1 for r in valid if r.hit)}/{len(valid)} questions with a correct chunk in top-{top_k})")
    print(f"  MRR@{top_k}      : {mrr:.4f}   (0.0=never found, 0.333=always at rank 3, 1.0=always at rank 1)")
    print()

    # Interpretive guidance — this is the part you actually act on.
    if hit_rate >= 0.95 and mrr >= 0.75:
        print("  ✅ Retrieval is production-grade for this corpus.")
    elif hit_rate >= 0.80:
        print("  ⚠️  Hit rate is OK but ranking could improve — consider a cross-encoder reranker.")
    else:
        print("  ❌ Hit rate below 0.80 — investigate chunking, embedder, or QA keywords before moving to Phase 4.")

    return {"hit_rate": hit_rate, "mrr": mrr, "n_valid": len(valid), "n_bugs": bugs}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality (Hit Rate@k, MRR@k).")
    parser.add_argument("--qa-file", type=Path, default=DEFAULT_QA_FILE,
                        help="Path to the QA ground-truth JSON.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR,
                        help="Where Phase 1 saved *_chunks.json files.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        qa_data = load_qa_set(args.qa_file)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Could not load QA set: %s", exc)
        return 2

    top_k = int(qa_data.get("top_k", 3))
    items = qa_data["items"]
    logger.info("Loaded %d QA items from %s (k=%d)", len(items), args.qa_file.name, top_k)

    # The corpus (Phase 1 chunks) is needed for the "ground truth exists?" check.
    corpus = load_chunks_from_json_dir(args.processed_dir)
    if not corpus:
        logger.error("No Phase 1 chunks in %s — run ingest_pdf.py first.", args.processed_dir)
        return 3

    store = VectorStore()
    if store.count() == 0:
        logger.error("Vector store is empty — run build_vector_store.py first.")
        return 4
    logger.info("Vector store ready: %d vectors, model=%s", store.count(), store.model_name)

    results = [evaluate_one(store, item, top_k, corpus) for item in items]

    print(format_results_table(results, top_k))
    summary = print_summary(results, top_k)

    # Exit non-zero on hard failure, so CI can gate on this.
    return 0 if summary["hit_rate"] >= 0.80 else 1


if __name__ == "__main__":
    sys.exit(main())