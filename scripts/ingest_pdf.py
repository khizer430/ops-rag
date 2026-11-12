"""
scripts/ingest_pdf.py
=====================
Phase 1 of ops-rag: PDF -> normalized, overlapping text chunks -> JSON.

Design decisions (and why):

1. PAGE-AWARE EXTRACTION
   We extract text page-by-page and attach `page` to every chunk. A RAG system
   is only as trustworthy as its citations -- when the Streamlit UI says
   "Source: manual.pdf p.42", that number must be real. Chunking the whole
   document as one blob would destroy that mapping.

2. NORMALIZATION BEFORE CHUNKING
   pypdf faithfully reproduces the PDF's physical line breaks, including
   hyphenated word splits ("servo-\nmotor"). Left alone, these become
   embedding noise and split words the model has never seen.

3. RECURSIVE SPLITTING (chunk_size=500, chunk_overlap=50)
   The splitter tries the "most semantic" separator first (\n\n paragraph),
   then \n, then sentence boundaries, then words, then raw characters. It only
   falls back to a harsher cut when a unit exceeds the budget.

4. STABLE CHUNK IDs
   Each chunk id is a hash of (source, page, index, text). This makes ingestion
   IDEMPOTENT: re-running on the same PDF produces identical ids, so a later
   ChromaDB `upsert` overwrites instead of duplicating.

Usage:
    python scripts/ingest_pdf.py                          # all PDFs in data/pdfs
    python scripts/ingest_pdf.py --pdf data/pdfs/x.pdf    # one file
    python scripts/ingest_pdf.py --chunk-size 400 --overlap 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# --------------------------------------------------------------------------- #
# Paths & logging
# --------------------------------------------------------------------------- #

# __file__ = <root>/scripts/ingest_pdf.py  ->  parents[1] = <root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR: Path = PROJECT_ROOT / "data" / "pdfs"
DEFAULT_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "processed"

logger = logging.getLogger("ops_rag.ingest")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class Chunk:
    """A single retrievable unit of text plus everything needed to cite it."""

    chunk_id: str      # stable, content-derived identifier (idempotent upserts)
    text: str          # the chunk content that gets embedded
    source: str        # originating file name, e.g. "ur5_manual.pdf"
    page: int          # 1-based page number (for citations in the UI)
    chunk_index: int   # 0-based position within the document
    char_count: int    # length of `text`, useful for diagnostics

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #

# "servo-\nmotor" -> "servomotor"  (only when a hyphen sits at a line break
# between two letters; preserves genuine hyphenated compounds mid-line).
_HYPHEN_LINEBREAK = re.compile(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])")
_HORIZONTAL_WS = re.compile(r"[ \t\u00a0\u2007\u202f]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(raw: str) -> str:
    """
    Clean pypdf output while PRESERVING paragraph structure.

    We deliberately keep newlines: RecursiveCharacterTextSplitter uses '\\n\\n'
    as its highest-priority separator, so destroying newlines would force it
    down to character-level cuts and produce semantically worse chunks.
    """
    if not raw:
        return ""

    text = raw.replace("\u00ad", "")          # strip soft hyphens
    text = _HYPHEN_LINEBREAK.sub("", text)    # rejoin hyphenated line breaks
    text = _HORIZONTAL_WS.sub(" ", text)      # collapse runs of spaces/tabs
    text = _EXCESS_NEWLINES.sub("\n\n", text) # cap blank-line runs at one blank line
    return text.strip()


# --------------------------------------------------------------------------- #
# PDF extraction
# --------------------------------------------------------------------------- #

def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Return [(page_number, normalized_text), ...] for all non-empty pages.

    Page numbers are 1-based because that is what a human reading the PDF sees.
    """
    reader = PdfReader(str(pdf_path))

    # Encrypted PDFs: try the empty-password case, which covers most
    # "protected but not really" manuals. Anything else we surface loudly.
    if reader.is_encrypted:
        logger.warning("%s is encrypted; attempting empty-password decrypt", pdf_path.name)
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise PdfReadError(f"Cannot decrypt {pdf_path.name}: {exc}") from exc

    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        # extract_text() returns None for image-only/scanned pages.
        cleaned = normalize_text(page.extract_text() or "")
        if not cleaned:
            logger.debug("%s page %d: no extractable text (likely scanned)", pdf_path.name, page_number)
            continue
        pages.append((page_number, cleaned))

    return pages


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Build the recursive splitter.

    Separator order matters -- it encodes a priority list of semantic units:
        "\n\n"  paragraph   (strongest boundary)
        "\n"    line
        ". "    sentence
        " "     word
        ""      character   (last resort, guarantees termination)
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,          # we measure in CHARACTERS (see "Why" note)
        separators=["\n\n", "\n", ". ", " ", ""],
        strip_whitespace=True,
    )


def make_chunk_id(source: str, page: int, index: int, text: str) -> str:
    """Deterministic id: same input -> same id -> safe re-ingestion."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    stem = Path(source).stem
    return f"{stem}::p{page}::c{index}::{digest}"


def chunk_pages(
    pages: list[tuple[int, str]],
    splitter: RecursiveCharacterTextSplitter,
    source: str,
) -> list[Chunk]:
    """Split each page into overlapping chunks, preserving page provenance."""
    chunks: list[Chunk] = []
    global_index = 0

    for page_number, page_text in pages:
        for piece in splitter.split_text(page_text):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(source, page_number, global_index, piece),
                    text=piece,
                    source=source,
                    page=page_number,
                    chunk_index=global_index,
                    char_count=len(piece),
                )
            )
            global_index += 1

    return chunks


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def ingest_pdf(
    pdf_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Path | None:
    """
    Parse one PDF, chunk it, and write `<stem>_chunks.json`.
    Returns the output path, or None if the PDF yielded no text.
    """
    logger.info("Parsing %s", pdf_path.name)
    pages = extract_pages(pdf_path)

    if not pages:
        logger.error(
            "%s produced no text. It is probably a scanned image PDF; "
            "you would need OCR (e.g. pytesseract) before RAG.", pdf_path.name
        )
        return None

    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks = chunk_pages(pages, splitter, pdf_path.name)

    payload = {
        "source": pdf_path.name,
        "num_pages": len(pages),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}_chunks.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Done: %s -> %d pages, %d chunks -> %s",
        pdf_path.name, len(pages), len(chunks), out_path.relative_to(PROJECT_ROOT),
    )
    return out_path


def _resolve_targets(target: Path) -> list[Path]:
    """Accept either a single .pdf file or a directory of PDFs."""
    if target.is_dir():
        return sorted(p for p in target.glob("*.pdf") if p.is_file())
    if target.is_file() and target.suffix.lower() == ".pdf":
        return [target]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into overlapping text chunks (ops-rag Phase 1)."
    )
    parser.add_argument(
        "--pdf", type=Path, default=DEFAULT_INPUT_DIR,
        help="Path to a PDF file or a directory of PDFs (default: data/pdfs).",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for chunk JSON output (default: data/processed).",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in characters.")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    targets = _resolve_targets(args.pdf)
    if not targets:
        logger.error("No PDFs found at %s. Drop a manual there and re-run.", args.pdf)
        return 1

    written: list[Path] = []
    for pdf in targets:
        try:
            out = ingest_pdf(pdf, args.out, args.chunk_size, args.overlap)
            if out:
                written.append(out)
        except PdfReadError as exc:
            logger.error("Could not read %s: %s", pdf.name, exc)
        except Exception as exc:  # noqa: BLE001 - keep batch ingestion going
            logger.exception("Unexpected failure on %s: %s", pdf.name, exc)

    if not written:
        logger.error("Nothing was written. Check the errors above.")
        return 1

    # --- Sanity preview so you can eyeball chunk quality immediately ---
    preview = json.loads(written[0].read_text(encoding="utf-8"))
    first = preview["chunks"][0]
    logger.info("-" * 70)
    logger.info("PREVIEW  %s  (chunk 1 of %d)", preview["source"], preview["num_chunks"])
    logger.info("page=%s  chars=%s", first["page"], first["char_count"])
    logger.info("-" * 70)
    print(first["text"][:400] + ("..." if len(first["text"]) > 400 else ""))
    logger.info("-" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())