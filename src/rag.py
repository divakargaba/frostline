"""BM25 RAG over playbook docs.

Owner: Div

Loads markdown docs from data/playbook/, chunks by section,
indexes with rank_bm25, and provides search(query, k) -> results.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi

log = logging.getLogger("frostline.rag")

PLAYBOOK_DIR = Path("data/playbook")


@dataclass
class Chunk:
    doc_id: str
    title: str
    section: str
    text: str
    sources: list[str]


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML front matter and body from markdown."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1])
                return meta or {}, parts[2].strip()
            except yaml.YAMLError:
                pass
    return {}, text


def _chunk_markdown(doc_id: str, title: str, body: str, sources: list[str]) -> list[Chunk]:
    """Split markdown body into chunks by ## headings."""
    chunks = []
    sections = re.split(r'\n(?=## )', body)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract section heading
        lines = section.split("\n", 1)
        heading = lines[0].lstrip("# ").strip()
        text = lines[1].strip() if len(lines) > 1 else heading

        # Skip very short chunks
        if len(text) < 20:
            continue

        chunks.append(Chunk(
            doc_id=doc_id,
            title=title,
            section=heading,
            text=text,
            sources=sources,
        ))

    # If no sections found, use whole body as one chunk
    if not chunks and body.strip():
        chunks.append(Chunk(
            doc_id=doc_id,
            title=title,
            section=title,
            text=body.strip(),
            sources=sources,
        ))

    return chunks


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\w+', text.lower())


class PlaybookIndex:
    """BM25 index over playbook chunks."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._loaded = False

    def load(self, playbook_dir: Path | None = None):
        """Load and index all playbook docs."""
        pdir = playbook_dir or PLAYBOOK_DIR
        if not pdir.exists():
            log.warning("Playbook dir %s does not exist", pdir)
            return

        self.chunks = []
        for md_file in sorted(pdir.glob("*.md")):
            if md_file.name == "SOURCES.md":
                continue

            text = md_file.read_text()
            meta, body = _parse_frontmatter(text)

            doc_id = meta.get("id", md_file.stem)
            title = meta.get("title", md_file.stem)
            sources = meta.get("sources", [])

            chunks = _chunk_markdown(doc_id, title, body, sources)
            self.chunks.extend(chunks)

        if self.chunks:
            corpus = [_tokenize(f"{c.title} {c.section} {c.text}") for c in self.chunks]
            self._bm25 = BM25Okapi(corpus)
            self._loaded = True
            log.info("Playbook index: %d chunks from %d docs", len(self.chunks),
                     len(set(c.doc_id for c in self.chunks)))

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Search the playbook. Returns top-k results."""
        if not self._loaded:
            self.load()

        if not self.chunks or self._bm25 is None:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get top-k unique docs (deduplicate by doc_id)
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        seen_docs: set[str] = set()
        results = []

        for idx, score in scored:
            if score <= 0:
                break
            chunk = self.chunks[idx]
            if chunk.doc_id in seen_docs:
                continue
            seen_docs.add(chunk.doc_id)

            # Build snippet from the chunk text (first 200 chars)
            snippet = chunk.text[:200].replace("\n", " ").strip()
            if len(chunk.text) > 200:
                snippet += "..."

            results.append({
                "id": chunk.doc_id,
                "title": chunk.title,
                "snippet": snippet,
                "sources": chunk.sources,
                "score": round(float(score), 3),
            })

            if len(results) >= k:
                break

        return results


# Global singleton index
_index = PlaybookIndex()


def search(query: str, top_k: int = 3) -> list[dict]:
    """Search the playbook. Public API used by tools.py."""
    return _index.search(query, k=top_k)


def get_index() -> PlaybookIndex:
    """Get the global index (for testing)."""
    return _index
