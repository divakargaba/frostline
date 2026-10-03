"""BM25 retrieval over the operations playbook.

Owner: Div

Indexes markdown documents in data/playbook/ using BM25 (rank_bm25)
and provides a search interface for the agent to retrieve relevant
operational procedures and hydrate-management guidance.

TODO:
  - Implement playbook document loading and chunking
  - Implement BM25 index construction
  - Implement search function with snippet extraction
"""


def load_playbook(playbook_dir: str = "data/playbook") -> list[dict]:
    """Load and chunk markdown documents from the playbook directory.

    Args:
        playbook_dir: Path to the playbook directory.

    Returns:
        List of dicts with doc_id, title, content, chunks.
    """
    raise NotImplementedError


def build_index(documents: list[dict]) -> object:
    """Build a BM25 index over playbook document chunks.

    Args:
        documents: List of document dicts from load_playbook.

    Returns:
        BM25 index object.
    """
    raise NotImplementedError


def search(query: str, top_k: int = 3, index: object = None) -> list[dict]:
    """Search the playbook using BM25.

    Args:
        query: Free-text search query.
        top_k: Number of results to return.
        index: Pre-built BM25 index (built on first call if None).

    Returns:
        List of dicts with doc_id, title, snippet, score.
    """
    raise NotImplementedError
