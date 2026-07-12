"""
semantic_cache.py — In-memory semantic cache using MiniLM + FAISS.

On first use, sentence-transformers downloads all-MiniLM-L6-v2 (~90 MB)
from Hugging Face. The container needs outbound HTTPS for this to succeed.
The model is cached to ~/.cache/huggingface inside the container; mount a
volume there if you want to persist it across restarts.

Index is in-memory only for Day 3. Disk persistence is a Day 4 option.
"""

from __future__ import annotations

import json
import os
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy imports — heavy libs loaded only when the cache is first instantiated
# ---------------------------------------------------------------------------
_model = None
_faiss = None

SIMILARITY_THRESHOLD = 0.92   # cosine similarity floor for a cache hit
EMBEDDING_DIM = 384            # all-MiniLM-L6-v2 output dimension


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss as _faiss_lib
        _faiss = _faiss_lib
    return _faiss


def _embed(text: str) -> np.ndarray:
    """Return a unit-normalised float32 embedding vector."""
    model = _get_model()
    vec = model.encode([text], normalize_embeddings=True)
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# SemanticCache class
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Thread-unsafe in-memory cache (Flask is single-threaded by default).
    For concurrent use, wrap operations in a threading.Lock.
    """

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD) -> None:
        faiss = _get_faiss()
        # IndexFlatIP with normalised vectors gives cosine similarity directly
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._entries: list[dict] = []   # parallel list: {task, answer}
        self.threshold = threshold

        # Tier 2: disk persistence setup
        self._save_counter = 0
        self._save_interval = 5  # Save to disk after every 5 stores
        self._disk_path = os.path.join(os.path.dirname(__file__), "cache_data")
        self.load_from_disk(self._disk_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, task: str) -> Optional[dict]:
        """
        Search the cache for a semantically similar past task.

        Returns:
            {"answer": str, "similarity": float} if similarity >= threshold,
            else None.
        """
        if self._index.ntotal == 0:
            return None

        query_vec = _embed(task)
        distances, indices = self._index.search(query_vec, k=1)

        similarity = float(distances[0][0])
        best_idx   = int(indices[0][0])

        if similarity >= self.threshold and best_idx >= 0:
            return {
                "answer":     self._entries[best_idx]["answer"],
                "similarity": round(similarity, 4),
            }
        return None

    def store(self, task: str, answer: str) -> None:
        """Embed the task and add it to the FAISS index."""
        vec = _embed(task)
        self._index.add(vec)
        self._entries.append({"task": task, "answer": answer})

        # Tier 2: periodic disk save
        self._save_counter += 1
        if self._save_counter >= self._save_interval:
            self.save_to_disk(self._disk_path)
            self._save_counter = 0

    def save_to_disk(self, path: Optional[str] = None) -> None:
        """Persist FAISS index and entry metadata to disk."""
        if path is None:
            path = self._disk_path
        try:
            os.makedirs(path, exist_ok=True)
            faiss = _get_faiss()
            faiss.write_index(self._index, os.path.join(path, "faiss.index"))
            with open(os.path.join(path, "entries.json"), "w", encoding="utf-8") as f:
                json.dump(self._entries, f)
        except Exception:
            pass

    def load_from_disk(self, path: Optional[str] = None) -> bool:
        """Load FAISS index and entry metadata from disk if present."""
        if path is None:
            path = self._disk_path
        idx_file = os.path.join(path, "faiss.index")
        ent_file = os.path.join(path, "entries.json")
        if os.path.exists(idx_file) and os.path.exists(ent_file):
            try:
                faiss = _get_faiss()
                loaded_index = faiss.read_index(idx_file)
                with open(ent_file, "r", encoding="utf-8") as f:
                    loaded_entries = json.load(f)
                if loaded_index.ntotal == len(loaded_entries):
                    self._index = loaded_index
                    self._entries = loaded_entries
                    return True
            except Exception:
                pass
        return False

    def __len__(self) -> int:
        return self._index.ntotal


# ---------------------------------------------------------------------------
# Module-level singleton shared across the Flask process lifetime
# ---------------------------------------------------------------------------
cache = SemanticCache()
