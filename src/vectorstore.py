"""
src.vectorstore
=================
A minimal, dependency-free vector index: cosine similarity via a plain
NumPy matmul. No FAISS/Chroma - at demo scale (a handful of documents,
hundreds of chunks) a brute-force matmul IS the correct choice: it's exact
(no approximate-NN recall loss), trivial to reason about, and fast enough
(a 1000x384 matrix search is sub-millisecond). Swap in FAISS/Chroma only
once the corpus is large enough that O(n) search actually matters - the
Embedder interface in embeddings.py means that swap doesn't touch
chunking.py or generator.py at all.
"""
from __future__ import annotations
import json
import os
import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self):
        self.vectors: np.ndarray | None = None   # (N, D), L2-normalized rows
        self.chunks: list[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: list[Chunk]):
        assert vectors.shape[0] == len(chunks), "one vector per chunk required"
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.concatenate([self.vectors, vectors], axis=0)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[tuple[Chunk, float]]:
        """query_vector: (D,), L2-normalized. Returns [(chunk, cosine_sim), ...],
        best match first."""
        if self.vectors is None or len(self.chunks) == 0:
            return []
        sims = self.vectors @ query_vector   # cosine sim, since both sides are L2-normalized
        top_k = min(top_k, len(self.chunks))
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]

    def __len__(self):
        return len(self.chunks)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta = [{"text": c.text, "doc_id": c.doc_id, "chunk_index": c.chunk_index} for c in self.chunks]
        np.savez(path + ".npz", vectors=self.vectors if self.vectors is not None else np.zeros((0, 0)))
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        store = cls()
        data = np.load(path + ".npz")
        vectors = data["vectors"]
        with open(path + ".json", encoding="utf-8") as f:
            meta = json.load(f)
        store.vectors = vectors if vectors.size > 0 else None
        store.chunks = [Chunk(text=m["text"], doc_id=m["doc_id"], chunk_index=m["chunk_index"]) for m in meta]
        return store
