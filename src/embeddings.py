"""
src.embeddings
================
Embedding backends behind one interface (`embed(list[str]) -> np.ndarray`)
so the rest of the pipeline (vectorstore, rag) never depends on which one
is in use.

SentenceTransformerEmbedder is the real, default backend: it downloads a
small (~80MB) local model on first use (no API key, no per-query cost) and
runs on CPU or GPU. HashEmbedder is a deterministic, dependency-free
stand-in used ONLY by the test suite, so tests run offline in CI/sandboxes
with no network and no multi-hundred-MB model download.
"""
from __future__ import annotations
import hashlib
import numpy as np


class Embedder:
    """Interface: any embedder must implement embed() and expose `dim`."""
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """Wraps sentence-transformers' all-MiniLM-L6-v2 (384-dim, fast, strong
    baseline quality for retrieval). Loaded lazily so importing this module
    never requires the model to be downloaded."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        embs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                   show_progress_bar=False)
        return embs.astype(np.float32)


class HashEmbedder(Embedder):
    """Deterministic, offline, dependency-free 'embedding' for tests only:
    hashes n-grams of each text into a fixed-size bag-of-hashes vector,
    L2-normalized. It has no real semantic understanding - it's here so
    tests can check the pipeline's PLUMBING (chunking -> index -> retrieve
    -> generate) without a network call or a large model download. Never
    use this for a real deployment."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out
