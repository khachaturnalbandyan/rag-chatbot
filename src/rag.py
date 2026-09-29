"""
src.rag
========
Ties chunking + embeddings + vector search + generation into one
RAGPipeline class:

    pipeline = RAGPipeline(embedder, generator)
    pipeline.ingest(["data/paper.txt"])
    answer, sources = pipeline.query("What is the CMDF module?")

Kept deliberately thin - each stage (chunking.py, embeddings.py,
vectorstore.py, generator.py) is independently testable and swappable;
this module just wires them together in the standard RAG order:
retrieve -> build prompt -> generate.
"""
from __future__ import annotations
import glob
import os

from .chunking import chunk_file, chunk_text, Chunk
from .embeddings import Embedder
from .generator import Generator, build_prompt
from .vectorstore import VectorStore


class RAGPipeline:
    def __init__(self, embedder: Embedder, generator: Generator,
                 chunk_size: int = 180, chunk_overlap: int = 40, top_k: int = 4):
        self.embedder = embedder
        self.generator = generator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.store = VectorStore()

    def ingest_text(self, text: str, doc_id: str):
        chunks = chunk_text(text, doc_id, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        self._add_chunks(chunks)

    def ingest_files(self, paths: list[str]):
        """Accepts .txt files directly, and .pdf files if pypdf is installed
        (extracted page-by-page, then chunked as one document)."""
        all_chunks: list[Chunk] = []
        for path in paths:
            if path.lower().endswith(".pdf"):
                text = _extract_pdf_text(path)
                doc_id = os.path.basename(path)
                all_chunks.extend(chunk_text(text, doc_id, chunk_size=self.chunk_size, overlap=self.chunk_overlap))
            else:
                all_chunks.extend(chunk_file(path, chunk_size=self.chunk_size, overlap=self.chunk_overlap))
        self._add_chunks(all_chunks)

    def ingest_directory(self, directory: str, patterns=("*.txt", "*.pdf")):
        paths = []
        for pattern in patterns:
            paths.extend(sorted(glob.glob(os.path.join(directory, pattern))))
        if not paths:
            raise FileNotFoundError(f"No matching files ({patterns}) found in {directory}")
        self.ingest_files(paths)
        return paths

    def _add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(vectors, chunks)

    def retrieve(self, question: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        query_vec = self.embedder.embed([question])[0]
        return self.store.search(query_vec, top_k=top_k or self.top_k)

    def query(self, question: str, top_k: int | None = None) -> tuple[str, list[tuple[Chunk, float]]]:
        results = self.retrieve(question, top_k=top_k)
        if not results:
            return "No documents have been ingested yet.", []
        prompt = build_prompt(question, [c.text for c, _ in results])
        answer = self.generator.generate(prompt)
        return answer, results

    def save(self, path: str):
        self.store.save(path)

    def load(self, path: str):
        self.store = VectorStore.load(path)


def _extract_pdf_text(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)
