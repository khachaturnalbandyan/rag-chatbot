"""
ingest.py
==========
One-time (or re-run-on-change) step: chunk + embed the documents in a
folder and save the resulting vector index to disk, so chat.py/app.py
can load it instantly without re-embedding every time.

Usage:
    python ingest.py --data_dir data --index_out index/paper_index
"""
import argparse

from src.embeddings import SentenceTransformerEmbedder
from src.generator import EchoGenerator
from src.rag import RAGPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Folder of .txt/.pdf files to ingest")
    parser.add_argument("--index_out", default="index/paper_index", help="Output path (no extension) for the saved index")
    parser.add_argument("--chunk_size", type=int, default=180, help="Chunk size in words")
    parser.add_argument("--chunk_overlap", type=int, default=40, help="Overlap between consecutive chunks, in words")
    args = parser.parse_args()

    print("Loading embedding model (all-MiniLM-L6-v2, downloads once on first run) ...")
    embedder = SentenceTransformerEmbedder()

    # The generator is never used during ingest - EchoGenerator is a
    # zero-cost placeholder so RAGPipeline can be constructed without an
    # API key just to build the index.
    pipeline = RAGPipeline(embedder, EchoGenerator(), chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    print(f"Ingesting documents from {args.data_dir} ...")
    paths = pipeline.ingest_directory(args.data_dir)
    print(f"  ingested {len(paths)} file(s): {[p for p in paths]}")
    print(f"  {len(pipeline.store)} chunks embedded")

    pipeline.save(args.index_out)
    print(f"Saved index to {args.index_out}.npz / {args.index_out}.json")


if __name__ == "__main__":
    main()
