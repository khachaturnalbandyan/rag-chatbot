"""
chat.py
========
Interactive CLI chat loop over a pre-built index (see ingest.py). Picks the
LLM backend automatically: ANTHROPIC_API_KEY or OPENAI_API_KEY in the
environment (or a .env file) -> that hosted API; otherwise falls back to a
small local Hugging Face model (no key needed, but lower quality and a
one-time ~3GB download).

Usage:
    python chat.py --index index/paper_index
    python chat.py --index index/paper_index --backend anthropic
"""
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv is optional - env vars can be set directly instead

from src.embeddings import SentenceTransformerEmbedder
from src.generator import build_generator
from src.rag import RAGPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index/paper_index", help="Path (no extension) to a saved index from ingest.py")
    parser.add_argument("--backend", default=None, choices=[None, "anthropic", "openai", "local", "echo"],
                         help="LLM backend; auto-detected from env vars if omitted")
    parser.add_argument("--top_k", type=int, default=4, help="Number of chunks retrieved per question")
    args = parser.parse_args()

    print("Loading embedding model ...")
    embedder = SentenceTransformerEmbedder()
    generator = build_generator(args.backend)

    pipeline = RAGPipeline(embedder, generator, top_k=args.top_k)
    pipeline.load(args.index)
    print(f"Loaded index: {len(pipeline.store)} chunks. Type a question, or 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        answer, sources = pipeline.query(question)
        print(f"\nAssistant: {answer}\n")
        print("Sources:")
        for chunk, score in sources:
            preview = chunk.text[:80].replace("\n", " ")
            print(f"  [{score:.2f}] {chunk.doc_id}#{chunk.chunk_index}: {preview}...")
        print()


if __name__ == "__main__":
    main()
