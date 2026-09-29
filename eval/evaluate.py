"""
eval/evaluate.py
==================
A simple RETRIEVAL-quality check: for each question in eval_questions.json,
retrieve the top-k chunks and check whether the expected keywords appear
anywhere in the retrieved text. This measures whether the retriever is
finding the right part of the document - it does NOT grade the LLM's final
answer (that's a separate, harder, more subjective evaluation - judging
retrieval quality first is the standard, cheaper place to start debugging a
RAG pipeline when answers look wrong).

Usage (run from the repo root):
    python eval/evaluate.py --index index/paper_index
"""
import argparse
import json
import os

from src.embeddings import SentenceTransformerEmbedder
from src.generator import EchoGenerator
from src.rag import RAGPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index/paper_index")
    parser.add_argument("--questions", default=os.path.join(os.path.dirname(__file__), "eval_questions.json"))
    parser.add_argument("--top_k", type=int, default=4)
    args = parser.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        cases = json.load(f)

    embedder = SentenceTransformerEmbedder()
    pipeline = RAGPipeline(embedder, EchoGenerator(), top_k=args.top_k)
    pipeline.load(args.index)

    hits, total = 0, len(cases)
    for case in cases:
        results = pipeline.retrieve(case["question"], top_k=args.top_k)
        retrieved_text = " ".join(c.text for c, _ in results).lower()
        found = [kw for kw in case["expected_keywords"] if kw.lower() in retrieved_text]
        ok = len(found) == len(case["expected_keywords"])
        hits += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['question']}")
        if not ok:
            missing = [kw for kw in case["expected_keywords"] if kw not in found]
            print(f"         missing keywords: {missing}")

    print(f"\nRetrieval hit rate: {hits}/{total} ({100 * hits / total:.0f}%)")


if __name__ == "__main__":
    main()
