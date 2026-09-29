# RAG Chatbot — Chat With a Document

A Retrieval-Augmented Generation (RAG) chatbot: ask questions about a
document in plain English, get answers grounded in that document's actual
content, with the exact source passages cited alongside every answer.

Ships with a demo corpus — the [HRDF-MER](https://doi.org/10.1016/j.csl.2026.101978)
multimodal emotion recognition paper — so it's runnable out of the box, but
the pipeline works on any `.txt` or `.pdf` you point it at.

## Why this exists

RAG is the standard pattern for grounding an LLM's answers in a specific,
private, or up-to-date knowledge source instead of relying on what the model
memorized during training. This repo is a from-first-principles
implementation of that pattern: no LangChain/LlamaIndex black box — every
stage (chunking, embedding, vector search, prompt construction, generation)
is a small, readable, independently-tested module, so the design tradeoffs
at each step are explicit rather than hidden behind a framework default.

## How it works

```
document (.txt/.pdf)
        │
        ▼
  chunking.py    — split into overlapping ~180-word chunks
        │
        ▼
 embeddings.py   — embed each chunk (sentence-transformers, local, free)
        │
        ▼
vectorstore.py   — store as a matrix; cosine-similarity search at query time
        │
        ▼
   (at query time)
  question ──► embed ──► top-k most similar chunks ──► generator.py ──► answer
                                                       (Claude / GPT / local LLM)
```

- **Chunking** (`src/chunking.py`): fixed-size sliding window with overlap,
  so a sentence that straddles a chunk boundary is still fully retrievable
  from whichever side gets matched.
- **Embeddings** (`src/embeddings.py`): `sentence-transformers`'
  `all-MiniLM-L6-v2` by default — small (~80MB), fast, runs locally on CPU
  or GPU, no API cost per query. The `Embedder` interface means swapping in
  a different model (or a hosted embeddings API) touches one file.
- **Vector store** (`src/vectorstore.py`): a plain NumPy cosine-similarity
  matmul, not FAISS/Chroma. At demo scale (hundreds of chunks) brute-force
  search is exact and sub-millisecond — pulling in an approximate-NN index
  here would trade correctness for a speed advantage that doesn't exist yet
  at this scale. The interface is the same shape a FAISS-backed store would
  have, so swapping it in later (for a much larger corpus) doesn't touch
  `rag.py`.
- **Generation** (`src/generator.py`): one interface, three real backends —
  Anthropic (Claude), OpenAI (GPT), or a fully local/offline Hugging Face
  model (`Qwen2.5-1.5B-Instruct`) for a zero-cost, no-API-key fallback. The
  backend is auto-detected from whichever API key is set in the
  environment.
- **Pipeline** (`src/rag.py`): wires the above into `ingest()` /
  `retrieve()` / `query()`.

## Setup

```bash
pip install -r requirements.txt
```

(Skip the `torch`/`transformers` lines in `requirements.txt` if you only
plan to use a hosted API backend, not the local-model fallback.)

Copy `.env.example` to `.env` and add **one** API key (or neither, to use
the free local-model fallback):

```bash
cp .env.example .env
# then edit .env: ANTHROPIC_API_KEY=sk-ant-... or OPENAI_API_KEY=sk-...
```

## Usage

**1. Build the index** (one-time, or whenever the source documents change):

```bash
python ingest.py --data_dir data --index_out index/paper_index
```

This embeds every `.txt`/`.pdf` in `data/` and saves the resulting vectors
to `index/paper_index.npz` + `.json`. Re-run it if you add your own
documents to `data/`.

**2. Chat, from the terminal:**

```bash
python chat.py --index index/paper_index
```

```
You: What is the CMDF module?
Assistant: CMDF (Cascaded Multimodal Dynamic Fusion) is a progressive
cross-modal interaction architecture that employs a hierarchical attention
mechanism to integrate semantic-acoustic features...

Sources:
  [0.71] hrdfmer_paper.txt#12: 3.3 Cascaded Multimodal Dynamic Fusion (CMDF)...
  [0.58] hrdfmer_paper.txt#3: propose a Cascaded Multimodal Dynamic Fusion...
```

**3. Or chat from a web UI:**

```bash
streamlit run app_streamlit.py
```

**4. Use your own document instead of the demo paper:**

```bash
cp /path/to/your/document.pdf data/
python ingest.py --data_dir data --index_out index/my_index
python chat.py --index index/my_index
```

## Evaluation

`eval/evaluate.py` is a lightweight **retrieval**-quality check: 10 known
questions about the demo paper, each with expected keywords that should
appear in the retrieved chunks. This isolates retrieval quality from
generation quality — the standard first place to debug a RAG pipeline when
answers look wrong, since a wrong answer is usually a retrieval miss, not a
generation failure.

```bash
python eval/evaluate.py --index index/paper_index
```

## Tests

```bash
pytest tests/ -v
```

16 tests, all offline (no network, no API key, no model download): chunking
correctness (chunk counts, overlap, edge cases), vector store behavior
(similarity ranking, save/load round-trip, empty-store handling), and full
pipeline wiring (ingest → retrieve → generate) using a deterministic
hash-based embedder and an echo generator as test-only stand-ins for the
real (network-dependent) `SentenceTransformerEmbedder` and LLM backends.

## Design choices worth calling out

- **No framework** (LangChain/LlamaIndex): at this scope, a framework adds
  an abstraction layer without saving meaningful code, and hides exactly
  the design decisions (chunk size/overlap, similarity metric, prompt
  construction) that are worth being able to point to and reason about
  directly.
- **Local embeddings, pluggable generation**: embedding every chunk through
  a hosted API would add cost and a network dependency to a step that a
  small local model handles well; generation quality benefits more from a
  larger hosted model, so that's where the API dependency is worth taking.
- **Retrieval-only evaluation**: judging the LLM's final answer quality is
  a harder, more subjective problem (semantic correctness, tone, hallucination
  detection). Checking retrieval first is cheaper and catches the more common
  failure mode.

## Limitations

- `all-MiniLM-L6-v2` is a small, general-purpose embedding model — retrieval
  quality on jargon-heavy technical text (like the demo paper's equations
  and table data) is decent but not as strong as a domain-tuned or larger
  embedding model would give.
- The demo corpus (`data/hrdfmer_paper.txt`) is a plain-text extraction of
  the PDF; extracted mathematical notation is imperfect (e.g. subscripts
  rendered inline), which can make equation-heavy questions harder to
  retrieve precisely than prose questions.
- No re-ranking step: the top-k cosine-similarity results are used directly.
  A cross-encoder re-ranker on top of the initial retrieval is a common
  next improvement for retrieval precision.
- The local-model fallback (`--backend local`) is meaningfully lower
  quality than a hosted frontier model — it exists for a zero-cost/offline
  path, not as the recommended default.

## License

MIT — see [LICENSE](LICENSE). The bundled demo document is a third-party
research paper excerpt, included as sample content only; see LICENSE for
details.
