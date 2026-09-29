"""
src.generator
==============
LLM generation backends behind one interface (`generate(prompt) -> str`).
The RAG pipeline builds a prompt (question + retrieved context) and hands
it to whichever backend is configured - swapping providers never touches
rag.py, ingest.py, or chat.py.

Backends:
- AnthropicGenerator: calls the Claude API (ANTHROPIC_API_KEY).
- OpenAIGenerator: calls the OpenAI API (OPENAI_API_KEY).
- LocalHFGenerator: runs a small open-weights instruct model locally via
  transformers - no API key, no internet at query time (after the one-time
  model download), free. Noticeably lower quality than a frontier hosted
  model, but a genuinely useful zero-cost/offline fallback and a good demo
  of local-inference tradeoffs.
- EchoGenerator: returns the prompt's retrieved context verbatim with no
  actual generation. Used ONLY by the test suite, so pipeline tests never
  need an API key or a model download.
"""
from __future__ import annotations
import os


SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a specific document. "
    "Use ONLY the provided context to answer. If the context doesn't contain the "
    "answer, say so plainly instead of guessing. Be concise."
)


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return (
        f"Context from the document:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}\n\nAnswer based only on the context above:"
    )


class Generator:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class AnthropicGenerator(Generator):
    def __init__(self, model: str = "claude-3-5-haiku-20241022", api_key: str | None = None, max_tokens: int = 500):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIGenerator(Generator):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None, max_tokens: int = 500):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


class LocalHFGenerator(Generator):
    """Zero-cost, offline fallback: a small instruct model run locally.
    Default is Qwen2.5-1.5B-Instruct (~3GB download, runs fine on CPU or a
    consumer GPU). Quality is noticeably below a hosted frontier model -
    that tradeoff is the point of offering this backend, not a bug."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.device = device
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        import torch
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            output = self.model.generate(inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = output[0][inputs.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class EchoGenerator(Generator):
    """Test-only stand-in: returns the context verbatim, no LLM call at all.
    Lets the pipeline's retrieval-then-generation wiring be tested without
    a network call, an API key, or a model download."""

    def generate(self, prompt: str) -> str:
        return prompt


def build_generator(backend: str | None = None) -> Generator:
    """Picks a backend from an explicit name, or auto-detects from
    environment variables: ANTHROPIC_API_KEY -> Anthropic,
    OPENAI_API_KEY -> OpenAI, otherwise falls back to the local HF model."""
    backend = backend or os.environ.get("RAG_GENERATOR_BACKEND")
    if backend == "anthropic" or (backend is None and os.environ.get("ANTHROPIC_API_KEY")):
        return AnthropicGenerator()
    if backend == "openai" or (backend is None and os.environ.get("OPENAI_API_KEY")):
        return OpenAIGenerator()
    if backend == "local":
        return LocalHFGenerator()
    if backend == "echo":
        return EchoGenerator()
    print("[generator] No ANTHROPIC_API_KEY or OPENAI_API_KEY found - falling back to a local "
          "Hugging Face model (Qwen2.5-1.5B-Instruct). This downloads ~3GB on first run and is "
          "noticeably lower quality than a hosted model; set an API key for better answers.")
    return LocalHFGenerator()
