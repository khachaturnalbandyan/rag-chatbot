"""
End-to-end pipeline tests using HashEmbedder + EchoGenerator (see
src/embeddings.py, src/generator.py) so the whole ingest -> retrieve ->
generate wiring is exercised with NO network call and NO model download -
these must run in any sandbox/CI with just numpy installed.

They test that the pipeline is WIRED correctly (right chunk retrieved,
right thing handed to the generator), not that retrieval quality is good -
HashEmbedder has no real semantic understanding, so that's what
SentenceTransformerEmbedder is for in real use (see README).
"""
from src.embeddings import HashEmbedder
from src.generator import EchoGenerator
from src.rag import RAGPipeline


def _make_pipeline():
    return RAGPipeline(HashEmbedder(dim=32), EchoGenerator(), chunk_size=20, chunk_overlap=5, top_k=2)


def test_ingest_text_and_retrieve():
    pipeline = _make_pipeline()
    pipeline.ingest_text("the quick brown fox jumps over the lazy dog " * 3, doc_id="animals")
    pipeline.ingest_text("interest rates and inflation affect the stock market " * 3, doc_id="finance")

    results = pipeline.retrieve("fox dog jumps", top_k=1)
    assert len(results) == 1
    assert results[0][0].doc_id == "animals"


def test_query_returns_context_via_echo_generator():
    pipeline = _make_pipeline()
    pipeline.ingest_text("HRDF-MER uses a cross-modal adaptive alignment module called CAA.", doc_id="paper")

    answer, sources = pipeline.query("What is CAA?")
    assert "CAA" in answer   # EchoGenerator returns the prompt verbatim, context included
    assert len(sources) >= 1
    assert sources[0][0].doc_id == "paper"


def test_query_with_no_ingested_documents():
    pipeline = _make_pipeline()
    answer, sources = pipeline.query("anything")
    assert sources == []
    assert "No documents" in answer


def test_ingest_files(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("gated fusion adaptively weights modalities in the CMDF module. " * 5)

    pipeline = _make_pipeline()
    pipeline.ingest_files([str(f)])
    assert len(pipeline.store) > 0

    answer, sources = pipeline.query("What does the gated fusion do?")
    assert "gated fusion" in answer.lower()


def test_save_and_load_index_roundtrip(tmp_path):
    pipeline = _make_pipeline()
    pipeline.ingest_text("some content about attention mechanisms and transformers", doc_id="doc")
    out = str(tmp_path / "index")
    pipeline.save(out)

    reloaded = _make_pipeline()
    reloaded.load(out)
    assert len(reloaded.store) == len(pipeline.store)

    answer, sources = reloaded.query("attention mechanisms")
    assert len(sources) >= 1
