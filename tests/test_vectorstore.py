import numpy as np

from src.chunking import Chunk
from src.vectorstore import VectorStore


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_search_returns_most_similar_first():
    store = VectorStore()
    chunks = [Chunk(text="about cats", doc_id="d", chunk_index=0),
              Chunk(text="about dogs", doc_id="d", chunk_index=1),
              Chunk(text="about cars", doc_id="d", chunk_index=2)]
    vectors = np.stack([_unit([1, 0, 0]), _unit([0.9, 0.1, 0]), _unit([0, 0, 1])])
    store.add(vectors, chunks)

    results = store.search(_unit([1, 0, 0]), top_k=2)
    assert len(results) == 2
    assert results[0][0].text == "about cats"
    assert results[0][1] > results[1][1]   # best match has higher cosine sim


def test_top_k_larger_than_store_size_is_clamped():
    store = VectorStore()
    chunks = [Chunk(text="only one", doc_id="d", chunk_index=0)]
    store.add(np.stack([_unit([1, 0])]), chunks)
    results = store.search(_unit([1, 0]), top_k=10)
    assert len(results) == 1


def test_empty_store_returns_no_results():
    store = VectorStore()
    assert store.search(_unit([1, 0]), top_k=5) == []
    assert len(store) == 0


def test_add_accumulates_across_calls():
    store = VectorStore()
    store.add(np.stack([_unit([1, 0])]), [Chunk(text="a", doc_id="d", chunk_index=0)])
    store.add(np.stack([_unit([0, 1])]), [Chunk(text="b", doc_id="d", chunk_index=1)])
    assert len(store) == 2


def test_save_and_load_roundtrip(tmp_path):
    store = VectorStore()
    chunks = [Chunk(text="hello world", doc_id="d.txt", chunk_index=0),
              Chunk(text="goodbye world", doc_id="d.txt", chunk_index=1)]
    vectors = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    store.add(vectors, chunks)

    out_path = str(tmp_path / "idx")
    store.save(out_path)
    loaded = VectorStore.load(out_path)

    assert len(loaded) == 2
    assert loaded.chunks[0].text == "hello world"
    assert np.allclose(loaded.vectors, store.vectors)

    results = loaded.search(_unit([1, 0, 0]), top_k=1)
    assert results[0][0].text == "hello world"
