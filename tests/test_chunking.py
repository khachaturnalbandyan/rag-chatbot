from src.chunking import chunk_text


def test_basic_chunking_produces_expected_count():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, doc_id="doc1", chunk_size=30, overlap=10)
    # step = 20 words/chunk; words 0-29, 20-49, 40-69, 60-89, 80-99(partial)
    assert len(chunks) == 5
    assert chunks[0].text.split()[0] == "word0"
    assert chunks[-1].text.split()[-1] == "word99"


def test_overlap_actually_overlaps():
    text = " ".join(f"word{i}" for i in range(60))
    chunks = chunk_text(text, doc_id="doc1", chunk_size=30, overlap=10)
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    overlap_words = first_words & second_words
    assert len(overlap_words) == 10


def test_short_text_yields_one_chunk():
    chunks = chunk_text("just a few words here", doc_id="doc1", chunk_size=180, overlap=40)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_empty_text_yields_no_chunks():
    assert chunk_text("", doc_id="doc1") == []
    assert chunk_text("   ", doc_id="doc1") == []


def test_doc_id_and_chunk_index_are_set():
    text = " ".join(f"w{i}" for i in range(50))
    chunks = chunk_text(text, doc_id="paper.txt", chunk_size=20, overlap=5)
    for i, c in enumerate(chunks):
        assert c.doc_id == "paper.txt"
        assert c.chunk_index == i


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("some words here", doc_id="doc1", chunk_size=10, overlap=10)
