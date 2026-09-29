"""
app_streamlit.py
==================
A minimal web UI over the same RAGPipeline used by chat.py - useful for a
live demo (e.g. in an interview) instead of a terminal.

Usage:
    streamlit run app_streamlit.py
"""
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.embeddings import SentenceTransformerEmbedder
from src.generator import build_generator
from src.rag import RAGPipeline

st.set_page_config(page_title="RAG chatbot", page_icon="🔍")
st.title("Chat with a document")
st.caption("RAG demo: retrieval over a locally embedded document, answered by an LLM.")


@st.cache_resource
def load_pipeline(index_path: str):
    embedder = SentenceTransformerEmbedder()
    generator = build_generator()
    pipeline = RAGPipeline(embedder, generator)
    pipeline.load(index_path)
    return pipeline


index_path = st.sidebar.text_input("Index path (no extension)", value="index/paper_index")

try:
    pipeline = load_pipeline(index_path)
    st.sidebar.success(f"Loaded index: {len(pipeline.store)} chunks")
except FileNotFoundError:
    st.sidebar.error(f"No index found at {index_path}.npz - run `python ingest.py` first.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.write(content)

question = st.chat_input("Ask a question about the document...")
if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            answer, sources = pipeline.query(question)
        st.write(answer)
        with st.expander("Retrieved sources"):
            for chunk, score in sources:
                st.markdown(f"**[{score:.2f}] {chunk.doc_id}#{chunk.chunk_index}**")
                st.text(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))
    st.session_state.history.append(("assistant", answer))
