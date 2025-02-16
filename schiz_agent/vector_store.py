import os
import logging
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

logger = logging.getLogger(__name__)

def build_or_load_index(document_chunks, index_path="data/faiss_index", embedding_model="sentence-transformers/all-mpnet-base-v2"):
    """
    Generate embeddings for each text chunk and build or load a FAISS index.
    Returns both the FAISS index and the embeddings object.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    texts = [doc["text"] for doc in document_chunks]

    if os.path.exists(index_path):
        logger.info("Loading existing FAISS index...")
        index = FAISS.load_local(index_path, embeddings)
    else:
        logger.info("Creating new FAISS index...")
        index = FAISS.from_texts(texts, embeddings)
        index.save_local(index_path)
        logger.info(f"FAISS index saved to '{index_path}'.")
    return index, embeddings

def retrieve_relevant_chunks(query, index, embeddings, k=5):
    """
    Given a user query, compute its embedding and perform a similarity search on the FAISS index.
    Returns the top-k relevant document chunks.
    """
    query_embedding = embeddings.embed_query(query)
    results = index.similarity_search_by_vector(query_embedding, k=k)
    return results
