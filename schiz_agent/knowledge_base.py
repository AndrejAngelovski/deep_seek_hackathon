import os
import json
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_documents(data_dir="data/txt_files/"):
    """
    Load all .txt files from the given directory.
    Each document is stored along with its filename as metadata.
    """
    documents = []
    if not os.path.exists(data_dir):
        logger.error(f"Data directory '{data_dir}' does not exist.")
        return documents

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append({"filename": filename, "text": text})
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")
    logger.info(f"Loaded {len(documents)} documents from '{data_dir}'.")
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split each document into manageable chunks using a recursive splitter.
    Retain the filename metadata with each chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks:
            document_chunks.append({
                "filename": doc["filename"],
                "text": chunk
            })
    logger.info(f"Created {len(document_chunks)} document chunks.")
    return document_chunks

def save_document_chunks(document_chunks, filepath="data/document_chunks.json"):
    """
    Save the document chunks to a JSON file for future reuse.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(document_chunks, f, indent=2)
        logger.info(f"Document chunks saved to '{filepath}'.")
    except Exception as e:
        logger.error(f"Error saving document chunks: {e}")

def load_document_chunks(filepath="data/document_chunks.json"):
    """
    Load preprocessed document chunks from a JSON file.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            document_chunks = json.load(f)
        logger.info(f"Loaded {len(document_chunks)} document chunks from '{filepath}'.")
        return document_chunks
    except Exception as e:
        logger.error(f"Error loading document chunks: {e}")
        return []
