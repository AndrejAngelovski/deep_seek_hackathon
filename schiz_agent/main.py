import os
import logging
from knowledge_base import load_documents, chunk_documents, save_document_chunks, load_document_chunks
from vector_store import build_or_load_index
from agent import SchizophreniaAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    # Set file paths and directories.
    data_dir = "data/txt_files/"
    document_chunks_file = "data/document_chunks.json"
    index_path = "data/faiss_index"

    # Step 1 & 2: Load documents and create document chunks if not already saved.
    if not os.path.exists(document_chunks_file):
        logger.info("Loading and processing raw documents...")
        documents = load_documents(data_dir)
        document_chunks = chunk_documents(documents)
        save_document_chunks(document_chunks, document_chunks_file)
    else:
        logger.info("Loading preprocessed document chunks...")
        document_chunks = load_document_chunks(document_chunks_file)

    # Step 3: Build or load the FAISS index.
    index, embeddings = build_or_load_index(document_chunks, index_path)

    # Step 4: Initialize the agent with the knowledge base and embeddings.
    agent = SchizophreniaAgent(knowledge_base=index, embeddings=embeddings)
    logger.info("Schizophrenia Agent initialized. Ready to answer queries.")

    # Interactive CLI loop.
    print("Enter your queries below (type 'exit' to quit):")
    while True:
        query = input("Query: ").strip()
        if query.lower() == "exit":
            print("Exiting application. Goodbye!")
            break
        response = agent.get_response(query)
        print("\nResponse:\n", response)

if __name__ == "__main__":
    main()
