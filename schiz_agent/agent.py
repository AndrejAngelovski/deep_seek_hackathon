import logging
from vector_store import retrieve_relevant_chunks

logger = logging.getLogger(__name__)

try:
    # If available, import the Camel ChatAgent for advanced LLM integration.
    from camel_agent import ChatAgent
    BASE_AGENT_CLASS = ChatAgent
except ImportError:
    # Fallback to a basic object if camel_agent is not installed.
    BASE_AGENT_CLASS = object
    logger.info("camel_agent not found; using basic agent class.")

class SchizophreniaAgent(BASE_AGENT_CLASS):
    """
    A domain-specific agent for schizophrenia.
    
    If camel_agent is available, this agent extends it with a system message and goal.
    Otherwise, it uses a basic retrieval-augmented approach.
    """
    def __init__(self, knowledge_base=None, embeddings=None):
        # If using Camel's ChatAgent, initialize with a system message and goal.
        if BASE_AGENT_CLASS is not object:
            system_message = (
                "You are a domain expert in schizophrenia research. Provide concise, evidence-based responses."
            )
            goal = (
                "Answer questions about schizophrenia research, clinical practice, and related insights."
            )
            super().__init__(system_message=system_message, goal=goal)
        self.knowledge_base = knowledge_base
        self.embeddings = embeddings

    def retrieve_documents(self, query, top_k=5):
        """Retrieve relevant documents from the knowledge base using similarity search."""
        if not self.knowledge_base or not self.embeddings:
            logger.warning("Knowledge base or embeddings not set; cannot retrieve documents.")
            return []
        return retrieve_relevant_chunks(query, self.knowledge_base, self.embeddings, k=top_k)

    def get_response(self, query, top_k=5):
        """
        Retrieve context from the knowledge base and then (optionally) pass it to an LLM for synthesis.
        If using Camel's ChatAgent, call its respond() method.
        """
        docs = self.retrieve_documents(query, top_k=top_k)
        context = "\n".join([doc.page_content for doc in docs])
        combined_input = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        logger.info("Combined input for response generated.")
        
        # If using Camel's ChatAgent, delegate to its respond() method.
        if hasattr(super(), "respond"):
            return super().respond(combined_input)
        else:
            # Basic echo response (placeholder for integration with an LLM)
            response = f"[Retrieved context included]\n{combined_input}"
            return response
