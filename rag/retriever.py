"""
Retriever Module for RAG
"""
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.build_vectorstore import VectorStoreBuilder


class Retriever:
    """Retrieve relevant context from vector store"""
    
    def __init__(
        self, 
        vectorstore_dir: Path,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize retriever
        
        Args:
            vectorstore_dir: Directory containing vector store
            embedding_model_name: Name of embedding model
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Load vector store
        self.vector_store = VectorStoreBuilder(
            embedding_model_name=embedding_model_name,
            vectorstore_dir=vectorstore_dir
        )
        self.vector_store.load()
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant chunks for query
        
        Args:
            query: Search query
            top_k: Override default top_k
            
        Returns:
            List of relevant chunks with metadata
        """
        k = top_k if top_k is not None else self.top_k
        
        # Search vector store
        results = self.vector_store.search(query, top_k=k)
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in results 
            if r['similarity'] >= self.similarity_threshold
        ]
        
        return filtered_results
    
    def format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved chunks into context string
        
        Args:
            results: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            paper_title = result['metadata'].get('paper_title', 'Unknown')
            text = result['text']
            
            context_parts.append(
                f"[Source {i}: {paper_title}]\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def retrieve_and_format(self, query: str, top_k: int = None) -> tuple:
        """
        Retrieve and format context in one step
        
        Args:
            query: Search query
            top_k: Override default top_k
            
        Returns:
            Tuple of (formatted_context, raw_results)
        """
        results = self.retrieve(query, top_k)
        context = self.format_context(results)
        return context, results


def test_retriever():
    """Test retriever functionality"""
    from config import VECTORSTORE_DIR, RAG_CONFIG
    
    # Initialize retriever
    retriever = Retriever(
        vectorstore_dir=VECTORSTORE_DIR,
        embedding_model_name=RAG_CONFIG['embedding_model'],
        top_k=RAG_CONFIG['top_k'],
        similarity_threshold=RAG_CONFIG['similarity_threshold']
    )
    
    # Test queries
    test_queries = [
        "What is attention mechanism?",
        "How do transformers work?",
        "What are the limitations of neural networks?",
    ]
    
    print("=" * 70)
    print("RETRIEVER TEST")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        context, results = retriever.retrieve_and_format(query, top_k=3)
        
        print(f"Retrieved {len(results)} chunks\n")
        print(context)
        print("=" * 70)


if __name__ == "__main__":
    test_retriever()
