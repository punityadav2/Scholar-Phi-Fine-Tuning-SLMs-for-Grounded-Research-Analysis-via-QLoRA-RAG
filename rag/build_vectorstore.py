"""
Build Vector Store for RAG using FAISS
"""
import json
import pickle
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class VectorStoreBuilder:
    """Build and manage FAISS vector store for research papers"""
    
    def __init__(
        self, 
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vectorstore_dir: Path = None
    ):
        """
        Initialize vector store builder
        
        Args:
            embedding_model_name: Name of sentence transformer model
            vectorstore_dir: Directory to save vector store
        """
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vectorstore_dir = vectorstore_dir
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.chunks = []
        self.metadata = []
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def build_index(self, papers: List[Dict]):
        """
        Build FAISS index from processed papers
        
        Args:
            papers: List of processed paper dictionaries
        """
        print("Building vector store...")
        
        # Collect all chunks and metadata
        all_chunks = []
        all_metadata = []
        
        for paper in papers:
            for chunk in paper['chunks']:
                all_chunks.append(chunk['text'])
                
                # Store metadata
                metadata = {
                    'paper_title': paper['metadata'].get('paper_title', ''),
                    'filename': paper['metadata'].get('filename', ''),
                    'chunk_id': chunk['chunk_id'],
                }
                all_metadata.append(metadata)
        
        print(f"Total chunks: {len(all_chunks)}")
        
        # Create embeddings
        embeddings = self.create_embeddings(all_chunks)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.chunks = all_chunks
        self.metadata = all_metadata
        
        print(f"Vector store built with {self.index.ntotal} vectors")
    
    def save(self):
        """Save vector store to disk"""
        if self.vectorstore_dir is None:
            raise ValueError("vectorstore_dir not specified")
        
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = self.vectorstore_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        
        # Save chunks and metadata
        data_path = self.vectorstore_dir / "chunks_metadata.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata
            }, f)
        
        print(f"Vector store saved to {self.vectorstore_dir}")
    
    def load(self):
        """Load vector store from disk"""
        if self.vectorstore_dir is None:
            raise ValueError("vectorstore_dir not specified")
        
        # Load FAISS index
        index_path = self.vectorstore_dir / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))
        
        # Load chunks and metadata
        data_path = self.vectorstore_dir / "chunks_metadata.pkl"
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
        
        print(f"Vector store loaded from {self.vectorstore_dir}")
        print(f"Total vectors: {self.index.ntotal}")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with text, metadata, and score
        """
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):  # Valid index
                result = {
                    'text': self.chunks[idx],
                    'metadata': self.metadata[idx],
                    'score': float(distance),
                    'similarity': 1 / (1 + float(distance))  # Convert distance to similarity
                }
                results.append(result)
        
        return results


def main():
    """Main execution function"""
    from config import PROCESSED_DIR, VECTORSTORE_DIR, RAG_CONFIG
    
    # Load processed papers
    processed_path = PROCESSED_DIR / "processed_papers.json"
    
    if not processed_path.exists():
        print("Error: Processed papers not found!")
        print("Please run data_preparation.py first")
        return
    
    with open(processed_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"Loaded {len(papers)} processed papers")
    
    # Build vector store
    builder = VectorStoreBuilder(
        embedding_model_name=RAG_CONFIG['embedding_model'],
        vectorstore_dir=VECTORSTORE_DIR
    )
    
    builder.build_index(papers)
    builder.save()
    
    # Test search
    print("\n" + "=" * 50)
    print("Testing Search")
    print("=" * 50)
    
    test_query = "What is attention mechanism in transformers?"
    results = builder.search(test_query, top_k=3)
    
    print(f"\nQuery: {test_query}\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Paper: {result['metadata']['paper_title']}")
        print(f"  Similarity: {result['similarity']:.3f}")
        print(f"  Text: {result['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
