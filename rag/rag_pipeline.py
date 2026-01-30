"""
Complete RAG Pipeline
"""
from pathlib import Path
from typing import Dict, List
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rag.retriever import Retriever
from config import PROMPT_TEMPLATES


class RAGPipeline:
    """End-to-end RAG pipeline for research paper Q&A"""
    
    def __init__(
        self,
        retriever: Retriever,
        model_inference_fn=None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            retriever: Retriever instance
            model_inference_fn: Function that takes prompt and returns response
        """
        self.retriever = retriever
        self.model_inference_fn = model_inference_fn
    
    def build_prompt(
        self, 
        query: str, 
        context: str, 
        task_type: str = "qa"
    ) -> str:
        """
        Build prompt from template
        
        Args:
            query: User query
            context: Retrieved context
            task_type: Type of task (qa, summarize, etc.)
            
        Returns:
            Formatted prompt
        """
        template = PROMPT_TEMPLATES.get(task_type, PROMPT_TEMPLATES['qa'])
        
        if task_type == "qa":
            prompt = template.format(context=context, question=query)
        else:
            prompt = template.format(context=context)
        
        return prompt
    
    def generate_response(
        self,
        query: str,
        task_type: str = "qa",
        top_k: int = None
    ) -> Dict:
        """
        Generate response using RAG
        
        Args:
            query: User query
            task_type: Type of task
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with response and metadata
        """
        # Retrieve context
        context, results = self.retriever.retrieve_and_format(query, top_k)
        
        # Build prompt
        prompt = self.build_prompt(query, context, task_type)
        
        # Generate response
        if self.model_inference_fn:
            response = self.model_inference_fn(prompt)
        else:
            response = "[Model not loaded - showing retrieved context only]"
        
        return {
            'query': query,
            'response': response,
            'context': context,
            'sources': results,
            'num_sources': len(results),
            'prompt': prompt
        }
    
    def summarize_paper(self, paper_title: str = None) -> Dict:
        """
        Summarize a research paper
        
        Args:
            paper_title: Optional paper title to focus on
            
        Returns:
            Summary response
        """
        query = f"Summarize the paper {paper_title}" if paper_title else "Summarize this research"
        return self.generate_response(query, task_type="summarize")
    
    def extract_contributions(self, paper_title: str = None) -> Dict:
        """Extract key contributions"""
        query = f"Key contributions of {paper_title}" if paper_title else "Key contributions"
        return self.generate_response(query, task_type="key_contributions")
    
    def identify_limitations(self, paper_title: str = None) -> Dict:
        """Identify limitations"""
        query = f"Limitations of {paper_title}" if paper_title else "Limitations"
        return self.generate_response(query, task_type="limitations")
    
    def suggest_future_work(self, paper_title: str = None) -> Dict:
        """Suggest future research directions"""
        query = f"Future work for {paper_title}" if paper_title else "Future research"
        return self.generate_response(query, task_type="future_work")


def test_rag_pipeline():
    """Test RAG pipeline without model"""
    from config import VECTORSTORE_DIR, RAG_CONFIG
    
    # Initialize retriever
    retriever = Retriever(
        vectorstore_dir=VECTORSTORE_DIR,
        embedding_model_name=RAG_CONFIG['embedding_model'],
        top_k=RAG_CONFIG['top_k']
    )
    
    # Initialize pipeline (without model for now)
    pipeline = RAGPipeline(retriever=retriever)
    
    # Test different tasks
    print("=" * 70)
    print("RAG PIPELINE TEST")
    print("=" * 70)
    
    # Test Q&A
    print("\n1. Question Answering")
    print("-" * 70)
    result = pipeline.generate_response(
        "What is the attention mechanism in transformers?",
        task_type="qa"
    )
    print(f"Query: {result['query']}")
    print(f"Sources found: {result['num_sources']}")
    print(f"Context preview: {result['context'][:300]}...")
    
    # Test summarization
    print("\n2. Summarization")
    print("-" * 70)
    result = pipeline.summarize_paper()
    print(f"Query: {result['query']}")
    print(f"Sources found: {result['num_sources']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_rag_pipeline()
