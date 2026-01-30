"""
Inference Module - Generate responses using Phi-3
"""
import torch
from pathlib import Path
from typing import Dict, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from inference.model_loader import ModelLoader
from config import GENERATION_CONFIG


class InferenceEngine:
    """Generate responses using Phi-3 model"""
    
    def __init__(
        self,
        model_loader: ModelLoader,
        generation_config: Dict = None
    ):
        """
        Initialize inference engine
        
        Args:
            model_loader: Loaded ModelLoader instance
            generation_config: Generation parameters
        """
        self.model_loader = model_loader
        self.generation_config = generation_config or GENERATION_CONFIG
        
        if not model_loader.is_loaded:
            raise ValueError("Model must be loaded before inference")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """
        Generate response for prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Prepare generation config
        gen_config = self.generation_config.copy()
        
        if max_new_tokens is not None:
            gen_config['max_new_tokens'] = max_new_tokens
        if temperature is not None:
            gen_config['temperature'] = temperature
        
        gen_config.update(kwargs)
        
        # Tokenize input
        inputs = self.model_loader.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        
        # Move to model device
        inputs = {k: v.to(self.model_loader.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model_loader.model.generate(
                **inputs,
                **gen_config,
                pad_token_id=self.model_loader.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.model_loader.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Extract only the new generated text (remove prompt)
        response = generated_text[len(prompt):].strip()
        
        return response
    
    def chat(
        self,
        messages: list,
        **kwargs
    ) -> str:
        """
        Chat-style generation with message history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Generation parameters
            
        Returns:
            Generated response
        """
        # Format messages into prompt
        prompt = self._format_chat_prompt(messages)
        return self.generate(prompt, **kwargs)
    
    def _format_chat_prompt(self, messages: list) -> str:
        """Format chat messages into prompt"""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)


class ResearchAssistant:
    """High-level interface for research paper assistant"""
    
    def __init__(
        self,
        model_loader: ModelLoader,
        use_rag: bool = True,
        rag_pipeline = None
    ):
        """
        Initialize research assistant
        
        Args:
            model_loader: Loaded model
            use_rag: Whether to use RAG
            rag_pipeline: RAG pipeline instance
        """
        self.inference_engine = InferenceEngine(model_loader)
        self.use_rag = use_rag
        self.rag_pipeline = rag_pipeline
    
    def answer_question(
        self,
        question: str,
        use_rag: bool = None
    ) -> Dict:
        """
        Answer a question about research papers
        
        Args:
            question: User question
            use_rag: Override RAG usage
            
        Returns:
            Response dictionary
        """
        use_rag_flag = use_rag if use_rag is not None else self.use_rag
        
        if use_rag_flag and self.rag_pipeline:
            # Use RAG pipeline
            result = self.rag_pipeline.generate_response(
                question,
                task_type="qa"
            )
            
            # Generate with model
            prompt = result['prompt']
            response = self.inference_engine.generate(prompt)
            
            result['response'] = response
            return result
        else:
            # Direct generation without RAG
            response = self.inference_engine.generate(question)
            return {
                'query': question,
                'response': response,
                'sources': [],
                'num_sources': 0
            }
    
    def summarize(self, paper_context: str = None) -> str:
        """Summarize research paper"""
        if self.use_rag and self.rag_pipeline and not paper_context:
            result = self.rag_pipeline.summarize_paper()
            prompt = result['prompt']
        else:
            prompt = f"Summarize the following research:\n\n{paper_context}"
        
        return self.inference_engine.generate(prompt)
    
    def extract_contributions(self, paper_context: str = None) -> str:
        """Extract key contributions"""
        if self.use_rag and self.rag_pipeline and not paper_context:
            result = self.rag_pipeline.extract_contributions()
            prompt = result['prompt']
        else:
            prompt = f"Extract key contributions from:\n\n{paper_context}"
        
        return self.inference_engine.generate(prompt)


def test_inference():
    """Test inference engine"""
    print("=" * 70)
    print("INFERENCE ENGINE TEST")
    print("=" * 70)
    
    print("\nLoading model...")
    loader = ModelLoader(use_4bit=True)
    
    try:
        loader.load_base_model()
        
        engine = InferenceEngine(loader)
        
        # Test simple generation
        print("\nTest Generation:")
        print("-" * 70)
        
        prompt = "Explain what a transformer model is in one sentence."
        print(f"Prompt: {prompt}\n")
        
        response = engine.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        
        print(f"Response: {response}")
        print("\n✓ Inference test successful")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nNote: Inference requires GPU or will be very slow on CPU")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_inference()
