"""
Configuration file for AI Research Paper Assistant
"""
import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Directory Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_PAPERS_DIR = DATA_DIR / "raw_papers"
PROCESSED_DIR = DATA_DIR / "processed"
FINETUNE_DATASET_PATH = DATA_DIR / "finetune_dataset.json"

RAG_DIR = PROJECT_ROOT / "rag"
VECTORSTORE_DIR = RAG_DIR / "vectorstore"

TRAINING_DIR = PROJECT_ROOT / "training"
CHECKPOINTS_DIR = TRAINING_DIR / "checkpoints"

INFERENCE_DIR = PROJECT_ROOT / "inference"
ADAPTERS_DIR = INFERENCE_DIR / "adapters"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"

# Model Configuration
MODEL_CONFIG = {
    "base_model": "microsoft/Phi-3-mini-4k-instruct",
    "model_max_length": 4096,
    "device_map": "auto",
}

# LoRA Configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# Training Configuration
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 50,
    "fp16": True,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_32bit",
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
}

# RAG Configuration
RAG_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5,
    "similarity_threshold": 0.7,
}

# Generation Configuration
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
}

# Dataset Configuration
DATASET_CONFIG = {
    "arxiv_categories": ["cs.AI", "cs.CL", "cs.LG", "cs.NE"],
    "max_papers": 30,
    "max_results_per_category": 8,
    "train_test_split": 0.9,
}

# Prompt Templates
PROMPT_TEMPLATES = {
    "summarize": """Based on the following research paper content, provide a concise summary in simple terms.

Context:
{context}

Question: Summarize this research paper.

Answer:""",
    
    "key_contributions": """Based on the following research paper content, extract the key contributions.

Context:
{context}

Question: What are the key contributions of this research?

Answer:""",
    
    "limitations": """Based on the following research paper content, identify the limitations.

Context:
{context}

Question: What are the limitations of this research?

Answer:""",
    
    "future_work": """Based on the following research paper content, suggest future research directions.

Context:
{context}

Question: What are potential future research directions?

Answer:""",
    
    "qa": """Based on the following research paper content, answer the question.

Context:
{context}

Question: {question}

Answer:""",
}

# Streamlit UI Configuration
UI_CONFIG = {
    "page_title": "AI Research Paper Assistant",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Create directories if they don't exist
for directory in [
    RAW_PAPERS_DIR, PROCESSED_DIR, VECTORSTORE_DIR,
    CHECKPOINTS_DIR, ADAPTERS_DIR, RESULTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
