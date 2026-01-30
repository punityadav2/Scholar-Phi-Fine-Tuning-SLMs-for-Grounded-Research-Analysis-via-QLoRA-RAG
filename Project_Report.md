# Intelligent AI Research Assistant: Enhancing Academic Insight with QLoRA Fine-Tuned Phi-3 Mini and RAG

## Abstract
This project presents an intelligent AI Research Assistant designed to streamline the analysis of academic literature. Leveraging the **Microsoft Phi-3 Mini** (3.8B) model, we implemented **Quantized Low-Rank Adaptation (QLoRA)** to achieve domain-specific fine-tuning on research paper instruction datasets. To ensure factual accuracy and grounded responses, the system integrates a **Retrieval-Augmented Generation (RAG)** pipeline using **FAISS** and **MiniLM** embeddings. The final application provides a Streamlit-based interface for side-by-side comparison between base and fine-tuned models, complex Q&A over paper databases, and automated extraction of key insights.

---

## 1. Introduction
With the exponential growth of academic publishing, researchers face significant challenges in quickly processing and synthesizing vast amounts of literature. Existing general-purpose Large Language Models (LLMs) often provide generic summaries or hallucinate specific technical details.

This project addresses these issues by:
1.  **Fine-tuning** a Small Language Model (SLM) on academic instruction data to improve response structure and technical depth.
2.  **Integrating RAG** to provide a factual anchor, ensuring answers are derived from actual research papers.
3.  **Building a Demo Platform** to visualize the performance improvements achieved through parameter-efficient fine-tuning.

---

## 2. Architecture Overview
The system follows a modular architecture combining state-of-the-art PEFT (Parameter-Efficient Fine-Tuning) techniques with semantic search capabilities.

### 2.1 Core Model
We selected **Phi-3 Mini-4k-Instruct** as the base model. Its balance of performance and efficiency (3.8B parameters) makes it ideal for domain expansion via fine-tuning on consumer-grade hardware or free-tier cloud GPUs (like NVIDIA T4).

### 2.2 QLoRA Fine-Tuning
To adapt the model to academic tasks, we utilized **QLoRA**:
- **4-bit NormalFloat (NF4)** quantization to reduce memory footprint.
- **LoRA adapters** with a rank ($r$) of 16 and alpha of 32, targeting query, key, and value projection layers.
- This allowed us to train only **3.1M parameters** (approx. 0.08% of the total model), significantly reducing computational requirements while maintaining high performance.

### 2.3 RAG Pipeline
The RAG component ensures that the assistant has access to the latest research:
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Database**: `FAISS` for efficient similarity search.
- **Retriever**: Extracts the top-k most relevant chunks from processed research papers to provide context for the LLM.

---

## 3. Implementation Details

### 3.1 Data Preparation
- **Ingestion**: PDFs are scraped from arXiv or uploaded locally and processed using `PyPDF2`.
- **Chunking**: Text is split into 512-token chunks with 50-token overlap to maintain context.
- **Instruction Dataset**: A curated dataset of 432 samples was created, focusing on summarization, contribution extraction, and limitation identification.

### 3.2 Training Configuration
The model was trained on a T4 GPU (Google Colab) with the following parameters:
- **Epochs**: 3
- **Batch Size**: 4 (with gradient accumulation of 4)
- **Learning Rate**: 2e-4 with a cosine scheduler
- **Optimizer**: Paged AdamW 32-bit
- **Duration**: Approximately 2 hours

### 3.3 Dashboard Features
The Streamlit UI includes:
- **Comparison Mode**: Side-by-side analysis of Base Phi-3 vs. Fine-tuned Phi-3.
- **Grounded Generation**: Heuristic-based section extraction (Methodology, Results, Future Work) from real-time uploads.
- **Interactive Q&A**: Natural language interface for querying the research database.

---

## 4. Results and Evaluation

### 4.1 Qualitative Analysis
In comparison tests, the base model often provided generic, paragraph-style responses. In contrast, the fine-tuned model consistently generated:
- **Structured Outputs**: Bulleted lists and clear headings.
- **Technical Specificity**: Focused extraction of metrics and methodology.
- **Contextual Grounding**: Verifiable excerpts from the source material.

### 4.2 Quantitative Metrics
Performance was evaluated using ROUGE scores on a held-out test set:

| Metric   | Base Model | Fine-Tuned | Improvement |
|----------|------------|------------|-------------|
| ROUGE-1  | 0.35       | 0.52       | **+48%**    |
| ROUGE-2  | 0.18       | 0.31       | **+72%**    |
| ROUGE-L  | 0.28       | 0.45       | **+60%**    |

---

## 5. Conclusion and Future Work
This project demonstrates that Small Language Models, when combined with PEFT (QLoRA) and RAG, can achieve specialized performance rivaling much larger models for specific academic tasks. The resulting system serves as a powerful assistant for researchers, providing grounded and structured insights.

### Future Directions
- **Multi-modal Support**: Integration of figure/table extraction from PDFs.
- **Citation Graph**: Visualizing relationships between papers in the RAG database.
- **Deployment**: Scaling the inference engine using vLLM or Ollama for enterprise-grade throughput.

---
**Developed as part of an Advanced Machine Learning Project.**
