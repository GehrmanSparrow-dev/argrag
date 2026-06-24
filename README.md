<div align="center">

# ArgRAG: Argument-Aware Retrieval-Augmented Generation

### Structured Reasoning for Explainable Natural-Language Processing

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)](LICENSE)
[![BEIR](https://img.shields.io/badge/Benchmark-BEIR_FEVER-orange?style=for-the-badge)](https://github.com/beir-cellar/beir)
[![HuggingFace](https://img.shields.io/badge/🤗_Models-HuggingFace-yellow?style=for-the-badge)](https://huggingface.co)

**Surya Teja Anupindi** · Saint Peter's University, Data Science Institute

[📄 Research Paper](#research-paper) · [🚀 Quick Start](#quick-start) · [📊 Results](#results) · [🏗️ Architecture](#architecture)

</div>

---

## Overview

**ArgRAG** is a novel Retrieval-Augmented Generation (RAG) framework that inserts an explicit **argumentation layer** between retrieval and generation. Rather than treating retrieved passages as an unstructured "bag of text," ArgRAG:

1. **Classifies** each retrieved passage as *supportive*, *adversarial*, or *neutral* relative to the input claim
2. **Aggregates** these stance-annotated arguments via a model ensemble (BM25 + DPR + ColBERT + SPLADE; BERT + RoBERTa + DeBERTa)
3. **Prompts** a generator (FLAN-T5-XXL / GPT-3.5) to reason over the structured argument set before producing a final verdict and natural-language justification

This approach directly addresses two critical weaknesses of vanilla RAG: **conflicting evidence handling** and **opaque explanations** — making it well-suited for high-stakes domains such as fact-checking, legal analysis, and medical decision support.

---

## Key Results

| System | Verdict EM | Verdict F1 | Recall@10 | Stance F1 | Explanation BLEU | Human Plausibility ↑ |
|---|---|---|---|---|---|---|
| Vanilla RAG | 71.2% | 73.8% | 62.5% | — | 21.4% | 2.8 / 5 |
| RAG-Hybrid | 73.1% | 75.6% | 66.3% | — | 23.1% | 3.1 / 5 |
| CoT-RAG | 74.0% | 76.4% | 66.8% | — | 24.5% | 3.2 / 5 |
| ArgRAG (No-Ensemble) | 74.2% | 76.7% | 68.5% | 69.4% | 27.8% | 3.6 / 5 |
| **ArgRAG (Full)** | **77.6%** | **80.1%** | **71.4%** | **71.2%** | **31.3%** | **4.5 / 5** |
| ArgRAG + GPT-3.5 | 78.2% | 80.5% | 71.0% | 71.0% | 32.0% | 4.7 / 5 |

> Bold = statistically significant improvement over strongest baseline (paired t-test, p < 0.01)

**Highlights:**
- ↑ **3.4%** factual accuracy over vanilla RAG baseline
- ↑ **21%** human-rated explanation quality
- ↓ **ECE 0.043** vs. baseline 0.087 (better calibrated confidence scores)
- **~397 ms** end-to-end latency — feasible for interactive applications

---

## Architecture

```
Input Claim/Query
       │
       ▼
┌─────────────────────────────────┐
│       Retrieval Ensemble        │
│  BM25 · DPR · ColBERT · SPLADE │
│     → Reciprocal Rank Fusion    │
│         Top-N Passages          │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Argument Construction         │
│  BERT · RoBERTa · DeBERTa-v3   │
│   → Weighted Voting (Stance)    │
│  {SUPPORT, ATTACK, NEUTRAL}     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Structured Generation       │
│  FLAN-T5-XXL / GPT-3.5-Turbo   │
│  → Verdict + Justification      │
└─────────────────────────────────┘
```

The pipeline is **fully modular** — any retriever, stance classifier, or generator can be swapped without retraining the full system.

---

## Quick Start

### Prerequisites

```bash
pip install beir sentence-transformers faiss-cpu transformers torch
```

> **Note:** GPU is strongly recommended for the stance classifier ensemble. All experiments were run on 4× NVIDIA A100 (40 GB).

### Run the Notebook

```bash
git clone https://github.com/YOUR_USERNAME/argrag.git
cd argrag
jupyter notebook Applied_Research_Final_Project.ipynb
```

The notebook will automatically download the BEIR-FEVER dataset on first run.

### Minimal Example

```python
from argrag import ArgRAGPipeline

pipeline = ArgRAGPipeline(
    retrievers=["bm25", "dpr", "colbert"],
    stance_models=["bert-base", "roberta-large", "deberta-v3"],
    generator="flan-t5-xxl"
)

verdict, justification = pipeline.run(
    claim="The Eiffel Tower is located in Paris."
)

print(verdict)        # SUPPORTED
print(justification)  # "Passage 1 states the tower is on the Champ de Mars in Paris..."
```

---

## Repository Structure

```
argrag/
├── Applied_Research_Final_Project.ipynb   # Full implementation & experiments
├── thesis_results_table.csv               # Quantitative results (all metrics)
├── Research_Paper_Surya_Teja_Anupindi.pdf # Published research paper
└── README.md
```

---

## Experimental Setup

| Component | Detail |
|---|---|
| Knowledge Corpus | Wikipedia (2023-01 dump) |
| Evaluation Benchmark | BEIR-FEVER (19K claims, 1M evidence sentences) |
| Stance Fine-tune Data | SciFact + FEVER (~30K examples) |
| Retrieval Index | FAISS (GPU-accelerated, ~3h build time) |
| Generator Fine-tune | Synthetic argument-aware dataset (~150K examples) |
| Hardware | 4× NVIDIA A100 (40 GB) |
| Seed | 42 (results reported as mean ± std over 3 runs) |

### Retrieval Latency Breakdown

| Component | Avg. Latency |
|---|---|
| Retrieval Ensemble (RRF) | 190 ms |
| Stance Classification (3-model voting) | 75 ms |
| Prompt Construction | 12 ms |
| Generation (FLAN-T5-XXL) | 120 ms |
| **Total** | **~397 ms** |

---

## Ablation Highlights

**Retrieval Ensemble vs. Single Retriever:**

| Setup | Verdict EM | Recall@10 |
|---|---|---|
| BM25 only | 71.3% | 58.9% |
| DPR only | 73.9% | 62.1% |
| ColBERT only | 73.2% | 63.7% |
| **RRF (4-retriever ensemble)** | **77.6%** | **71.4%** |

**Stance Classifier Ensemble:**

| Model | Stance F1 | Verdict F1 |
|---|---|---|
| BERT-base (single) | 64.1% | 75.1% |
| RoBERTa-large (single) | 66.8% | 76.3% |
| DeBERTa-v3 (single) | 68.5% | 77.2% |
| **Weighted Voting (3-model)** | **71.2%** | **80.1%** |

---

## Research Paper

> **Argument-Aware Retrieval-Augmented Generation (ArgRAG): Structured Reasoning for Explainable Natural-Language Processing**  
> Surya Teja Anupindi  
> Saint Peter's University, Data Science Institute  

The full paper (`Research_Paper_Surya_Teja_Anupindi.pdf`) covers:
- Full ArgRAG framework specification and algorithms
- Ablation studies across retrieval depth, stance models, and prompt variants
- Qualitative analysis with real conflicting-evidence examples
- Error analysis across 200 manually inspected failure cases
- Discussion of limitations and broader impacts (including EU AI Act alignment)

---

## Citation

If you use ArgRAG in your research, please cite:

```bibtex
@article{anupindi2024argrag,
  title     = {Argument-Aware Retrieval-Augmented Generation (ArgRAG):
               Structured Reasoning for Explainable Natural-Language Processing},
  author    = {Anupindi, Surya Teja},
  institution = {Saint Peter's University, Data Science Institute},
  year      = {2024}
}
```

---

## License

This project is released under the [Apache 2.0 License](LICENSE) to enable reproducibility and further research at the intersection of retrieval, argument mining, and explainable generation.

---

<div align="center">

*Built at Saint Peter's University · Data Science Institute · Jersey City, NJ*

</div>
