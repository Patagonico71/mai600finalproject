# References and Source Links

For: MAI 600 — Module 6 Final Project Proposal
Project: Local RAG Assistant for Customer Support Ticket Triage

## Dataset

- **Customer IT Support — Ticket Dataset** (tobiasbueck / Open Ticket AI). Public, synthetic customer/IT support tickets with agent answers. Kaggle.
  https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets

## Model and serving

- **Ollama** — local model serving used to run the small language model.
  https://ollama.com
- **Llama 3.2 (3B)** — the small language model used for generation.
  https://ollama.com/library/llama3.2

## Retrieval stack

- **Sentence-Transformers** — embedding library used to vectorize tickets.
  https://www.sbert.net
- **all-MiniLM-L6-v2** — the specific embedding model.
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **FAISS** — vector similarity search used for the retrieval index.
  https://github.com/facebookresearch/faiss


## Course materials

- MAI 600 — Module 6 Ollama + RAG Colab starter notebook (course-provided).

*Note: URLs were current at the time of writing. If any project page has moved, search the project or dataset name to find the maintained location.*
