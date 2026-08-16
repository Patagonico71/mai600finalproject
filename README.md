# Local RAG Assistant for Customer Support Ticket Triage

MAI 600 — Module 6 Final Project (Proposal milestone)
Approach: **Local SLM + RAG** — Llama 3.2 (3B) served with Ollama, retrieval with FAISS and Sentence-Transformers.

## Overview

Support teams read every incoming ticket, route it to a queue, set a priority, and write a first reply — by hand, which is slow and inconsistent. This project builds an assistant that runs on the agent's own machine and, for each new ticket, retrieves the most similar resolved tickets and drafts a triage (suggested queue, type, priority, and a first-response reply) with citations back to the tickets it used. Running locally keeps ticket text in-house instead of sending it to a cloud API.

This repository currently holds the **proposal milestone**: it defines the problem, the data, the approach, the baseline, the evaluation plan, and the risks. The full system is built in the next milestone.

## Approach

- **Generation:** Llama 3.2 (3B) via Ollama, running locally.
- **Retrieval:** past resolved tickets embedded with `all-MiniLM-L6-v2` (Sentence-Transformers) and searched with a FAISS index.
- **Grounding:** for a new ticket, the top-k similar tickets are passed as context; every factual part of the draft cites the ticket it came from.
- **Baseline:** the same model prompted zero-shot, with no retrieval, so the RAG version can be measured against it.

## Dataset

*Customer IT Support — Ticket Dataset* (tobiasbueck / Open Ticket AI), public and synthetic, from Kaggle:
https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets

20,000 tickets across several languages; this project uses the English subset (~12,800). Each row has a subject, body, agent answer, type, queue, priority, language, and up to eight tags.

The raw dataset is **not committed** to this repo (size and Kaggle licensing). Download it from the link above and place it under `data/`. A small sample (`data/sample_tickets.csv`) is included so the notebook can run end to end without the full download.

## Repository structure

```
.
├── README.md                     # this file
├── proposal.md                   # the 14-section proposal (main deliverable)
├── docs/
│   ├── AI_USAGE_DISCLOSURE.md     # how AI tools were used and verified
│   └── REFERENCES.md             # sources and links
├── data/
│   ├── sample_tickets.csv        # small sample (safe to commit)
│   └── .gitkeep                  # full dataset goes here, not committed
├── notebook			  # the python code
├── results			   # the results of the RAG
└── images			  # images of the process and result

```


## How to run the prototype

1. Install Ollama: https://ollama.com
2. Pull the model:
   ```
   ollama pull llama3.2:3b
   ```
3. Install the Python dependencies:
   ```
   pip install pandas sentence-transformers faiss-cpu requests
   ```
4. Open `notebook/ollama_rag_starter.ipynb` and point it at your local dataset (or the included sample). It chunks the tickets, builds the FAISS index, retrieves similar tickets, and calls the local Ollama model to generate a cited triage.

Ollama must be running locally (it listens on `http://localhost:11434`) before you run the retrieval-and-generation cells.

## Evaluation plan

Three RAG metrics plus two that fit a local triage assistant: retrieval hit rate, citation accuracy, groundedness, routing accuracy (queue / type / priority vs. the real labels), and response time. Targets and measurement details are in `proposal.md`, Section 11.

## Project status

Module 6 — proposal milestone. The next milestone builds and evaluates the system described here.

## AI usage and academic integrity

See `docs/AI_USAGE_DISCLOSURE.md`. All data used is public and synthetic; no real, confidential, or protected information is included.
