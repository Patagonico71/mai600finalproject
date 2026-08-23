# Methodology Draft

## Problem and approach

This project builds a local retrieval-augmented triage assistant for customer support
tickets. Given an incoming ticket, the system retrieves guidance from a knowledge base of
past resolutions and asks a small language model to propose a routing queue, a ticket
type, a priority, a draft first response, and citations. Everything runs on the analyst's
own laptop, because support tickets routinely contain customer names, phone numbers, and
account details that should not be sent to a hosted API.

## Data

The data is the public Customer IT Support ticket dataset (Kaggle, tobiasbueck / Open
Ticket AI), which is synthetic and carries no real customer information. The file holds
20,000 tickets in English and German. Only the English rows were kept, and rows missing a
subject, body, answer, queue, type, or priority were dropped, leaving 10,888 usable
tickets across 10 support queues.

20 tickets were held out as test cases, 2 from every one of
the 10 queues so that no queue is missing from the evaluation. The
remaining 10,868 tickets formed the pool used to build the knowledge base. The
split matters: if a test ticket's own answer sits in the corpus, the system retrieves
itself and every retrieval metric is meaningless.

## Knowledge base construction

Rather than indexing individual tickets, the corpus is 10 documents, one per
queue. Each document collects 14 past cases from that queue, giving
140 chunks in total. This matches how a help desk actually stores knowledge,
and it makes citations legible: the model cites a named queue playbook instead of an
opaque ticket id.

Two choices here decide most of the retrieval quality.

**What gets embedded is the customer's problem, not the agent's answer.** A first version
indexed the resolution text and retrieved poorly. Agent answers are roughly half courtesy
boilerplate -- apologies, requests for a convenient time to call -- and once that is in the
vector, similarity is driven by the tone of a support email rather than by the problem.
Every incoming ticket sounds like a frustrated technical complaint, so it matched the
boilerplate of any queue equally well. In one case the top-ranked chunk, at a similarity
of 0.797, contained no topical content whatsoever. Matching problem text against problem
text keeps the comparison inside a single register. The resolution is still carried
alongside each chunk and shown to the model as payload, so it can ground the draft
response; it simply does not participate in the search.

**Each past case is its own chunk.** A fixed-width word window straddles case boundaries
and glues the tail of one problem onto the head of an unrelated one, producing chunks that
represent nothing in particular.

## Retrieval

Chunks were embedded with the `all-MiniLM-L6-v2` sentence-transformer (384 dimensions) and
indexed in FAISS using inner product over L2-normalised vectors, which is cosine
similarity. Each query retrieves the top 5 chunks. Dense embeddings were chosen over
TF-IDF because customers describe problems in their own words, so lexical overlap is a
weak signal for this task.

The retrieved chunks are then combined into a single queue prediction by a score-weighted
vote: similarities are summed per queue and the highest total wins. Reading only the
top-ranked chunk is noisy, because one strongly matching case can outrank a queue that
placed three moderate matches.

## Generation and local serving

Models are served locally through Ollama, which exposes an HTTP API at
`localhost:11434`. Prompts are sent with `requests.post()` to `/api/generate` at
temperature 0.2. The hardware is a MacBook Pro with an Apple M4 Pro chip and 48 GB of
unified memory. No GPU cloud runtime was used and no data left the machine.

Two system versions were compared on the same 20 test cases:

- **V0 (baseline)** — `llama3.2:3b` with no retrieval.
- **V1 (RAG)** — `llama3.2:3b` with the retrieved context.

The same model serves both, so retrieval is the only variable that changes between them.
Both receive an identical output-format instruction, so answers can be parsed and
scored the same way. The list of valid queue names is deliberately withheld from the
baseline prompt. Knowing which queues an organisation actually operates is precisely the
local knowledge retrieval is meant to supply, so giving it to the baseline for free would
conceal the effect under measurement.

## Evaluation

Scoring is automatic and runs against the dataset's own labels, so the numbers reproduce
on a re-run. Retrieval hit rate at 3 asks whether the correct queue document appeared
among the retrieved chunks. Routing accuracy compares the predicted queue with the true
one. Queue validity checks whether the predicted queue exists at all in the taxonomy,
which separates a wrong answer from an invented one. Citation accuracy checks that the
supporting document id appears in the model's citation field. Type and priority accuracy
are exact matches against the labels. Response time is wall-clock seconds per generation.

## Tools

Python 3.13, pandas, numpy, scikit-learn, sentence-transformers, faiss-cpu, matplotlib,
requests, Jupyter, and Ollama.
