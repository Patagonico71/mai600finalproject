# Final Project Proposal — AI Solution Design with Local SLM + RAG

**Course:** MAI 600 — Module 6
**Approach:** Local SLM + RAG
**Model / tooling:** Llama 3.2 (3B) served locally with Ollama, retrieval with FAISS and Sentence-Transformers

---

## 1. Project Title

Local RAG Assistant for Customer Support Ticket Triage (Ollama + Llama 3.2)

## 2. Problem Definition

A customer support team receives a large stream of tickets every day. For each one, an agent has to read the message, decide which team or queue it belongs to, set a priority, and write a first reply. Done by hand this is slow and uneven — two agents can route and prioritize the same ticket differently, and newer agents take longer because they don't yet know how similar cases were handled before. That handling knowledge exists, but it sits buried across thousands of past tickets and nobody re-reads them.

This project proposes an assistant that runs on the agent's own machine and, for each new ticket, pulls up the most similar resolved tickets and drafts a triage: a suggested queue, a ticket type, a priority, and a first-response reply, with citations back to the tickets it used. Because it runs locally, ticket text stays in-house — in a real deployment that content can include customer and internal details, which is exactly what teams don't want sent to a cloud API.

If the problem goes unsolved, response times stay high, routing and priority stay inconsistent, and the knowledge in past tickets keeps going to waste. A working solution would cut handling time, make routing and priority more consistent, and leave a traceable trail of which past cases informed each decision.

## 3. Project Relevance

This sits in customer support and IT service management. High ticket volume and steady agent turnover make consistency hard to maintain, and onboarding a new agent is slow. A local triage assistant helps on three fronts: it speeds up routing, it keeps the criteria steady between experienced and new agents, and — by running locally — it avoids shipping ticket text to outside services, which is a real compliance concern in support work. The professional payoff is measurable: shorter time-to-first-response and fewer mis-routed or mis-prioritized tickets.

## 4. Dataset or Document Collection

The project uses a public Kaggle dataset: *Customer IT Support — Ticket Dataset* (published by tobiasbueck / Open Ticket AI), a fully synthetic help-desk ticket set. The full file holds **20,000 tickets** across multiple languages; this project filters to the **English subset (~12,800 tickets, about 64%)**.

Each record is one support ticket and includes a subject, the customer's message body, the agent's answer, a ticket type, a queue, a priority, a language field, and up to eight topic tags.

The data is safe to use because it is public and synthetic — it contains no real customers. Names, phone numbers, and similar details appear only as placeholder tokens such as `<name>` and `<tel_num>`. There are no confidentiality or privacy concerns, since nothing traces back to a real person or organization. In a real-world version of this system, the same local-inference design is what would keep genuine ticket data protected.

## 5. Background and Data Description

The domain is multi-channel customer and IT support. One row represents a single ticket together with the response an agent gave it.

Important fields:

- **subject** — a short title for the ticket (about 6 words on average).
- **body** — the customer's message (about 57 words on average).
- **answer** — the agent's reply, which holds the resolution or the first-response request for more detail (about 60 words on average). This field is what the retrieval step draws on.
- **type** — the ticket class, ITIL-style: Incident, Request, Problem, or Change.
- **queue** — the routing destination, one of ten categories (for example Technical Support, Billing and Payments, Product Support).
- **priority** — low, medium, or high.
- **tag_1 … tag_8** — free topic tags such as Security, Outage, Billing, Hardware.

Known limitations of the data:

- It is synthetic, so phrasing may not match a real support inbox.
- A share of the answers are first-response clarifications ("please provide more details") rather than final fixes, so grounding can be thin for some tickets.
- The subject is missing in roughly 6% of English rows.
- The file mixes languages, so a filtering step is required.

Preprocessing plan: filter to `language == en`; repair or drop rows with a missing subject; strip placeholder tokens; select a manageable subset of tickets to serve as the retrieval knowledge base; and hold out a separate set of tickets as an evaluation set the system never sees during retrieval.

## 6. Proposed AI Approach

**Selected approach: Local SLM + RAG.**

The local small language model (Llama 3.2 3B, served with Ollama) generates the triage. Before it generates anything, a retrieval step finds the most similar resolved tickets in the dataset and feeds them in as context. Embeddings come from the `all-MiniLM-L6-v2` Sentence-Transformers model, and vector search runs on a FAISS index — the same pipeline used in the course starter notebook.

## 7. Approach Justification

Plain prompting is not enough here, because a good triage depends on how similar past tickets were routed and answered, and that context can't fit into one prompt when there are thousands of tickets to draw from. Retrieval is what supplies the right handful of past cases at question time.

RAG is needed because the value is in grounding: the draft should point back to real tickets the agent can check, not to the model's guesswork. Fine-tuning is not part of the first version — the task is retrieval and routing, not teaching the model a new writing style — though it could be a sensible later step if a consistent house format becomes the goal.

A local model earns its place because, in a real support setting, ticket text is sensitive; local inference keeps it in-house, works offline, and carries no per-call cost.

The trade-off accepted is that a 3B model running on a laptop is slower and less polished than a hosted API model, and retrieval quality has to be tested rather than assumed. Two limitations are expected up front: some answers in the data are generic first-responses, so grounding will be thin for those tickets, and retrieval will struggle when an incoming ticket looks unlike anything in the knowledge base.

## 8. Expected System Output

For each incoming ticket, the system produces a structured triage record:

| Suggested Queue | Type | Priority | Draft First Response | Evidence | Source Citation |
|---|---|---|---|---|---|

Every factual part of the draft carries a citation such as `[1]` or `[2]` that points to the retrieved ticket it came from. When retrieval turns up nothing relevant, the system is expected to say the knowledge base does not contain enough information rather than invent an answer. The output can be rendered as a markdown table or as JSON.

## 9. Baseline Model or Method

**Baseline:** Llama 3.2 3B prompted zero-shot — the ticket goes in, a triage comes out, with no retrieval.

**Improved method:** the same local model with RAG retrieval over the ticket knowledge base.

The comparison checks whether retrieval improves routing accuracy (does the suggested queue, type, and priority match the real labels?), groundedness, and citation quality against the no-retrieval baseline.

## 10. Initial EDA / Data Review

The review below is drawn from a representative 2,000-ticket sample of the file, with the full-file count confirmed separately.

| Item | Result |
|---|---|
| Total tickets (full file) | 20,000 |
| English tickets (used) | ~12,800 (about 64%) |
| Columns | 15 (subject, body, answer, type, queue, priority, language, 8 tags) |
| Queues / routing categories | 10 (Technical Support is the largest) |
| Ticket types | 4: Incident, Request, Problem, Change |
| Priority levels | 3: low, medium, high |
| Average body length | ~57 words (median 52) |
| Average answer length | ~60 words (median 58) |
| Average subject length | ~6 words |
| Missing values | subject missing in ~6% of English rows; body / answer / queue / type / priority complete |
| Duplicate tickets | none |
| Expected chunks | ~1 chunk per ticket answer (answers are short); e.g. ~2,000 chunks for a 2,000-ticket KB subset |
| Sensitive data present? | No — synthetic; names and phone numbers are placeholders |

Main topics, from the queues and tags: technical issues, account and IT support, billing and payments, product questions, service outages, returns, and sales inquiries.

## 11. Evaluation Metrics

The plan uses the three metrics expected for a RAG project, plus two that fit a local triage assistant.

| Metric | How it will be measured | Target |
|---|---|---|
| Retrieval hit rate | For a set of held-out test tickets with known queue/topic, whether a retrieved neighbor shares the correct queue in the top-3 results | 80% or higher |
| Citation accuracy | Whether the cited tickets actually support the drafted response (human score, 1–5) | 4/5 average or higher |
| Groundedness | Whether the draft avoids claims not supported by the retrieved tickets (human score, 1–5) | 4/5 average or higher |
| Routing accuracy | Suggested queue / type / priority compared against the ticket's real labels | Queue 70%+; priority above baseline |
| Response time | Seconds per ticket generated locally on the laptop | Reported; usable for interactive review |

## 12. AI Tools Usage Plan

Ollama runs Llama 3.2 3B locally for generation. Sentence-Transformers (`all-MiniLM-L6-v2`) produces the embeddings, and FAISS stores them for vector search. pandas handles the EDA and preprocessing. The course Colab starter notebook serves as the pipeline template.

Claude will be used to brainstorm evaluation questions and assist with the development of the project. ChatGPT will be used for language correction, including translating phrases from Spanish to English, correcting grammatical errors, and improving the readability and clarity of English sentences. All technical decisions, labels, citations, and evaluation scores will be reviewed and verified by me. No real, confidential, or protected data is used at any stage.

## 13. Risks, Limitations, and Ethics

The main risks are hallucinated next steps, a wrong queue or priority, and over-reliance — an agent trusting a draft without reading the original ticket. Grounding can also be thin when the retrieved answers are generic first-responses, and a synthetic dataset may not match the shape of real support traffic.

Mitigations: the data is public and synthetic; every factual claim is cited so it can be checked; a human agent stays in the loop and approves any reply before it is sent; the system takes no automated action on a ticket; the evaluation set deliberately includes cases with no good match to test whether the system correctly says it lacks enough information; and priority and routing suggestions are treated as drafts, not decisions.

## 14. References or Source Links

- Customer IT Support — Ticket Dataset (Kaggle, tobiasbueck / Open Ticket AI)
- Ollama documentation
- FAISS documentation
- Sentence-Transformers documentation
- MAI 600 Module 6 Ollama + RAG Colab starter notebook (course-provided.  ajusted for running in Jupiter notebook)

---

Appendix — AI Usage Disclosure

The full AI usage disclosure and academic integrity statement are in a separate document: docs/AI_USAGE_DISCLOSURE.md.
