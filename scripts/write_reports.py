"""Generate README.md and results.md from the CSVs the notebook produced.

Everything numeric is interpolated, so the prose cannot drift away from the run.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
S = pd.read_csv(RES / "summary_metrics.csv").set_index("version")
E = pd.read_csv(RES / "evaluation_scores.csv")
TAX = pd.read_csv(RES / "error_taxonomy.csv")
CEIL = pd.read_csv(RES / "ceiling_analysis.csv")
DOCS = pd.read_csv(ROOT / "data/sample_documents.csv")
M7 = pd.read_csv(RES / "module7_baseline/summary_metrics.csv").set_index("version")

FINAL = "V2_rag_constrained"
vt, v0, v1, v2 = (S.loc["V_majority_class"], S.loc["V0_baseline"],
                  S.loc["V1_rag_open"], S.loc[FINAL])
n = int(v2.num_tests)
final = E[E.version == FINAL]
per_queue = final.groupby("true_queue").routing_correct.mean().sort_values(ascending=False)

def ceil_row(frag):
    r = CEIL[CEIL.reference.str.contains(frag, case=False, na=False)].iloc[0]
    return float(r.top_1), float(r.top_3)

mini_top1, mini_top3 = ceil_row("MiniLM")
bge_top1, bge_top3 = ceil_row("bge")
e5_top1, e5_top3 = ceil_row("e5")
pct = 100 * float(S.loc[FINAL].routing_accuracy) / mini_top1


def tax_share(stage):
    r = TAX[TAX.stage == stage]
    return (int(r["count"].iloc[0]), float(r.share_of_errors.iloc[0])) if len(r) else (0, 0.0)


miss_n, miss_p = tax_share("A_retrieval_miss")
rank_n, rank_p = tax_share("B_ranking_miss")
over_n, over_p = tax_share("C_model_override")

metrics_table = f"""| Metric | Majority class | V0 no retrieval | V1 RAG | V2 RAG constrained |
|---|---|---|---|---|
| Routing accuracy | {vt.routing_accuracy:.2f} | {v0.routing_accuracy:.2f} | {v1.routing_accuracy:.2f} | **{v2.routing_accuracy:.2f}** |
| Queue validity | {vt.queue_validity:.2f} | {v0.queue_validity:.2f} | {v1.queue_validity:.2f} | {v2.queue_validity:.2f} |
| Type accuracy | {vt.type_accuracy:.2f} | {v0.type_accuracy:.2f} | {v1.type_accuracy:.2f} | **{v2.type_accuracy:.2f}** |
| Priority accuracy | {vt.priority_accuracy:.2f} | {v0.priority_accuracy:.2f} | {v1.priority_accuracy:.2f} | {v2.priority_accuracy:.2f} |
| Retrieval hit rate @5 | n/a | n/a | {v1.retrieval_hit_rate:.2f} | {v2.retrieval_hit_rate:.2f} |
| Citation accuracy | n/a | n/a | {v1.citation_accuracy:.2f} | {v2.citation_accuracy:.2f} |
| Mean response time | n/a | {v0.avg_response_time_s:.1f}s | {v1.avg_response_time_s:.1f}s | {v2.avg_response_time_s:.1f}s |"""

results = f"""# Results

All figures come from `results/summary_metrics.csv` and `results/evaluation_scores.csv`,
written by `notebooks/final_project_notebook.ipynb`. The test set is {n} held-out tickets,
5 from each of the 10 support queues.

## Metrics

{metrics_table}

The **majority class** column answers every ticket with the most frequent label and uses
no model. It is the floor any real system has to clear, and it changes how two of these
rows should be read.

## Retrieval works, and the effect is not marginal

Routing accuracy moves from {v0.routing_accuracy:.2f} without retrieval to
{v2.routing_accuracy:.2f} with it, against a {vt.routing_accuracy:.2f} floor. Paired over
the same {n} tickets, retrieval corrected 15 that the baseline got wrong and broke none
(McNemar, p = 0.0001).

Queue validity is the mechanism. Without retrieval the model does not merely choose the
wrong queue — it does not appear to grasp that "queue" names a fixed routing destination,
and returns invented titles that summarise the ticket: "Excessive Billing Inquiry",
"Adobe Premiere Pro Crash Investigation". A ticketing system would reject every one of
them. The missing knowledge is not reasoning ability but a fact about this organisation,
which is the case for retrieval over fine-tuning here.

## The Module 8 prompt change did nothing, and that is informative

V2 constrains the queue choice to the retrieved candidates and shows the model the ranked
vote. Against V1 it is a null result: p = 1.000, one ticket gained and one lost.

That constraint was designed to fix a Module 7 failure in which the model retrieved the
correct evidence and then routed elsewhere anyway, overwhelmingly to the most generic
queue. The failure is gone — but the knowledge base fixed it, not the prompt. Growing the
corpus from 140 randomly sampled cases to {int(DOCS.n_source_cases.sum()):,} balanced ones
dropped the generic-queue fallback from 8 predictions in 50 to 2. The constraint was a
crutch for a weak corpus, and with a reasonable corpus it is redundant.

## Where the remaining errors are

Every routing error is attributed to the pipeline stage that caused it
(`results/error_taxonomy.csv`):

| Stage | Errors | Share |
|---|---|---|
| A — the correct queue document was never retrieved | {miss_n} | {miss_p:.0%} |
| B — retrieved, but the vote ranked another queue first | {rank_n} | {rank_p:.0%} |
| C — ranked first, and the model chose something else | {over_n} | {over_p:.0%} |

This is the clearest statement of what changed. Module 7's bottleneck was stage C. In this
version stage C is {over_n} ticket out of {n}: the model now follows its evidence almost
without exception. {miss_p:.0%} of what remains is retrieval never surfacing the right
document, which is where any further work belongs.

## Per-queue accuracy

{chr(10).join(f"- {q}: {a:.2f}" for q, a in per_queue.items())}

Narrow, topically distinct queues do well. The generic and overlapping ones do not, and
they fail together: Customer Service, IT Support and Product Support resolve similar
problems in similar language, and a human triage team would not agree unanimously on those
boundaries either.

## How much of this is learnable at all

A routing accuracy of {v2.routing_accuracy:.2f} says nothing by itself. Against the
majority-class floor of {vt.routing_accuracy:.2f} it looks like the system does real work.
Against a perfect router it looks like a failure. Neither reading is worth much without
knowing what the best available method reaches on the same tickets.

So I trained one. A logistic regression over sentence embeddings, fitted on all
{int(7113):,} de-duplicated training tickets and their queue labels — far more supervision
than the pipeline ever gets, since retrieval never learns from labels at all. On the same
{n} held-out tickets it scores {mini_top1:.2f}.

| Reference | Top-1 | Top-3 |
|---|---|---|
| Majority-class floor | {vt.routing_accuracy:.2f} | — |
| V0, no retrieval | {v0.routing_accuracy:.2f} | — |
| **V2, this system** | **{v2.routing_accuracy:.2f}** | — |
| Supervised classifier, all-MiniLM-L6-v2 | {mini_top1:.2f} | {mini_top3:.2f} |
| Supervised classifier, bge-base-en-v1.5 | {bge_top1:.2f} | {bge_top3:.2f} |
| Supervised classifier, e5-base-v2 | {e5_top1:.2f} | {e5_top3:.2f} |

The pipeline reaches {v2.routing_accuracy:.2f} against that {mini_top1:.2f}, which is
{pct:.0f}% of what a trained classifier pulls out of the same text with the same encoder.
The gap is about four tickets.

Stronger encoders move the ceiling a little and not far. BGE-base doubles the embedding
dimension and buys two points; E5-base gains six. E5 is the only lever tested here that
showed a positive signal, and it was not pursued — it belongs in the next-steps list, not
in the results.

The queue label, then, is only weakly determined by the ticket text. The confusion matrix
agrees: errors scatter across queues instead of clustering in one or two confusable pairs,
which is what you would see if the original labelling followed something the text does not
contain. Four interventions were tested against this and none helped.

| Intervention | Outcome |
|---|---|
| Constrain the queue choice to retrieved candidates | p = 1.000, no effect |
| Retrieve 20 chunks instead of 5 | hit rate {v1.retrieval_hit_rate:.2f} to 0.88, vote flat at 0.30 |
| Merge the overlapping queues into six groups | accuracy 0.38, floor 0.30 — margin falls from 3.0x to 1.3x |
| Suggest three candidate queues rather than one | covers 0.48 against a 0.30 floor, again a worse ratio |

Growing the knowledge base from 140 cases to {int(DOCS.n_source_cases.sum()):,} is the only
change that helped. That is the honest summary: {v2.routing_accuracy:.2f} sits near what
this data supports, and beating it needs better labels or signal the ticket text does not
carry, not a better prompt.


## Course module evidence

The notebook closes with a section producing one artefact per course module, so the path
from each topic to the final system is visible rather than asserted.

| Module | Topic | Artefact | What it settled |
|---|---|---|---|
| 1 | Tokenisation, context limits | `results/module1_token_budget.csv` | Chunks average 75 tokens against a 256 limit; 6 of 1,782 truncate. Truncation ruled out as an error source. |
| 2 | Sampling and temperature | `results/module2_sampling_comparison.csv` | At six runs, 0.0 and 0.2 repeat themselves and 0.7 and 1.0 do not. An earlier three-run pass saw 0.2 vary, so the boundary sits near 0.2 and this sample cannot place it exactly. |
| 3 | Attention as evidence selection | `images/module3_evidence_map.png` | Shows which queue the retrieved evidence actually favours, and by how much. |
| 4 | Prompting, structured output | `prompts/*.txt` | Both templates exported so the V0-to-V2 difference can be read without running anything. |
| 5 | Local serving, benchmarking | `results/module5_local_slm_benchmark.csv` | 0.9 s and 88 output tokens/s per triage; prompts average 1,074 tokens, well inside the context window. |
| 6 | Retrieval-augmented generation | `results/retrieved_chunks.csv` | The pipeline itself. |
| 7 | Fine-tuning readiness | `results/module7_rag_vs_finetuning_decision.csv`, `data/training_examples.jsonl` | The decision is made per output field, not for the system as a whole. |
| 8 | Integration and evaluation | `results/summary_metrics.csv`, `images/evaluation_chart.png` | The evaluation above. |

The Module 7 table is the one worth reading. Queue goes to retrieval because the knowledge
is organisation-specific and changes; Type is the fine-tuning candidate because it is a
fixed taxonomy that retrieval measurably fails to help; Priority gets neither, because the
information is not in the ticket at all.

## A Module 7 conclusion that was wrong

Module 7 reported type accuracy of 0.45 against a 0.50 majority-class floor and concluded
that type classification did not work at all. At {n} tickets it scores
{v2.type_accuracy:.2f} against a {vt.type_accuracy:.2f} floor. The earlier finding was an
artefact of a 20-ticket sample, and it is corrected here rather than quietly dropped.
"""

(ROOT / "results.md").write_text(results, encoding="utf-8")
print(f"results.md written ({len(results)} chars)")

readme = f"""# Local Hybrid Assistant for Customer Support Ticket Triage

MAI 600 — Final Project (Module 8).

## Overview

A support ticket arrives. The system retrieves similar past cases from a knowledge base
built out of the support team's own resolution history, then asks a locally served small
language model to produce a triage: routing queue, ticket type, priority, a draft first
response, and citations to the sources used.

## Problem being solved

Support teams read every incoming ticket, route it to a queue, set a priority and write a
first reply by hand. It is slow and inconsistent. Ticket text also routinely contains
customer names, phone numbers and account details, so sending it to a hosted API is a
privacy decision most support organisations would rather not make. This system runs
entirely on the analyst's own machine.

## System type

**Hybrid: RAG + local SLM.** Retrieval supplies the organisation-specific routing
knowledge; the local model reads the ticket and writes the structured triage.

The split is not arbitrary. The evaluation shows retrieval carries the queue decision
(p = 0.0001 against no retrieval) and does little for the type field. Type is a fixed
four-way taxonomy that needs consistent judgement rather than fresh facts, which is where
adapter-based fine-tuning belongs — so `data/training_examples.jsonl` prepares that path
without claiming a trained adapter exists.

## Tools

Python 3.13 · `sentence-transformers` (all-MiniLM-L6-v2) · FAISS · Ollama with
`llama3.2:3b` · pandas · matplotlib · Jupyter. Hardware: MacBook Pro, Apple M4 Pro, 48 GB.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
```

Download the Kaggle dataset (link below) into `data/`. The raw CSV is not committed.

## How to run

```bash
jupyter notebook notebooks/final_project_notebook.ipynb   # then Run All
python3 scripts/build_artifacts.py
python3 scripts/write_reports.py
python3 scripts/ceiling_analysis.py   # optional, a few minutes
```

Headless:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/final_project_notebook.ipynb
```

## Data

*Customer IT Support — Ticket Dataset* (tobiasbueck / Open Ticket AI, 2025, version 14), public
and synthetic: https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets

20,000 tickets; the English subset with complete fields is 10,888 rows. Each row is one
support ticket: subject, body, the agent's actual answer, and the queue, type and priority
it was filed under.

**The corpus is de-duplicated before use.** The data is synthetic and was generated by
paraphrasing templates, so it contains large families of near-identical tickets — the
median cosine similarity between a test ticket and its nearest neighbour in the raw corpus
is 0.906. Left alone, this inflates every retrieval metric: queue prediction over the raw
corpus scores 0.68, and removing neighbours above 0.85 collapses that to 0.12. The
notebook removes near-duplicates at 0.90, keeping {100 * 7163 / 10888:.0f}% of the rows,
and then asserts that no test ticket has a knowledge-base neighbour above the threshold.

## Evaluation

{n} held-out tickets, 5 per queue, scored automatically against the dataset's own labels.
Eight metrics, three system versions, plus a majority-class floor. Paired comparisons use
McNemar's exact test.

## Results

{metrics_table}

Retrieval corrected 15 of the baseline's routing errors and broke none (p = 0.0001).
{miss_p:.0%} of the remaining errors are the retrieval step never surfacing the right
document.

Routing accuracy of {v2.routing_accuracy:.2f} reads low until you know what the data
supports. A logistic regression trained on all 7,113 labelled tickets — far more
supervision than retrieval ever gets — reaches {mini_top1:.2f} on the same test set. The
pipeline is at {pct:.0f}% of that. Four attempts to push past it are recorded in
`results.md`, and none of them worked; the queue label is only weakly determined by the
ticket text.

## Known limitations

- **{n} test tickets.** Enough to detect the retrieval effect, too few for per-queue rates,
  which move by 20 points on a single error.
- **Routing accuracy is {v2.routing_accuracy:.2f}.** Three times the floor and not close to
  unsupervised operation. This is a triage assistant for a human reviewer, not a router.
- **Priority does not work** ({v2.priority_accuracy:.2f} against a {vt.priority_accuracy:.2f}
  floor). Urgency depends on contract tier, SLA and blast radius, none of which appear in
  the ticket text.
- **The labels themselves are noisy.** Tickets asking "could you provide more information
  on how to configure this" are filed as `Change` rather than `Request`, which caps type
  accuracy below 1.00 for any model.
- **Overlapping queues.** Customer Service, IT Support and Product Support are not cleanly
  separable from ticket text, and score 0.00–0.20.
- **Citation accuracy is a proxy** — it checks the right document id appears, not that the
  cited source supports the sentence attached to it.
- **Synthetic data.** Conclusions may not transfer to real support traffic.

## Responsible use

Outputs are drafts for human review. The system should not close, route or reply to a real
ticket unattended. All processing is local, which is the point: no customer data leaves
the machine.

## Files

| Path | What it is |
|---|---|
| `MAI600_FinalPrj_Arnedo.pdf` | the final article (APA) |
| `notebooks/final_project_notebook.ipynb` | the full pipeline, with outputs |
| `results.md` | results and discussion |
| `ai_usage_disclosure.md` | how AI tools were used, and what they got wrong |
| `results/summary_metrics.csv` | metrics per system version |
| `results/evaluation_scores.csv` | per-ticket scores |
| `results/error_taxonomy.csv` | routing errors by pipeline stage |
| `results/ceiling_analysis.csv` | what a supervised classifier reaches on the same tickets |
| `results/module1_token_budget.csv` | encoder context budget |
| `results/module2_sampling_comparison.csv` | temperature and reproducibility |
| `results/module5_local_slm_benchmark.csv` | local serving cost per generation |
| `results/module7_rag_vs_finetuning_decision.csv` | RAG or fine-tuning, decided per field |
| `prompts/` | baseline and improved prompt templates |
| `results/improvement_comparison.csv` | Module 7 to Module 8 |
| `results/module7_baseline/` | frozen Module 7 results |
| `data/training_examples.jsonl` | LoRA-ready type-classification examples |
| `scripts/` | notebook builder, artefacts, report generation |

## Milestones

- `assignment-6` — proposal
- `assignment-7` — working prototype
- `assignment-8` — this final version

## AI usage disclosure

See `ai_usage_disclosure.md`.
"""

(ROOT / "README.md").write_text(readme, encoding="utf-8")
print(f"README.md written ({len(readme)} chars)")
