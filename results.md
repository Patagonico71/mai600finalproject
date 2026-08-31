# Results

All figures come from `results/summary_metrics.csv` and `results/evaluation_scores.csv`,
written by `notebooks/final_project_notebook.ipynb`. The test set is 50 held-out tickets,
5 from each of the 10 support queues.

## Metrics

| Metric | Majority class | V0 no retrieval | V1 RAG | V2 RAG constrained |
|---|---|---|---|---|
| Routing accuracy | 0.10 | 0.00 | 0.30 | **0.30** |
| Queue validity | 1.00 | 0.02 | 1.00 | 1.00 |
| Type accuracy | 0.40 | 0.58 | 0.70 | **0.72** |
| Priority accuracy | 0.40 | 0.46 | 0.44 | 0.44 |
| Retrieval hit rate @5 | n/a | n/a | 0.50 | 0.50 |
| Citation accuracy | n/a | n/a | 0.40 | 0.34 |
| Mean response time | n/a | 1.0s | 1.6s | 1.8s |

The **majority class** column answers every ticket with the most frequent label and uses
no model. It is the floor any real system has to clear, and it changes how two of these
rows should be read.

## Retrieval works, and the effect is not marginal

Routing accuracy moves from 0.00 without retrieval to
0.30 with it, against a 0.10 floor. Paired over
the same 50 tickets, retrieval corrected 15 that the baseline got wrong and broke none
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
corpus from 140 randomly sampled cases to 1,782 balanced ones
dropped the generic-queue fallback from 8 predictions in 50 to 2. The constraint was a
crutch for a weak corpus, and with a reasonable corpus it is redundant.

## Where the remaining errors are

Every routing error is attributed to the pipeline stage that caused it
(`results/error_taxonomy.csv`):

| Stage | Errors | Share |
|---|---|---|
| A — the correct queue document was never retrieved | 25 | 71% |
| B — retrieved, but the vote ranked another queue first | 9 | 26% |
| C — ranked first, and the model chose something else | 1 | 3% |

This is the clearest statement of what changed. Module 7's bottleneck was stage C. In this
version stage C is 1 ticket out of 50: the model now follows its evidence almost
without exception. 71% of what remains is retrieval never surfacing the right
document, which is where any further work belongs.

## Per-queue accuracy

- Billing and Payments: 0.80
- Service Outages and Maintenance: 0.80
- Returns and Exchanges: 0.40
- General Inquiry: 0.20
- Human Resources: 0.20
- Product Support: 0.20
- Sales and Pre-Sales: 0.20
- Technical Support: 0.20
- Customer Service: 0.00
- IT Support: 0.00

Narrow, topically distinct queues do well. The generic and overlapping ones do not, and
they fail together: Customer Service, IT Support and Product Support resolve similar
problems in similar language, and a human triage team would not agree unanimously on those
boundaries either.

## How much of this is learnable at all

A routing accuracy of 0.30 says nothing by itself. Against the
majority-class floor of 0.10 it looks like the system does real work.
Against a perfect router it looks like a failure. Neither reading is worth much without
knowing what the best available method reaches on the same tickets.

So I trained one. A logistic regression over sentence embeddings, fitted on all
7,113 de-duplicated training tickets and their queue labels — far more supervision
than the pipeline ever gets, since retrieval never learns from labels at all. On the same
50 held-out tickets it scores 0.34.

| Reference | Top-1 | Top-3 |
|---|---|---|
| Majority-class floor | 0.10 | — |
| V0, no retrieval | 0.00 | — |
| **V2, this system** | **0.30** | — |
| Supervised classifier, all-MiniLM-L6-v2 | 0.34 | 0.54 |
| Supervised classifier, bge-base-en-v1.5 | 0.36 | 0.58 |
| Supervised classifier, e5-base-v2 | 0.40 | 0.56 |

The pipeline reaches 0.30 against that 0.34, which is
88% of what a trained classifier pulls out of the same text with the same encoder.
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
| Retrieve 20 chunks instead of 5 | hit rate 0.50 to 0.88, vote flat at 0.30 |
| Merge the overlapping queues into six groups | accuracy 0.38, floor 0.30 — margin falls from 3.0x to 1.3x |
| Suggest three candidate queues rather than one | covers 0.48 against a 0.30 floor, again a worse ratio |

Growing the knowledge base from 140 cases to 1,782 is the only
change that helped. That is the honest summary: 0.30 sits near what
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
that type classification did not work at all. At 50 tickets it scores
0.72 against a 0.40 floor. The earlier finding was an
artefact of a 20-ticket sample, and it is corrected here rather than quietly dropped.
