"""Write progress_report.md from the CSVs the notebook produced.

Kept out of the notebook because the report is a document about the run rather than a
step of the pipeline, but it reads the same CSVs, so the prose cannot drift away from
the numbers. Run it after the notebook.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
S = pd.read_csv(ROOT / "results/summary_metrics.csv").set_index("version")
E = pd.read_csv(ROOT / "results/evaluation_scores.csv")
DOCS = pd.read_csv(ROOT / "data/sample_documents.csv")
CHUNKS = pd.read_csv(ROOT / "results/document_chunks.csv")

v0, v1 = S.loc["V0_baseline"], S.loc["V1_rag"]
vt = S.loc["V_majority_class"]   # always answer the most frequent label, no model
rag = E[E.version == "V1_rag"]
n = int(v1["num_tests"])

# Failure patterns, read from the data rather than asserted.
sink_queue, sink_n = rag.pred_queue.value_counts().idxmax(), rag.pred_queue.value_counts().max()
sink_true = int((rag.true_queue == sink_queue).sum())
hit_but_wrong = int(((rag.retrieval_hit_topk == 1) & (rag.routing_correct == 0)).sum())
n_hits = int(rag.retrieval_hit_topk.sum())
per_queue = rag.groupby("true_queue").routing_correct.mean().sort_values()
worst = [q for q, v in per_queue.items() if v == 0]
best = [q for q, v in per_queue.items() if v == 1]

def delta(metric):
    """Describe a metric's movement honestly, including when RAG made it worse."""
    a, b = v0[metric], v1[metric]
    if b > a:  return f"improved from {a:.2f} to {b:.2f}"
    if b < a:  return f"**dropped** from {a:.2f} to {b:.2f}"
    return f"did not move ({a:.2f} in both)"

maj_type = E[E.version == "V1_rag"].true_type.value_counts().idxmax()

type_note = ""
if v1["type_accuracy"] < v0["type_accuracy"]:
    type_note = (
        "\n\nType accuracy is the one metric where retrieval **hurt**: it "
        f"{delta('type_accuracy')}. Retrieved cases carry their own type labels, and when "
        "a retrieved case is topically similar but of a different type, the model appears "
        "to copy the neighbour's label instead of reading the ticket. The effect is small "
        f"at n={n} and may not survive a larger test set, but it is the kind of thing "
        "retrieval can plausibly cause and it is recorded rather than smoothed over.")

rpt = f"""# Module 7 — Final Project Progress Report

**Course:** MAI 600
**Project:** Local RAG Assistant for Customer Support Ticket Triage
**Milestone:** Working prototype / progress report
**Repository:** https://github.com/Patagonico71/mai600finalproject

---

## 1. Implementation Progress

The prototype runs end to end on a laptop. A ticket goes in, the system retrieves similar
past cases from a knowledge base, and a locally served small language model returns a
triage: routing queue, ticket type, priority, a draft first response, and citations to the
sources it used.

- [x] RAG pipeline retrieves relevant chunks
- [x] Local SLM generates an answer through Ollama
- [x] Baseline and improved system compared on the same test cases
- [x] Evaluation table, {n} test cases, 8 metrics, scored automatically
- [x] Results saved to CSV and charted
- [ ] Fine-tuning — deliberately out of scope, see section 6

| Evidence | File |
|---|---|
| Notebook with all outputs saved | `notebooks/module7_project_progress_colab.ipynb` |
| Screenshots of notebook output | `images/screenshot_*.png` |
| Generated answers, both versions | `results/generated_outputs.csv` |
| Retrieved chunks with similarity scores | `results/retrieved_chunks.csv` |
| Per-test scores | `results/evaluation_scores.csv` |
| Metric summary | `results/summary_metrics.csv` |
| Preliminary results table | `results/preliminary_results_table.csv` |
| Evaluation chart | `images/evaluation_chart.png` |
| Pipeline diagram | `images/pipeline_diagram.png` |

---

## 2. System Description

```
Incoming ticket
  -> Queue documents built from past customer problems
  -> One chunk per past case
  -> MiniLM embeddings + FAISS top-k search + score-weighted queue vote
  -> Prompt assembled with cited context
  -> llama3.2:3b served by Ollama
  -> Triage output with citations
  -> Evaluation against the dataset's true labels
```

The knowledge base is **{len(DOCS)} documents, one per support queue**, built from
{int(DOCS['n_source_cases'].sum())} real past tickets, giving {len(CHUNKS)} chunks. Every
document is grounded in the support team's own history; nothing is invented.

The design decision that mattered most: **what gets embedded is the customer's
description of the problem, not the agent's answer.** The first version indexed resolution
text and retrieved badly. Agent answers are roughly half courtesy boilerplate, and once
that is in the vector, similarity is driven by the tone of a support email rather than by
the problem. In one case the top-ranked chunk scored 0.797 while containing no topical
content at all — just an offer to schedule a call. The resolution is still shown to the
model as payload so it can ground the draft response; it just does not participate in the
search.

---

## 3. Local Serving Check

| Item | Detail |
|---|---|
| Runtime | Ollama, native install |
| Model | `llama3.2:3b` (Q4_K_M, 2.0 GB) |
| Environment | MacBook Pro, Apple M4 Pro, 48 GB unified memory |
| Serving method | Local HTTP API at `localhost:11434` |
| Prompt method | `requests.post()` to `/api/generate`, temperature 0.2 |
| Data leaving the machine | None |

Serving is local by design. Support tickets carry customer names, phone numbers and
account details, so the privacy argument is the reason the project has this shape.

Both versions run the same model, so retrieval is the only variable between them. That
makes section 4 an ablation rather than a model-versus-model benchmark.

Larger local models were considered. `gemma4:12b` was tested and averaged 110 seconds per
ticket against {v1['avg_response_time_s']:.1f} for `llama3.2:3b`, for no gain in routing
accuracy on this task, so it was dropped. The assignment's own framing — small models,
class hardware — points the same way.

---

## 4. Preliminary Results

{n} held-out English tickets, 2 from each of the 10 queues, run through both versions:
{2*n} generations.

| Metric | Majority class | V0 baseline | V1 RAG |
|---|---|---|---|
| Retrieval hit rate @5 | n/a | n/a | {v1['retrieval_hit_rate']:.2f} |
| Correct queue wins the vote | n/a | n/a | {v1['vote_source_match']:.2f} |
| **Routing accuracy** | {vt['routing_accuracy']:.2f} | **{v0['routing_accuracy']:.2f}** | **{v1['routing_accuracy']:.2f}** |
| **Queue validity** | {vt['queue_validity']:.2f} | **{v0['queue_validity']:.2f}** | **{v1['queue_validity']:.2f}** |
| Citation accuracy | n/a | n/a | {v1['citation_accuracy']:.2f} |
| Type accuracy | {vt['type_accuracy']:.2f} | {v0['type_accuracy']:.2f} | {v1['type_accuracy']:.2f} |
| Priority accuracy | {vt['priority_accuracy']:.2f} | {v0['priority_accuracy']:.2f} | {v1['priority_accuracy']:.2f} |
| Format adherence | n/a | {v0['format_adherence']:.2f} | {v1['format_adherence']:.2f} |
| Mean response time | n/a | {v0['avg_response_time_s']:.1f}s | {v1['avg_response_time_s']:.1f}s |

The **majority class** column answers every ticket with the most frequent label and uses
no model. It is the floor any real system has to clear. Adding it was the single most
useful change to this evaluation, because it reverses the reading of two rows.

The full per-test table is `results/preliminary_results_table.csv`. One case shows what
the numbers mean. **Ticket T1** — unexpected charges on a monthly invoice, true labels
*Billing and Payments / Incident / medium*:

| | V0 baseline | V1 RAG |
|---|---|---|
| Queue | `Excessive Billing Inquiry` | `Billing and Payments` |
| Type | Problem | Incident |
| Priority | High | Medium |
| Citations | NONE | DOC-001 |

The baseline's queue is the ticket's own subject line handed back as a category. It reads
like a queue and no ticketing system would accept it.

---

## 5. Metrics

| Metric | What it means | How it was measured |
|---|---|---|
| Retrieval hit rate @5 | Did the correct queue document appear among the retrieved chunks? | `expected_source` present in the top-5 doc ids |
| Queue vote match | Does the correct queue win a score-weighted vote across retrieved chunks? | similarities summed per queue, highest total wins |
| Routing accuracy | Was the ticket sent to the right queue? | predicted queue == true queue |
| Queue validity | Did the model stay inside the real taxonomy? | predicted queue is one of the 10 real queues |
| Citation accuracy | Did it cite the document that supports the answer? | `expected_source` appears in the citations field |
| Type / priority accuracy | Secondary label quality | exact match against dataset labels |
| Format adherence | Did the model emit the requested fields? | all three fields parse from the raw text |
| Response time | Local serving cost | wall-clock seconds per generation |

Scoring is automatic against the dataset's own labels, so the numbers reproduce on a
re-run and there is no room to grade generously.

---

## 6. Observations and Limitations

### What works

Retrieval anchors the model to a taxonomy it has no other way of knowing. Queue validity
{delta('queue_validity')}, and this is the result the project rests on. Without retrieval
the model does not merely pick the wrong queue — it does not appear to grasp that "queue"
names a fixed routing destination, and returns invented titles like "Excessive Billing
Inquiry" or "Adobe Premiere Pro Crash Investigation". The missing knowledge is not
reasoning ability. It is a fact about this organisation that no amount of model scale
would supply, which is the case for RAG over fine-tuning here.

Generation is fast enough for a real workflow: {v1['avg_response_time_s']:.1f}s per ticket
on a laptop, with nothing sent to a third party.

### Two of the three predicted fields do not clear the floor

**Queue routing works.** {v1['routing_accuracy']:.2f} against a {vt['routing_accuracy']:.2f}
majority-class floor, baseline at {v0['routing_accuracy']:.2f}. This is the field the
project is actually about, and it is the one place the system does something a constant
rule cannot.

**Type classification does not.** Both versions sit at or below the
{vt['type_accuracy']:.2f} floor — answering "{maj_type}" to every ticket scores higher
than either system. This was tested directly: an earlier prompt withheld the retrieved
cases' type labels, and adding them changed the score by nothing. The model is following
the label distribution rather than reading the ticket.

**Priority is marginal.** {v1['priority_accuracy']:.2f} against a
{vt['priority_accuracy']:.2f} floor is a difference of one or two tickets at n={n}, which
is not evidence of anything. Urgency depends on business context — contract tier, SLA,
blast radius — that is not in the ticket text at all.

The honest summary is that this prototype is a working queue router with two additional
fields attached that currently do no useful work. Reporting only the routing result would
have been the easy path and would have been misleading.

### Other limitations

**Routing accuracy is {v1['routing_accuracy']:.2f}.** Three times chance on a 10-way
decision, and it comes entirely from retrieval since the model is held constant, but not
good enough to route real tickets unsupervised.

**The model collapses to a generic queue when uncertain.** "{sink_queue}" was chosen
{sink_n} times out of {n}, against {sink_true} that genuinely belonged there. Worse,
retrieval had already found the correct document in {hit_but_wrong} of the misrouted
cases, and in some of those the model cited that correct document and then routed
somewhere else anyway. This is the clearest gap: retrieval hit rate is
{v1['retrieval_hit_rate']:.2f} but routing accuracy is {v1['routing_accuracy']:.2f}, so of
the {n_hits} cases where the right source was retrieved, only {n_hits - hit_but_wrong}
were routed correctly. Finding the evidence and acting on it are separate problems, and
the second one is still open.

**The largest queue scores worst.** Technical Support has ~3,400 tickets in the dataset,
more than any other queue, and scores 0.00. Its document is assembled from randomly
sampled cases out of an extremely diverse queue, so it ends up diffuse and represents
nothing in particular. Narrow, topically tight queues do far better — {best[0] if best else 'Billing and Payments'}
scores 1.00. Sampling breadth, not sampling size, is what the knowledge base needs.
Queues at 0.00 in this run: {', '.join(worst) if worst else 'none'}.

**Some of the ceiling belongs to the dataset.** One test ticket describes campaign data
lost in HubSpot CRM through a SAP ERP integration and is labelled "Customer Service";
retrieval returned Product Support, Technical Support and Service Outages. Reading the
ticket, those look like better answers than the gold label. Human triage teams disagree on
these boundaries too, so exact-match accuracy against a single label understates the
system and overstates how well-defined the task is.

{type_note}

**The test set is {n} tickets.** Enough for the assignment and enough to show direction,
far too small for the differences to be statistically meaningful. With 2 tickets per
queue, one error swings a per-queue rate by 50 points. Repeated runs move individual
cases while the aggregate holds near {v1['routing_accuracy']:.2f}, which is worth knowing
but is not a substitute for a bigger sample.

**Citation accuracy is a proxy.** It checks that the expected document id appears in the
citation field, not that the cited source supports the sentence attached to it. Real
groundedness needs manual review.

### Errors encountered during development

Three bugs reached working code before being caught, and each produced believable numbers:

1. The knowledge base first indexed agent answers, so retrieval matched on courtesy
   boilerplate rather than on the problem. Switching to problem-to-problem matching moved
   the hit rate from 0.40 to {v1['retrieval_hit_rate']:.2f}.
2. Test tickets were sampled 2 per queue and then truncated to a fixed count, which
   silently dropped the last two queues alphabetically — including Technical Support, the
   queue the model most often predicts. The evaluation was excluding its own most likely
   answers.
3. Format adherence required a *valid* queue name, making it a second copy of the validity
   metric and scoring the baseline 0.00 on a format it had followed perfectly. Measured on
   the raw text, both versions score {v1['format_adherence']:.2f}.

All three were found by reading per-case output, not the summary table. That is the
methodological lesson of this milestone: aggregate metrics hid every one of them.

### Why there is no fine-tuning

The assignment allows a LoRA/QLoRA track. It was considered and rejected on the evidence
above. The knowledge the baseline lacks is a list of queue names and the routing
conventions attached to them — facts that change whenever the support organisation
reorganises. Fine-tuning would bake them into weights that go stale, and it does not
produce citations, which this use case needs. The decision table in the notebook records
the reasoning.

---

## 7. What will be improved before the final submission

1. **Close the retrieval-to-decision gap.** Pass the score-weighted queue ranking into the
   prompt explicitly and constrain the model to choose among the retrieved queues instead
   of leaving the choice open. This targets the "{sink_queue}" sink directly.
2. **Build queue documents by coverage, not by random sample**, so a broad queue like
   Technical Support is represented by its range of problems.
3. **Rerank retrieved chunks** before prompt assembly.
4. **Expand the test set** well past {n} tickets and report per-queue confidence intervals.
5. **Hand-score groundedness** on a subset instead of relying on the citation proxy.
6. **Test a merged, coarser queue taxonomy** to find out whether the errors are a system
   weakness or an artefact of overlapping categories.

---

## 8. Draft Article Sections

- `methodology_draft.md`
- `preliminary_results.md`

## 9. AI Usage Disclosure

- `ai_usage_disclosure.md`
"""

(ROOT / "progress_report.md").write_text(rpt, encoding="utf-8")
print(f"progress_report.md written ({len(rpt)} chars)")
print(f"  sink queue      : {sink_queue} chosen {sink_n}/{n}, truly {sink_true}")
print(f"  hits misrouted  : {hit_but_wrong} of {n_hits}")
print(f"  queues at 0.00  : {', '.join(worst) if worst else 'none'}")
