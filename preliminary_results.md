# Preliminary Results Draft

The prototype was tested on 20 held-out English tickets, 2
from each of the 10 support queues. Each ticket was run through both system
versions, for 40 generations in total. Both versions use `llama3.2:3b`, so any
difference between them comes from retrieval alone.

## Metric summary

| Metric | Majority class | V0 baseline | V1 RAG |
|---|---|---|---|
| Retrieval hit rate @5 | n/a | n/a | 0.50 |
| Correct queue wins the vote | n/a | n/a | 0.30 |
| Routing accuracy | 0.10 | 0.00 | 0.25 |
| Queue validity | 1.00 | 0.00 | 1.00 |
| Citation accuracy | n/a | n/a | 0.40 |
| Type accuracy | 0.50 | 0.40 | 0.45 |
| Priority accuracy | 0.35 | 0.35 | 0.40 |
| Format adherence | n/a | 1.00 | 1.00 |
| Mean response time | n/a | 1.8s | 2.6s |

The **majority class** column answers every ticket with the most frequent label and uses
no model at all. It is the floor any real system has to clear, and including it changes
how two of these rows should be read.

Per-test detail is in `results/evaluation_scores.csv`; the full generated text is in
`results/generated_outputs.csv`.

## The clearest effect: the baseline does not know the taxonomy exists

Queue validity separates being wrong from being ungrounded, and it is where the two
versions differ most: 0.00 for the baseline against
1.00 with retrieval.

Without retrieval the model almost never picks a real queue. It does not appear to
understand that "queue" names a fixed routing destination at all, and instead invents a
title that summarises the ticket -- "Excessive Billing Inquiry", "Adobe Premiere Pro Crash
Investigation", "Digital Campaign Enhancement". These read well and a ticketing system
would reject every one of them. Retrieval supplies the ten queues that actually exist, and
the model stays inside them.

Across repeated runs the baseline occasionally lands on a real queue name by coincidence, always Technical Support -- the most generic label in the taxonomy and the most frequent one in the dataset. It did not happen in this run. Nothing about it resembles knowledge of the taxonomy.

This is worth stating carefully in the final article, because it is a better argument for
RAG than any accuracy delta. The knowledge the baseline is missing is not reasoning
ability; it is a fact about this organisation that no amount of model scale would supply.

## Only queue routing clears the floor

Read against the majority-class column, the three predicted fields separate sharply.

**Queue routing works.** 0.25 against a 0.10
floor, with the baseline at 0.00. This is the one field where the
system does something a trivial rule cannot.

**Type classification does not.** Both versions sit at or below the
0.50 floor. Answering "Incident" to every ticket would score higher than either system. Retrieval does not change this:
an earlier version of the prompt withheld the retrieved cases' type labels entirely, and
adding them moved the score by nothing at all. The model is not reading the ticket for
type; it is following the shape of the label distribution, and slightly worse than a
constant would.

**Priority is marginal.** 0.40 against a
0.35 floor is a difference of two tickets out of
20, which at this sample size is not evidence of anything.

Reporting only the first of these three would have been the easy path. The honest summary
is that this prototype is a queue router with two extra fields attached that currently do
no useful work.

## Where the routing ceiling is

The absolute number is modest and the reason is worth being honest about: several of these queues overlap semantically. "Technical Support",
"IT Support" and "Product Support" resolve similar problems in similar language, and the
errors cluster on exactly those boundaries.

Some of the ceiling belongs to the dataset rather than to the system. One test ticket
describes campaign data lost in HubSpot CRM through a SAP ERP integration, and is labelled
"Customer Service"; retrieval returned Product Support, Technical Support and Service
Outages. Reading the ticket, those look like better answers than the label. A human
triage team would not agree unanimously on these either, so exact-match accuracy against a
single gold label understates the system and overstates how well-defined the task is.

## Limitations

The test set is 20 tickets. That satisfies the assignment and shows
direction, but it is far too small for the difference between versions to be
statistically meaningful, and with 2 tickets per queue a single error
moves a per-queue rate by a large margin.

The queue distribution in the source data is heavily skewed -- Technical Support carries
roughly twenty times the volume of General Inquiry -- while the queue documents are built
from equal-sized samples. Rare queues are therefore better represented in the knowledge
base than they are in reality, which flatters them.

Priority is the weakest label throughout, for both versions. Urgency depends on business
context that simply is not present in the ticket text, so neither the model nor a human
reading only the ticket could reliably recover it.

Citation accuracy is measured by a proxy: whether the expected document id appears in the
model's citation field. That checks the model cited the right source, not that the cited
source actually supports the sentence it is attached to. Real groundedness needs manual
review.

## Next steps

Hand-score groundedness on a subset instead of relying only on the citation proxy. Try
reranking the retrieved chunks before prompt assembly. Test whether merging the
overlapping support queues into coarser routing targets raises accuracy enough to justify
the loss of granularity, since that would test whether the errors are a system weakness or
a taxonomy artefact. Expand the test set well beyond 20 tickets.
