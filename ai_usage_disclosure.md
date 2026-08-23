# AI Usage Disclosure

**For:** MAI 600 — Module 7 Final Project Progress Report
**Project:** Local RAG Assistant for Customer Support Ticket Triage

| Item | Response |
|---|---|
| **AI tools used** | Claude (Anthropic) as a coding assistant for the notebook and the report drafts. ChatGPT for grammar and for translating some passages first written in Spanish. Hugging Face for model and embedding research. The prototype itself runs `llama3.2:3b` locally through Ollama — that model is the object of study, not an authoring tool. |
| **How AI was used** | To adapt the instructor's Module 7 notebook to this project's dataset, to write the evaluation and charting code, to diagnose retrieval failures, and to draft the report sections. Every metric in this repository was produced by running the code, not written by hand. |
| **Prompts used (examples)** | "Adapt this notebook to build the knowledge base from real agent answers instead of fictional policy documents." / "Retrieval hit rate is 0.40 — show me the retrieved chunks for the failing cases so I can see why." / "Explain what changed between the baseline and the RAG version for this ticket." |
| **What the AI got wrong** | Three defects reached working code before being caught, all of them producing believable numbers. (1) The knowledge base was first built by embedding agent answers, so retrieval matched on courtesy boilerplate instead of on the problem. (2) The test split sampled 2 tickets per queue and then truncated the list, silently dropping the last two queues alphabetically — including the queue the model predicts most often. (3) The format-adherence metric required a valid queue name, which turned it into a duplicate of the validity metric and scored the baseline 0.00 on a format it had followed correctly. All three were found by inspecting per-case output rather than by reading the summary table, which is the lesson recorded here. |
| **Verification performed** | The notebook runs end to end and every number in `progress_report.md` and `preliminary_results.md` is interpolated from the CSVs it writes, so the report cannot drift from the results. Retrieval was inspected case by case, not only in aggregate. The claim that the baseline invents queue names was checked against the raw generated text in `results/generated_outputs.csv`. |
| **Academic integrity statement** | AI was used as a coding and drafting assistant, not as a substitute for the student's own work. The system design, the decision to drop fine-tuning, the choice of metrics, the interpretation of the results, and the final wording are the student's own and the student's responsibility. |

### Addendum — findings from the evaluation review

Two further problems surfaced after the first full set of results, both of them cases
where the code ran correctly and the numbers were still misleading:

- **The prompt passed the retrieved cases' queue but not their type or priority.** Those
  labels existed in the data and were dropped during a refactor. Retrieval could therefore
  not inform two of the three predicted fields, which is exactly what the metrics showed —
  queue improved, type and priority did not move. Adding the labels back was the obvious
  fix and, tested directly, it changed the scores by nothing. The negative result is
  reported rather than dropped.
- **There was no trivial baseline.** Every metric was being read against the other system
  version instead of against the floor. Once a majority-class column was added — answer
  the most frequent label, no model — it became clear that type accuracy sits *below* that
  floor and priority barely above it. The apparent scores were reflecting the shape of the
  label distribution, not classification ability.

The second one matters most: the evaluation looked complete and was not, and no amount of
re-reading the summary table would have revealed it. The lesson recorded for the final
submission is to define the trivial baseline before running anything else.
