"""Post-run artefacts: error taxonomy, improvement comparison, LoRA training examples.

Reads the CSVs the notebook wrote. Run after notebooks/final_project_notebook.ipynb.
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
E = pd.read_csv(RES / "evaluation_scores.csv")
S = pd.read_csv(RES / "summary_metrics.csv").set_index("version")
FINAL = "V2_rag_constrained"

# ---------------------------------------------------------------- error taxonomy
# Every routing error is attributed to the stage that caused it. Counting errors says
# how many; this says which part of the pipeline to fix, which is the useful question.
v = E[E.version == FINAL].copy()


def stage(r):
    if r.routing_correct:
        return "correct"
    if not r.retrieval_hit_topk:
        return "A_retrieval_miss"      # right queue document never retrieved
    if not r.vote_source_match:
        return "B_ranking_miss"        # retrieved, but the vote ranked another queue first
    return "C_model_override"          # ranked first and the model chose something else


v["failure_stage"] = v.apply(stage, axis=1)
tax = (v.failure_stage.value_counts().rename_axis("stage").reset_index(name="count"))
tax["share_of_all"] = (tax["count"] / len(v)).round(3)
errs = tax[tax.stage != "correct"]["count"].sum()
tax["share_of_errors"] = tax.apply(
    lambda r: round(r["count"] / errs, 3) if r.stage != "correct" else None, axis=1)
tax.to_csv(RES / "error_taxonomy.csv", index=False)

detail = v[v.failure_stage != "correct"][
    ["test_id", "failure_stage", "true_queue", "pred_queue",
     "retrieval_hit_topk", "vote_source_match"]]
detail.to_csv(RES / "error_detail.csv", index=False)

print("Error taxonomy")
print(tax.to_string(index=False))

# ---------------------------------------------------------------- improvement table
m7 = pd.read_csv(RES / "module7_baseline/summary_metrics.csv").set_index("version")


def row(area, m7v, m8v, evidence):
    return {"Area": area, "Module 7 prototype": m7v,
            "Module 8 final": m8v, "Improvement evidence": evidence}


imp = pd.DataFrame([
    row("Data integrity",
        "no duplicate check",
        f"de-duplicated at 0.90, asserted in the notebook",
        "highest test-to-knowledge-base similarity 0.899, below threshold; raw corpus "
        "scored 0.68 on queue prediction purely from near-duplicate leakage"),
    row("Knowledge base",
        "140 cases, sampled at random",
        "1,782 cases, balanced across queues",
        "generic-queue fallback fell from 8/50 to 2/50 predictions"),
    row("Routing accuracy",
        f"{m7.loc['V1_rag'].routing_accuracy:.2f}",
        f"{S.loc[FINAL].routing_accuracy:.2f}",
        "McNemar against the no-retrieval baseline: p = 0.0001, 15 tickets corrected, "
        "0 broken"),
    row("Type accuracy",
        f"{m7.loc['V1_rag'].type_accuracy:.2f} (below the 0.50 floor)",
        f"{S.loc[FINAL].type_accuracy:.2f} (floor {S.loc['V_majority_class'].type_accuracy:.2f})",
        "the Module 7 finding was a small-sample artefact and is corrected here"),
    row("Queue vote",
        "sum of similarities",
        "mean of similarities",
        "offline ablation over logged retrievals: mean ranks the correct queue first "
        "0.24 of the time against 0.18 for sum"),
    row("Test set", "20 tickets", "50 tickets", "5 per queue instead of 2"),
    row("Statistics", "none", "McNemar on every paired comparison",
        "differences are now reported with a p-value instead of asserted from a delta"),
    row("Error analysis", "counts only", "failure attributed to pipeline stage",
        "results/error_taxonomy.csv"),
])
imp.to_csv(RES / "improvement_comparison.csv", index=False)
print(f"\nimprovement_comparison.csv: {len(imp)} rows")

# ---------------------------------------------------------------- LoRA-ready examples
# Not trained. The evidence says retrieval carries the queue field and does little for
# type, so type is where adaptation belongs -- a fixed four-way taxonomy that needs
# consistent judgement rather than fresh facts. These examples make that testable later.
df = pd.read_csv(ROOT / "data/dataset-tickets-multi-lang-4-20k.csv")
df = df[df.language == "en"].dropna(
    subset=["subject", "body", "answer", "queue", "type", "priority"])
test_bodies = set(pd.read_csv(ROOT / "data/test_cases.csv").body)
df = df[~df.body.isin(test_bodies)]

# Balanced across the four types on purpose: the raw distribution is skewed toward
# Incident, and an adapter trained on that skew would learn to answer Incident.
per_type = 75
picked = pd.concat([g.sample(min(per_type, len(g)), random_state=42)
                    for _, g in df.groupby("type")]).reset_index(drop=True)

out = ROOT / "data/training_examples.jsonl"
with open(out, "w", encoding="utf-8") as fh:
    for _, r in picked.iterrows():
        fh.write(json.dumps({
            "instruction": ("Classify the support ticket into exactly one type: "
                            "Incident, Request, Problem, or Change."),
            "input": f"{str(r['subject']).strip()}. {str(r['body']).strip()}",
            "output": r["type"],
        }, ensure_ascii=False) + "\n")

print(f"training_examples.jsonl: {len(picked)} examples "
      f"({picked.type.value_counts().to_dict()})")
