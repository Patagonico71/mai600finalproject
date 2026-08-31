"""How much of this task is actually learnable from the ticket text?

Routing accuracy of 0.30 means nothing on its own. It needs two reference points: the
floor a system has to beat, and the best any method reaches on the same data. This script
measures the second one by training a supervised classifier that gets to see every label
in the corpus -- far more supervision than the retrieval pipeline ever has -- and by
repeating that probe on stronger sentence encoders to check whether the embedding space
is what limits the result.

Writes results/ceiling_analysis.csv. Takes a few minutes; it embeds the corpus once per
encoder.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch and sklearn each load libomp

from pathlib import Path
import numpy as np
import pandas as pd
import faiss

faiss.omp_set_num_threads(1)  # see the OpenMP note in the notebook

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
SEED, DEDUP, PER_QUEUE = 42, 0.90, 5
ENCODERS = ["all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"]

df = pd.read_csv(ROOT / "data/dataset-tickets-multi-lang-4-20k.csv")
df = df[df.language == "en"].dropna(
    subset=["subject", "body", "answer", "queue", "type", "priority"]).reset_index(drop=True)
text = (df.subject.astype(str).str.strip() + ". " + df.body.astype(str).str.strip()).tolist()

# Same de-duplication the pipeline uses, so the probe and the system share a corpus.
base = SentenceTransformer(ENCODERS[0])
E = base.encode(text, convert_to_numpy=True, show_progress_bar=False,
                batch_size=64).astype("float32")
faiss.normalize_L2(E)
ix = faiss.IndexFlatIP(E.shape[1]); ix.add(E)

keep = np.ones(len(df), bool)
for start in range(0, len(df), 512):
    stop = min(start + 512, len(df))
    sim, nbr = ix.search(E[start:stop], 30)
    for r in range(stop - start):
        a = start + r
        if not keep[a]:
            continue
        for c in range(1, 30):
            b = int(nbr[r, c])
            if sim[r, c] >= DEDUP and b != a and keep[b]:
                keep[b] = False

D = df[keep].reset_index(drop=True)
kept_text = [text[i] for i in np.where(keep)[0]]

rng = np.random.RandomState(SEED)
test_i = np.concatenate([rng.choice(np.where(D.queue == q)[0], PER_QUEUE, replace=False)
                         for q in sorted(D.queue.unique())])
train_mask = np.ones(len(D), bool); train_mask[test_i] = False
y_test = D.queue.values[test_i]

print(f"corpus {len(df)} -> de-duplicated {len(D)} | train {train_mask.sum()} | test {len(test_i)}\n")

rows = []
for name in ENCODERS:
    model = SentenceTransformer(name)
    prefix = "query: " if "e5" in name else ""     # e5 expects a task prefix
    Xe = model.encode([prefix + t for t in kept_text], convert_to_numpy=True,
                      show_progress_bar=False, batch_size=32).astype("float32")
    faiss.normalize_L2(Xe)

    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    clf.fit(Xe[train_mask], D.queue.values[train_mask])
    proba, classes = clf.predict_proba(Xe[test_i]), clf.classes_

    top1 = float((clf.predict(Xe[test_i]) == y_test).mean())
    top3 = float(np.mean([y_test[i] in classes[np.argsort(-proba[i])[:3]]
                          for i in range(len(y_test))]))
    rows.append({"reference": f"Supervised classifier on {name}",
                 "dimensions": Xe.shape[1], "top_1": round(top1, 3), "top_3": round(top3, 3)})
    print(f"{name:<30} dim {Xe.shape[1]:<5} top-1 {top1:.2f}  top-3 {top3:.2f}")

S = pd.read_csv(ROOT / "results/summary_metrics.csv").set_index("version")
out = pd.DataFrame(
    [{"reference": "Majority-class floor", "dimensions": None,
      "top_1": float(S.loc["V_majority_class"].routing_accuracy), "top_3": None},
     {"reference": "V0 - no retrieval", "dimensions": None,
      "top_1": float(S.loc["V0_baseline"].routing_accuracy), "top_3": None},
     {"reference": "V2 - RAG pipeline (this system)", "dimensions": 384,
      "top_1": float(S.loc["V2_rag_constrained"].routing_accuracy), "top_3": None}] + rows)

best = max(r["top_1"] for r in rows)
system = float(S.loc["V2_rag_constrained"].routing_accuracy)
same_encoder = rows[0]["top_1"]
out.to_csv(ROOT / "results/ceiling_analysis.csv", index=False)

print(f"\nsystem {system:.2f} against {same_encoder:.2f} on the same encoder "
      f"= {system / same_encoder:.0%} of it")
print(f"best supervised result across encoders: {best:.2f}")
print("written: results/ceiling_analysis.csv")
