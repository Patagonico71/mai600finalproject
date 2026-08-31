"""Two presentation charts. Not the evaluation figures -- those carry six metrics and
technical labels, which is right for the article and unreadable on a projector.

Each of these makes one point, states it in the title, and is legible from the back of a
room. Reads its numbers from the same CSVs as everything else.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
S = pd.read_csv(ROOT / "results/summary_metrics.csv").set_index("version")
C = pd.read_csv(ROOT / "results/ceiling_analysis.csv")

plt.rcParams.update({"font.size": 15, "axes.titlesize": 19, "axes.labelsize": 15,
                     "axes.spines.top": False, "axes.spines.right": False})

GREY, BLUE, GREEN, RED = "#9aa0a6", "#3b7dd8", "#1e8e5a", "#c0392b"

# ---------------------------------------------------------------- slide 1
floor = float(S.loc["V_majority_class"].routing_accuracy)
base = float(S.loc["V0_baseline"].routing_accuracy)
rag = float(S.loc["V2_rag_constrained"].routing_accuracy)
ceil = float(C[C.reference.str.contains("MiniLM")].top_1.iloc[0])

labels = ["No retrieval", "Answer the most\ncommon queue", "This system\n(RAG)",
          "Supervised model\ntrained on 7,113 labels"]
values = [base, floor, rag, ceil]
colors = [GREY, GREY, GREEN, BLUE]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(labels, values, color=colors, width=0.62)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.014, f"{v:.2f}",
            ha="center", fontsize=21, fontweight="bold")

ax.axhline(ceil, color=BLUE, linestyle=":", linewidth=1.6, zorder=0)
# The gap between the system and the ceiling, drawn in the space between the two bars
# so it does not sit on top of either value label.
ax.annotate("", xy=(2.5, rag), xytext=(2.5, ceil),
            arrowprops=dict(arrowstyle="<->", color=RED, linewidth=2))
ax.text(2.5, (rag + ceil) / 2, "  4 tickets", color=RED, fontsize=14,
        va="center", ha="left")

ax.set_ylim(0, ceil + 0.13)
ax.set_ylabel("Correct queue (50 tickets)")
ax.set_title("Retrieval does the work — and gets 88% of the way to a trained model",
             pad=16, loc="left")
ax.grid(axis="y", alpha=0.18)
plt.tight_layout()
plt.savefig(ROOT / "images/slide1_routing_in_context.png", dpi=170)
print("slide1_routing_in_context.png")

# ---------------------------------------------------------------- slide 2
thresholds = ["Raw corpus\n(no cleaning)", "Remove twins\nabove 0.95",
              "Remove twins\nabove 0.90", "Remove twins\nabove 0.85"]
acc = [0.68, 0.54, 0.28, 0.12]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(thresholds, acc, color=[RED] + [GREY] * 3, width=0.62)
for b, v in zip(bars, acc):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.014, f"{v:.2f}",
            ha="center", fontsize=20, fontweight="bold")

ax.axhline(floor, color="#444", linestyle="--", linewidth=1.5)
ax.text(0.42, floor + 0.018, "chance", fontsize=13, ha="left", color="#444")

ax.set_ylim(0, 0.80)
ax.set_ylabel("Queue prediction accuracy")
ax.set_title("The 0.68 was the system finding a reworded copy of the test ticket",
             pad=16, loc="left")
ax.grid(axis="y", alpha=0.18)
plt.tight_layout()
plt.savefig(ROOT / "images/slide2_duplicate_contamination.png", dpi=170)
print("slide2_duplicate_contamination.png")
