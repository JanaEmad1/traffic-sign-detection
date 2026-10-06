"""Draw the results charts from docs/results.json (numbers copied from the notebook's saved outputs).

    python scripts/make_charts.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

DOCS = Path(__file__).resolve().parents[1] / "docs"
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.titleweight": "bold",
    "axes.titlecolor": INK,
})

# GTSRB class names, by ClassId
NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60", "Speed limit 70",
    "Speed limit 80", "End of speed limit 80", "Speed limit 100", "Speed limit 120", "No passing",
    "No passing for trucks", "Right-of-way at next junction", "Priority road", "Yield", "Stop",
    "No vehicles", "Trucks prohibited", "No entry", "General caution", "Dangerous curve left",
    "Dangerous curve right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware of ice/snow", "Wild animals crossing", "End of all limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout", "End of no passing", "End of no passing (trucks)",
]

res = json.loads((DOCS / "results.json").read_text())
rows = sorted(res["per_class"], key=lambda r: r["f1"])

# 1. Per-class F1, all 43 classes, weakest highlighted
fig, ax = plt.subplots(figsize=(8, 10))
labels = [f'{NAMES[r["class_id"]]} ({r["class_id"]})' for r in rows]
f1 = [r["f1"] for r in rows]
colors = [ORANGE if v < 0.6 else (AQUA if v >= 0.9 else BLUE) for v in f1]
ax.barh(labels, f1, color=colors, height=0.7)
ax.axvline(res["test_accuracy"], color=INK, lw=1)
ax.text(res["test_accuracy"] - 0.01, 2,  f'overall accuracy {res["test_accuracy"]:.1%}',
        color=INK, fontsize=9, ha="right", va="center", bbox=dict(fc="white", ec="none", pad=1))
ax.set_xlim(0, 1)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", labelsize=8.5)
ax.set_xlabel("F1 score on the GTSRB test set (12,630 images)")
ax.set_title("HOG + linear SVM: F1 per sign class")
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=AQUA, label="F1 ≥ 0.90"), Patch(color=BLUE, label="0.60–0.89"),
                   Patch(color=ORANGE, label="F1 < 0.60")],
          loc="lower right", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(DOCS / "per_class_f1.png", dpi=150)

# 2. Where the errors are: accuracy by sign family
families = {
    "Speed limits": [0, 1, 2, 3, 4, 5, 7, 8],
    "Prohibitions": [9, 10, 15, 16, 17],
    "Warnings (triangles)": [11, *range(18, 32)],
    "Mandatory (blue)": range(33, 41),
    "Unique shapes": [12, 13, 14],
    "End-of signs": [6, 32, 41, 42],
}
by_id = {r["class_id"]: r for r in res["per_class"]}
fam_acc = {}
for name, ids in families.items():
    correct = sum(by_id[i]["recall"] * by_id[i]["support"] for i in ids)
    total = sum(by_id[i]["support"] for i in ids)
    fam_acc[name] = (correct / total, total)
fam = sorted(fam_acc.items(), key=lambda kv: kv[1][0])
fig, ax = plt.subplots(figsize=(8, 3.4))
bars = ax.barh([k for k, _ in fam], [v[0] for _, v in fam], color=BLUE, height=0.6)
for b, (_, (acc, n)) in zip(bars, fam):
    ax.text(acc + 0.01, b.get_y() + b.get_height() / 2, f"{acc:.0%}  (n={n:,})", va="center", color=INK, fontsize=9)
ax.set_xlim(0, 1.15)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Share of test images classified correctly (support-weighted recall)")
ax.set_title("Signs that differ only in an inner symbol are the hard ones")
fig.tight_layout()
fig.savefig(DOCS / "accuracy_by_family.png", dpi=150)

assert sum(n for _, n in fam_acc.values()) == res["n_test"]
for name, (acc, n) in fam_acc.items():
    print(f"{name:22s} {acc:.3f}  n={n}")
