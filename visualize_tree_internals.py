"""
Two panels:
  1. The actual tree data structure — nodes, thresholds, Gini impurity
  2. How Random Forest averages probability vectors, not just labels
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

rng = np.random.RandomState(42)

X, y = make_classification(
    n_samples=300, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, flip_y=0.1, random_state=42
)
feature_names = ["Age", "Days Until Appt"]
class_names   = ["Show", "No-show"]

# ── shallow tree so we can read the structure ─────────────────────────────────
shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
shallow.fit(X, y)

rf = RandomForestClassifier(n_estimators=5, random_state=42)  # tiny RF for illustration
rf.fit(X, y)

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Decision Tree Internals + RF Averaging", fontsize=14, fontweight='bold')

# ── Panel 1: the actual tree structure ───────────────────────────────────────
ax1 = fig.add_subplot(1, 2, 1)
plot_tree(
    shallow,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,           # color = majority class
    rounded=True,
    impurity=True,         # show Gini at each node
    proportion=False,      # show raw counts not proportions
    ax=ax1,
    fontsize=8,
)
ax1.set_title(
    "① The Tree Data Structure (depth=3)\n"
    "Each node: feature | threshold | gini | samples | class counts\n"
    "Leaf nodes store class probabilities → prediction",
    fontsize=10, fontweight='bold'
)

# annotation explaining a node
ax1.annotate(
    "Each split is:\nfeature ≤ threshold\n(axis-aligned cut)",
    xy=(0.5, 0.92), xycoords='axes fraction',
    fontsize=8, color='#333',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffde7', alpha=0.9),
    ha='center'
)

# ── Panel 2: how averaging works ─────────────────────────────────────────────
ax2 = fig.add_subplot(1, 2, 2)
ax2.axis('off')

# pick one test point and show how each tree votes
test_point = X[42:43]  # one patient
true_label = y[42]

tree_probs = np.array([t.predict_proba(test_point)[0] for t in rf.estimators_])
avg_probs  = tree_probs.mean(axis=0)
final_pred = class_names[np.argmax(avg_probs)]

title_text = (
    f"② How RF Averages (5 trees, 1 test patient)\n"
    f"True label: {class_names[true_label]}    Final prediction: {final_pred}"
)
ax2.set_title(title_text, fontsize=10, fontweight='bold', pad=12)

# table header
cols = ["Tree", "P(Show)", "P(No-show)", "Predicted"]
col_x = [0.08, 0.32, 0.58, 0.82]
row_y_start = 0.88
row_height  = 0.09

# header
for cx, col in zip(col_x, cols):
    ax2.text(cx, row_y_start, col, fontsize=10, fontweight='bold',
             ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#dddddd', alpha=0.9))

# rows — one per tree
colors_pred = {"Show": "#d0e8ff", "No-show": "#ffddd0"}
for i, probs in enumerate(tree_probs):
    row_y = row_y_start - (i + 1) * row_height
    pred  = class_names[np.argmax(probs)]
    row_data = [f"Tree {i+1}", f"{probs[0]:.2f}", f"{probs[1]:.2f}", pred]
    bg = colors_pred[pred]
    for cx, val in zip(col_x, row_data):
        ax2.text(cx, row_y, val, fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor=bg, alpha=0.7))

# separator line
sep_y = row_y_start - (len(tree_probs) + 1) * row_height + 0.025
ax2.axhline(sep_y + 0.06, xmin=0.02, xmax=0.98, color='#888', linewidth=1.2, linestyle='--')

# average row
avg_y = row_y_start - (len(tree_probs) + 1.2) * row_height
pred_label = class_names[np.argmax(avg_probs)]
avg_data = ["AVERAGE", f"{avg_probs[0]:.2f}", f"{avg_probs[1]:.2f}", f"→ {pred_label}"]
avg_bg = colors_pred[pred_label]
for cx, val in zip(col_x, avg_data):
    ax2.text(cx, avg_y, val, fontsize=10, fontweight='bold', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor=avg_bg, alpha=0.95))

# explanation box
explain_y = avg_y - 0.22
ax2.text(0.5, explain_y,
    "Each tree outputs a probability vector, not just a label.\n"
    "RF averages the vectors across all trees (soft voting).\n"
    "Final class = argmax of the averaged probabilities.\n\n"
    "Why probabilities, not labels?\n"
    "A tree that's 51% confident counts the same as one\n"
    "that's 99% confident if you only take the label.\n"
    "Averaging probabilities preserves that confidence signal.",
    fontsize=9.5, ha='center', va='top',
    bbox=dict(boxstyle='round,pad=0.6', facecolor='#f5f5f5', alpha=0.95),
    linespacing=1.6
)

plt.tight_layout()
plt.savefig("tree_internals.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved to tree_internals.png")
