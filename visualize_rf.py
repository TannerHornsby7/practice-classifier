"""
Random Forest visual explainer — 4 panels:
  1. Bootstrap sampling  (how each tree gets its training data)
  2. Single tree         (high variance, jagged boundaries)
  3. Individual trees    (each sees something different)
  4. The ensemble        (smooth, stable, averaged)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

rng = np.random.RandomState(42)

# --- synthetic 2-feature dataset (think: Age vs Days_Until_Appointment) ---
X, y = make_classification(
    n_samples=300, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, flip_y=0.1, random_state=42
)
feature_names = ["Age", "Days Until Appt"]

# color maps
COLORS = ["#4C9BE8", "#E8754C"]  # blue=show, orange=no-show
cmap_pts  = ListedColormap(COLORS)
cmap_bg   = ListedColormap(["#d0e8ff", "#ffddd0"])

def plot_decision_boundary(ax, model, X, y, title, alpha=0.4, show_points=True):
    h = 0.05
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=alpha, cmap=cmap_bg)
    if show_points:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pts,
                   edgecolors='white', linewidths=0.5, s=30, zorder=3)
    ax.set_xlabel(feature_names[0], fontsize=9)
    ax.set_ylabel(feature_names[1], fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

fig = plt.figure(figsize=(14, 10))
fig.suptitle("How Random Forest Works", fontsize=15, fontweight='bold', y=0.98)

# ── Panel 1: Bootstrap Sampling ──────────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
n = len(X)
for i, (color, label) in enumerate(zip(["#4C9BE8","#E8754C","#6DBF67"],
                                        ["Tree 1 sample","Tree 2 sample","Tree 3 sample"])):
    idx = rng.choice(n, size=n, replace=True)   # bootstrap: sample WITH replacement
    jitter = rng.normal(0, 0.03, (len(idx), 2))
    ax1.scatter(X[idx, 0] + jitter[:,0],
                X[idx, 1] + jitter[:,1],
                alpha=0.25, s=18, color=color, label=label)

ax1.set_xlabel(feature_names[0], fontsize=9)
ax1.set_ylabel(feature_names[1], fontsize=9)
ax1.set_title("① Bootstrap Sampling\nEach tree trains on a random\nresample (with replacement)", fontsize=10, fontweight='bold')
ax1.legend(fontsize=7, loc='upper right')
ax1.set_xticks([]); ax1.set_yticks([])

# ── Panel 2: Single deep tree (overfit) ──────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
single_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
single_tree.fit(X, y)
plot_decision_boundary(ax2, single_tree, X, y,
    "② Single Deep Tree\nOverfits — memorises noise,\njagged unstable boundaries")

# add annotation arrows showing jagged region
ax2.annotate("jagged\nboundary", xy=(0.8, 0.3), xycoords='axes fraction',
             xytext=(0.55, 0.15), textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=8, color='red')

# ── Panel 3: 4 individual trees from RF (each different) ─────────────────────
axes_trees = [fig.add_subplot(2, 3, i) for i in [4, 5]]  # bottom-left two
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X, y)

for i, ax in enumerate(axes_trees):
    tree = rf_full.estimators_[i]
    plot_decision_boundary(ax, tree, X, y,
        f"③ Individual Tree #{i+1}\n(random subset of data+features\n→ each learns something different)",
        alpha=0.5)

# ── Panel 4: Full Random Forest ensemble ─────────────────────────────────────
ax4 = fig.add_subplot(2, 3, 3)
plot_decision_boundary(ax4, rf_full, X, y,
    "④ Random Forest (100 trees)\nMajority vote averages out\nindividual errors → smooth, stable",
    alpha=0.5)

score = rf_full.score(X, y)
ax4.text(0.03, 0.97, f"Accuracy: {score:.1%}", transform=ax4.transAxes,
         fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ── Panel 5: Feature importance bar chart ────────────────────────────────────
ax5 = fig.add_subplot(2, 3, 6)
importances = rf_full.feature_importances_
bars = ax5.barh(feature_names, importances, color=["#4C9BE8", "#E8754C"])
ax5.set_xlabel("Importance", fontsize=9)
ax5.set_title("⑤ Feature Importance\n(which features drove\nthe splits most?)", fontsize=10, fontweight='bold')
for bar, val in zip(bars, importances):
    ax5.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', fontsize=9)
ax5.set_xlim(0, max(importances) * 1.3)

# ── Legend ───────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=COLORS[0], label='Show (No-show=No)'),
    mpatches.Patch(color=COLORS[1], label='No-show (No-show=Yes)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig("rf_explainer.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved to rf_explainer.png")
