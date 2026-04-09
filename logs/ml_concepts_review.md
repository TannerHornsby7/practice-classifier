# ML Concepts Review
*Generated during interview prep session — April 2026*

Visualizations in this folder:
- `rf_explainer.png` — bootstrap sampling, single tree vs ensemble, feature importance
- `tree_internals.png` — the binary tree data structure, how RF averages probability vectors
- `validation_curve.png` — train vs val AUC across max_depth values
- `roc_curve.png` — final model ROC curve on test set
- `feature_importance.png` — feature importance from final model

---

## Train / Validation / Test / Streaming Split

**Why three splits, not two?**
Every decision you make during development — which features, which model, which hyperparameters — is implicitly guided by whichever data you evaluate on. If you use the test set for those decisions you've leaked signal from it and your final accuracy is optimistic. The test set must be touched exactly once, at the very end.

```
Train      → fit model weights
Validation → tune hyperparameters, compare models (gets "used up" by repeated evaluation)
Test       → final unbiased accuracy number, touched once
Streaming  → carved off before everything else, replayed as live inference
```

**Why does the test percentage shrink with large datasets?**
You need the test set large enough to give a *stable estimate*, not large enough as a percentage. Stability is a function of absolute count, not proportion. ~10K samples gives accuracy estimates stable to within ~1%. Beyond that you're wasting training signal.

**Train / Val / Test split for 110K rows:**
- Streaming: 10% (carved first)
- Train: ~72% of total
- Val: ~9% of total
- Test: ~9% of total

**Retrain on Train + Val before final evaluation:**
After hyperparameter tuning on val, retrain on train + val combined. You've already used val to make decisions — no reason to leave that data out of the final model.

---

## Random Forest

**Decision tree data structure:**
A binary tree where each internal node stores:
- Which feature to split on
- The threshold value (`feature ≤ threshold`)
- Left and right child pointers

Each leaf stores class counts from training samples that landed there → normalized to probabilities at inference.

sklearn stores this in flat arrays for cache efficiency: `tree_.feature`, `tree_.threshold`, `tree_.children_left`, `tree_.children_right`, `tree_.value`.

**Splits are always axis-aligned** — every cut is a horizontal or vertical line in feature space. A tree can split on the same feature multiple times at different thresholds to carve out a range (e.g., `Age > 20 AND Age ≤ 50`).

**How the tree picks where to split:**
Tries every feature and every possible threshold. Picks the one that maximally reduces Gini impurity.

**Gini impurity:**
```
Gini = 1 - (p_show² + p_noshow²)

50/50 split → 1 - (0.5² + 0.5²) = 0.5   ← maximum chaos
90/10 split → 1 - (0.9² + 0.1²) = 0.18  ← one class dominates
100/0 split → 1 - (1.0² + 0.0²) = 0.0   ← perfect purity
```
Intuition: probability that two randomly picked samples from this node have different labels.
Max Gini for binary classification is 0.5. Values above 0.5 only occur in multi-class problems.

**Why a single deep tree fails:**
High variance — small changes in training data produce a completely different tree. Overfits by memorizing noise, carves jagged arbitrary boundaries.

**How Random Forest fixes this:**
- Train N trees, each on a random bootstrap sample (sampling with replacement)
- At each node, each tree only considers a random subset of features (`max_features`)
- Prediction = average probability vectors across all trees

Key insight: if each tree's errors are uncorrelated, they cancel when averaged. Forcing random feature subsets ensures trees are diverse — if one strong feature (Age) always dominated, all trees would be nearly identical and averaging would do nothing.

**How averaging works (soft voting, not hard voting):**
Each tree outputs a probability vector `[P(show), P(no-show)]`. RF averages the vectors. Final class = argmax of averaged probabilities. This preserves confidence — a tree that's 99% sure pulls the average harder than one that's 51% sure.

**Leaf node probabilities:**
Each leaf stores class counts of training samples that reached it. Shallow leaves = many samples = probabilities near global base rate. Deep leaves = few samples = extreme probabilities, less trustworthy. `min_samples_leaf` enforces a minimum so no leaf makes estimates from 2-3 data points.

**Output dimensionality:**
- Input: N features (91 in our case)
- Output: always length = number of classes (2 for show/no-show: `[P(show), P(no-show)]`)
- These are completely unrelated. 91 features compress to 2 output probabilities.

---

## Hyperparameters

| Hyperparameter | What it controls |
|---|---|
| `max_depth` | How deep each tree grows. Deeper = more complex rules = more overfitting risk |
| `n_estimators` | Number of trees. More is generally better until diminishing returns |
| `max_features` | How many features each node considers. Forces tree diversity |
| `min_samples_leaf` | Min samples at a leaf. Prevents memorizing individual outliers |
| `class_weight` | Upweights minority class. Use `'balanced'` when classes are imbalanced |

**Is feature order a hyperparameter?** No. At every node the tree tries ALL features and picks the best by Gini reduction. The root feature appears there because it gave the biggest global impurity reduction, not because of its position in the dataset.

**Finding the max_depth range:**
- Hard upper bound: `log₂(n_samples)` — for 110K rows that's ~17
- Practical search range: `[3, 5, 7, 10, 15, 20]`
- Use a validation curve: plot train AUC vs val AUC across depths, pick where val peaks before dropping
- In Random Forest: slightly shallow trees are often better because the ensemble handles variance

**Hyperparameter search — RandomizedSearchCV:**
Tries N random combinations from the search space. For each combination runs k-fold cross-validation on the training set (validation set not touched). `scoring='roc_auc'` optimizes for AUC.

Random search beats grid search because most hyperparameter spaces have some dimensions that barely matter — random search wastes less time exhaustively covering them.

---

## Why strings can't go into ML models

All ML algorithms are linear algebra under the hood — matrix multiplications, dot products, Gini calculations. These require numbers. "M" and "F" have no natural ordering or distance a mathematical operation can exploit.

**Binary/ordinal categories** (Gender M/F) → label encode to 0/1.

**Multi-value unordered categories** (Neighbourhood, 80 values) → one-hot encode. Create one binary column per value. `drop_first=True` avoids multicollinearity (one column is redundant since the others determine it).

**Dates** → engineer a numeric feature. `AppointmentDay - ScheduledDay` = `days_until_appt` (integer). The model can now ask "is waiting > 30 days associated with higher no-show rates?" and measure the Gini reduction of that split.

---

## AUC

**What AUC measures:**
Randomly pick one no-show patient and one show patient. AUC = probability the model assigns a higher risk score to the no-show patient.

```
AUC 0.5 → coin flip, model has no signal
AUC 0.7 → model ranks the no-show higher 70% of the time
AUC 1.0 → perfect separation
```

**Why better than accuracy for imbalanced data:**
A model that predicts "show" for everyone gets 80% accuracy on this dataset. AUC = 0.5 — immediately exposed as a coin flip because it assigns equal scores to everyone.

**The ROC curve:**
Generated by sweeping a classification threshold from 0 to 1 and plotting at each point:
- X axis: False Positive Rate (of actual shows, fraction wrongly flagged as no-show)
- Y axis: True Positive Rate / Recall (of actual no-shows, fraction correctly caught)

AUC = area under that curve = threshold-independent summary of model quality.

**AUC is not the whole story — calibration matters too:**
AUC measures ranking quality but not whether probabilities are accurate. Two models can have identical AUC but one says "80% chance of no-show" when the real rate is 40%. If downstream decisions depend on the probability value (e.g., "send a reminder to anyone above 60% risk"), miscalibrated probabilities make your threshold meaningless. Check calibration with a reliability diagram.

**Benchmark for this dataset:**
Published papers land at 0.74-0.76. Above 0.80 = check for data leakage. The ceiling is low because the dataset lacks the strongest predictor: patient-level historical no-show behavior (in our version we engineered this and jumped from 0.71 → 0.74).

---

## Random Forest vs other models

**Why not logistic regression?**
Assumes a linear decision boundary. Misses interaction effects — "elderly + diabetic + appointment 60 days away" is a specific risk profile more than the sum of its parts. Trees capture this automatically.

**Why not a neural network?**
Tree ensembles almost always outperform neural networks on tabular data. NNs need feature scaling, more tuning, more data. Overkill here.

**Why LightGBM/XGBoost often beats Random Forest:**
Random Forest = parallel trees (bagging). Gradient boosting = sequential trees where each new tree corrects the specific mistakes of the previous ones. Sequential error correction is more efficient. Typically 1-3% better AUC on tabular benchmarks.

**Senior engineer answer:**
Train both RF and LightGBM. Report both. "RF gives X, LightGBM gives Y. I'd use LightGBM in production but RF is more interpretable for stakeholders."

---

## Things to extend understanding

- **Calibration**: plot a reliability diagram (predicted probability bins vs actual frequency). Use `CalibratedClassifierCV` to fix miscalibration.
- **SHAP values**: per-prediction feature importance. Explains *why* the model made a specific prediction (useful for EliseAI's healthcare use case where decisions need to be explainable).
- **Learning curves**: plot AUC vs training set size. Tells you whether you're data-limited or model-limited.
- **LightGBM**: drop-in replacement for RF with usually better performance. `pip install lightgbm`, `from lightgbm import LGBMClassifier`.
- **Stratified k-fold**: ensure class balance is preserved across folds during cross-validation.
- **Partial dependence plots**: show the marginal effect of one feature on the predicted outcome, holding others constant.
