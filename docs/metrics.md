# Reading the evaluation results

`evaluate(y, folds, files, X, names, out, seed=42, epochs=12, batch_size=32)` expects one aligned row per session: a binary label, an integer fold ID, a file identifier and one row in each feature array in the dictionary `X`. Class 1 is positive. File identifiers appear in the output but are not passed to models.

## What is held out?

The evaluator uses the five supplied folds without reshuffling. In sorted fold order, test fold `i` uses the next fold (wrapping at the end) for threshold selection. A fresh model, including any learned preprocessing, fits only on the other three folds with seed `seed + i` (zero-based position). The same fit supplies validation and test probabilities; there is no refit after selecting thresholds. All models use the same split. The caller is responsible for constructing independent folds and avoiding leakage in the input features.

## How are abstentions counted?

A NaN probability is an abstention, not a negative decision or a probability of 0.5. `support` gives the total row count `n`, the count with a decision `scored`, and `coverage = scored/n`, both overall and by true class. Metrics exclude abstentions. Read the counts alongside the metrics: a model can look better by scoring fewer difficult sessions. For comparison, `common_support_metrics` evaluates every requested model on the intersection of their finite out-of-fold (OOF) scores. Its coverage denominator is still the entire input. Changing the model list can change this intersection.

## What is measured?

- `ap` is sklearn average precision (not trapezoidal PR area); `auc` is ROC AUC. Both use probabilities. `brier` is mean squared probability error and `log_loss` uses sklearn's binary log loss with natural logs and numerical clipping.
- At **score ≥ 0.5**, `confusion` gives TN, FP, FN and TP counts. The report includes accuracy, balanced accuracy (mean of recall and specificity), Matthews correlation (`mcc`), positive-class F1, precision, recall and specificity. Default decisions are never tuned on the test fold.
- A ratio without a denominator is `null`. ROC AUC, balanced accuracy and MCC require both true classes; AP requires at least one positive. F1 uses sklearn's zero-division value of zero for nonempty input. On empty support all performance metrics are `null` and confusion counts are zero.
- Model-level `metrics` pool OOF sessions, rather than averaging five fold metrics. Individual fold results remain available to show variation that pooling can hide.

## How to interpret the 1% and 5% operating points

For each target, sort the **scored validation negatives** from highest probability down. With `n` such negatives, allow `k = floor(n × target)` false positives, and take the (k+1)-th highest probability as the cutoff. Predict positive only for a score **strictly greater** than this cutoff (`threshold_exclusive`). Ties are never split. For example, with 20 negatives and a 5% target, one false positive is allowed. If the two largest scores tie at 0.92, neither passes. When `k = 0`, a test score above the largest validation negative can still pass.

`low_fpr["1"]` and `low_fpr["5"]` report **achieved test** FPR and TPR, confusion and decision coverage. The target bounds the empirical *validation* FPR, not the test or deployment FPR. If validation has no scored negatives, the cutoff and validation FPR are `null`; that fold contributes no decisions at that operating point but still contributes any finite probabilities to the ordinary metrics. Pooled operating-point rates sum held-out counts under each fold's own cutoff.

The 95% Wilson intervals for FPR and TPR describe binomial sampling variation conditional on the scored decisions. They do not account for training, cutoff selection or correlated sessions. `fold_spread` reports the sample standard deviation across defined outer-fold metrics and the number of contributing folds. It is descriptive, not a confidence interval for model performance.

## Outputs and timing

- `out/<model>/oof.csv`: one row per input, in original order, including file, fold, true label, probability, the default decision, two low-FPR decisions and the common-support flag. Blank scores or decisions mean unavailable, never zero.
- `out/<model>/summary.json`: pooled and common-support results, per-fold metrics, validation fold and support, operating points, uncertainty and timings. Fold `seconds` splits wall time into fit (including model creation), validation prediction plus threshold selection, and test prediction; each includes feature slicing. Model `seconds` also includes fold reporting and pooled metrics, but not common-support reporting or file writes.
- `out/summary.json`: all model reports keyed under `models`, with the base seed, epochs and batch size. The function returns the same JSON-compatible dictionary. Use a separate output directory per run to preserve earlier results.
