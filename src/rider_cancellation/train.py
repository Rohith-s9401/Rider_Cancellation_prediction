"""Reproducible training, forward validation, reporting, and batch prediction."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import chronological_split, validate_training
from .evaluation import choose_threshold, metrics
from .eda import generate_cohorts
from .features import AllocationFeatures
from .predict import predict_frame

SEED = 42


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def candidates():
    return {
        "prior_baseline": Pipeline([("features", AllocationFeatures()),
                                     ("model", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline([
            ("features", AllocationFeatures()), ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=600, random_state=SEED))]),
        **{f"gradient_boosting_{leaves}_leaves": Pipeline([
            ("features", AllocationFeatures()),
            ("model", HistGradientBoostingClassifier(
                max_iter=140, learning_rate=.08, max_leaf_nodes=leaves,
                min_samples_leaf=100, l2_regularization=10,
                early_stopping=False, random_state=SEED))]) for leaves in (15, 31)},
    }


def drift_table(reference, current):
    rows = []
    for col in reference:
        a, b = reference[col], current[col]
        edges = np.unique(np.nanquantile(a, np.linspace(0, 1, 11)))
        if len(edges) < 2:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        ac = np.r_[np.histogram(a.dropna(), bins=edges)[0], a.isna().sum()] + .5
        bc = np.r_[np.histogram(b.dropna(), bins=edges)[0], b.isna().sum()] + .5
        ap, bp = ac / ac.sum(), bc / bc.sum()
        rows.append({"feature": col, "psi": float(np.sum((bp - ap) * np.log(bp / ap))),
                     "reference_missing": float(a.isna().mean()),
                     "current_missing": float(b.isna().mean())})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def make_plots(raw, holdout, probability, threshold, validation_scores, importance, figures):
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    daily = raw.groupby(pd.to_datetime(raw.order_time).dt.date).cancelled.agg(["size", "mean"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily["mean"] * 100, marker="o", color="#166c89")
    ax.set(ylabel="Cancellation rate (%)", title="Observed cancellation rate by order date")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(figures / "daily_cancellations.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    raw.isna().mean().sort_values().plot.barh(ax=ax, color="#166c89")
    ax.set(xlabel="Fraction missing", title="Missingness audit: raw training fields")
    fig.tight_layout(); fig.savefig(figures / "missingness.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    PrecisionRecallDisplay.from_predictions(holdout.cancelled, probability, ax=axes[0])
    axes[0].axhline(holdout.cancelled.mean(), color="gray", linestyle="--", label="Prevalence")
    axes[0].legend(); axes[0].set_title("Untouched temporal holdout")
    observed, predicted = calibration_curve(holdout.cancelled, probability, n_bins=10, strategy="quantile")
    axes[1].plot(predicted, observed, "o-", color="#166c89")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    limit = max(float(max(observed)), float(max(predicted)), .02) * 1.1
    axes[1].set(xlim=(0, limit), ylim=(0, limit), xlabel="Mean predicted probability",
                ylabel="Observed cancellation rate", title="Reliability by probability decile")
    fig.tight_layout(); fig.savefig(figures / "holdout_pr_calibration.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(holdout.cancelled, probability >= threshold,
                                           ax=ax, colorbar=False, values_format="d")
    ax.set_title("Validation-selected threshold on holdout")
    fig.tight_layout(); fig.savefig(figures / "confusion_matrix.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    table = pd.DataFrame(validation_scores).T.sort_values("average_precision")
    table.average_precision.plot.barh(ax=ax, color="#166c89")
    ax.set(xlabel="Average precision", title="Candidate comparison: validation only")
    fig.tight_layout(); fig.savefig(figures / "model_comparison.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ordered = importance.sort_values("mean_ap_decrease")
    ax.barh(ordered.feature, ordered.mean_ap_decrease, xerr=ordered.std_ap_decrease, color="#166c89")
    ax.set(xlabel="Average precision decrease after permutation", title="Validation feature importance (3 repeats)")
    fig.tight_layout(); fig.savefig(figures / "feature_importance.png"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--false-negative-cost", type=float, default=10.)
    parser.add_argument("--false-positive-cost", type=float, default=1.)
    args = parser.parse_args()
    out = args.output_dir
    reports, figures, models, predictions = [out / p for p in
        ("reports", "reports/figures", "models", "predictions")]
    for path in (reports, figures, models, predictions):
        path.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    template = pd.read_csv(args.data_dir / "sample_submission.csv")
    clean, audit = validate_training(raw)
    if not test.order_id.is_unique or test.order_id.isna().any():
        raise ValueError("Test IDs must be unique and present")
    if set(clean.order_id) & set(test.order_id):
        raise ValueError("Training and test IDs overlap")
    if list(template.columns) != ["order_id", "cancelled"] or not template.order_id.equals(test.order_id):
        raise ValueError("Submission template differs from test ID order or expected schema")
    train, val, holdout, split = chronological_split(clean)
    audit["raw_columns"] = list(raw.columns)
    audit["test_rows"] = len(test)
    audit["train_only_columns"] = sorted(set(raw) - set(test))
    audit["missing_fractions"] = raw.isna().mean().to_dict()
    audit["split"] = split
    write_json(reports / "data_audit.json", audit)
    daily = clean.groupby(pd.to_datetime(clean.order_time).dt.date).cancelled.agg(["size", "sum", "mean"])
    daily.to_csv(reports / "daily_cancellation_rates.csv")
    generate_cohorts(clean, reports)
    print(f"Eligible rows: {len(clean):,}; temporal splits: {len(train):,}/{len(val):,}/{len(holdout):,}", flush=True)
    scores, thresholds, fitted = {}, {}, {}
    with threadpool_limits(limits=4):
        for name, pipe in candidates().items():
            print(f"Fitting {name}...", flush=True)
            pipe.fit(train, train.cancelled)
            p = pipe.predict_proba(val)[:, 1]
            threshold = choose_threshold(val.cancelled, p, args.false_negative_cost, args.false_positive_cost)
            thresholds[name] = threshold
            scores[name] = metrics(val.cancelled, p, threshold)
            fitted[name] = pipe
            print(f"Validation AP = {scores[name]['average_precision']:.5f}", flush=True)
        winner = max(scores, key=lambda name: scores[name]["average_precision"])
        pipe, threshold = fitted[winner], thresholds[winner]
        # Freeze model and threshold before the first holdout evaluation.
        p = pipe.predict_proba(holdout)[:, 1]
        holdout_scores = metrics(holdout.cancelled, p, threshold)
        baseline = fitted["prior_baseline"].predict_proba(holdout)[:, 1]
        baseline_scores = metrics(holdout.cancelled, baseline, thresholds["prior_baseline"])
        joblib.dump({"pipeline": pipe, "threshold": threshold, "model_name": winner,
                     "purpose": "frozen temporal holdout evaluation", "seed": SEED},
                    models / "evaluation_model.joblib", compress=3)
        transformed = pipe.named_steps["features"].transform(val)
        sample = transformed.sample(n=min(15000, len(transformed)), random_state=SEED)
        estimator_steps = list(pipe.steps[1:])
        estimator = Pipeline(estimator_steps)
        result = permutation_importance(estimator, sample, val.loc[sample.index, "cancelled"],
                    scoring="average_precision", n_repeats=3, random_state=SEED, n_jobs=1)
        importance = pd.DataFrame({"feature": transformed.columns,
            "mean_ap_decrease": result.importances_mean, "std_ap_decrease": result.importances_std})
        importance.to_csv(reports / "feature_importance.csv", index=False)
        holdout_output = predict_frame({"pipeline": pipe, "threshold": threshold}, holdout)
        holdout_output["actual_cancelled"] = holdout.cancelled.to_numpy()
        holdout_output["allocation_date"] = pd.to_datetime(holdout.allot_time).dt.date.to_numpy()
        holdout_output.to_csv(predictions / "holdout_predictions.csv", index=False)
        slices = []
        for day, group in holdout_output.groupby("allocation_date"):
            if group.actual_cancelled.nunique() == 2:
                slices.append({"allocation_date": str(day), "rows": len(group),
                    **metrics(group.actual_cancelled, group.cancellation_probability, threshold)})
        pd.DataFrame(slices).to_csv(reports / "holdout_daily_metrics.csv", index=False)
        costs = []
        for fn_cost in (1, 5, 10, 25, 50):
            t = choose_threshold(val.cancelled, pipe.predict_proba(val)[:, 1], fn_cost, 1)
            m = metrics(val.cancelled, pipe.predict_proba(val)[:, 1], t)
            costs.append({"false_negative_cost": fn_cost, "false_positive_cost": 1,
                "validation_cost_per_1000": (fn_cost * m["fn"] + m["fp"]) / len(val) * 1000,
                **m})
        pd.DataFrame(costs).to_csv(reports / "validation_cost_sensitivity.csv", index=False)
        test_start = pd.to_datetime(test.allot_time).min()
        outcomes = pd.to_datetime(clean.delivered_time.where(clean.cancelled.eq(0), clean.cancelled_time))
        refit = clean.loc[(pd.to_datetime(clean.allot_time) < test_start) & (outcomes < test_start)]
        production = candidates()[winner]
        print(f"Refitting {winner} on {len(refit):,} labels available before test allocation...", flush=True)
        production.fit(refit, refit.cancelled)
        bundle = {"pipeline": production, "threshold": threshold, "model_name": winner,
                  "purpose": "batch scoring; refit performance not independently evaluated",
                  "training_cutoff": str(test_start), "seed": SEED}
        joblib.dump(bundle, models / "model.joblib", compress=3)
        scored = predict_frame(bundle, test)
        scored.to_csv(predictions / "test_probabilities.csv", index=False)
        scored[["order_id", "cancelled"]].to_csv(predictions / "submission.csv", index=False)
    drift = drift_table(AllocationFeatures().transform(train), AllocationFeatures().transform(test))
    drift.to_csv(reports / "test_feature_drift.csv", index=False)
    payload = {"selected_model": winner, "selection_metric": "validation average precision",
        "seed": SEED, "validation": scores, "holdout": holdout_scores,
        "holdout_baseline": baseline_scores, "threshold": threshold,
        "illustrative_costs": {"false_negative": args.false_negative_cost, "false_positive": args.false_positive_cost},
        "production_refit_rows": len(refit), "production_label_cutoff": str(test_start)}
    write_json(reports / "metrics.json", payload)
    pd.DataFrame(scores).T.to_csv(reports / "validation_model_comparison.csv")
    make_plots(raw, holdout, p, threshold, scores, importance, figures)
    versions = {"python": platform.python_version(), "pandas": pd.__version__,
                "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__, "matplotlib": matplotlib.__version__}
    write_json(reports / "run_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": versions, "files": [fingerprint(args.data_dir / f) for f in
        ("train.csv", "test.csv", "sample_submission.csv")], "parameters": vars(args)})
    make_report(reports, audit, payload, drift, importance, versions)
    print(json.dumps({"selected_model": winner, "holdout": holdout_scores}, indent=2), flush=True)


def make_report(reports, audit, result, drift, importance, versions):
    h, b, split = result["holdout"], result["holdout_baseline"], audit["split"]
    cost = result["illustrative_costs"]
    model_cost = cost["false_negative"] * h["fn"] + cost["false_positive"] * h["fp"]
    no_action_cost = cost["false_negative"] * (h["tp"] + h["fn"])
    text = f"""# Rider cancellation: experimental findings

## Decision and prediction contract
Rank orders for optional human review immediately after allocation. Predict the supplied
`cancelled` label; the data does not establish who initiated cancellation or its cause.
This is a retrospective portfolio study, not evidence that an intervention prevents cancellation.
Distances are assumed to be available when the allocation event occurs; verify this upstream.

## Data and evaluation
- Raw training rows: {audit['raw_rows']:,}; test rows: {audit['test_rows']:,}.
- Exact duplicates removed: {audit['exact_duplicates_removed']}; outcomes already known at allocation removed: {audit['outcome_already_known_at_allocation_removed']}.
- Eligible cancellation prevalence: {audit['cancellation_rate']:.3%}.
- Forward validation starts {split['validation_start']}; untouched holdout starts {split['holdout_start']}.
- Training/validation labels crossing the next boundary were purged: {split['train_labels_purged']}/{split['validation_labels_purged']}.
- Fit preprocessing on training only. Select among four candidates by validation average precision.
- Select the decision threshold on validation only; freeze it before holdout evaluation.

## Measured holdout results
Selected model: **{result['selected_model']}**.

| Metric | Selected model | Prior-only baseline |
|---|---:|---:|
| Average precision | {h['average_precision']:.5f} | {b['average_precision']:.5f} |
| ROC AUC | {h['roc_auc']:.4f} | {b['roc_auc']:.4f} |
| Brier score (lower is better) | {h['brier_score']:.6f} | {b['brier_score']:.6f} |
| Log loss (lower is better) | {h['log_loss']:.6f} | {b['log_loss']:.6f} |

At threshold **{result['threshold']:.6g}**, precision is **{h['precision']:.2%}**,
recall **{h['recall']:.2%}**, F1 **{h['f1']:.4f}**, and alert rate **{h['alert_rate']:.2%}**.
The highest-risk 5% captures **{h['recall_at_top_5_percent']:.2%}** of cancellations
with **{h['precision_at_top_5_percent']:.2%}** precision (**{h['lift_at_top_5_percent']:.2f}x** prevalence).
Confusion counts: TP {h['tp']:,}, FP {h['fp']:,}, FN {h['fn']:,}, TN {h['tn']:,}.
The 5% queue is a ranking diagnostic with deterministic tie breaking, not a promised operating capacity.

![Validation comparison](figures/model_comparison.png)
![Holdout precision-recall and reliability](figures/holdout_pr_calibration.png)
![Holdout confusion matrix](figures/confusion_matrix.png)

## Illustrative operating costs
The default scenario assigns a missed cancellation {cost['false_negative']:g} cost units and
a false alert {cost['false_positive']:g}. These values are assumptions, not measured business costs.
Holdout classification loss is {model_cost:,.0f} units, compared with {no_action_cost:,.0f}
for flagging no orders. This loss is **not realized savings or ROI**: review cost for true alerts,
intervention effectiveness, and adverse effects are not modeled.
See `validation_cost_sensitivity.csv` for alternative assumptions chosen exclusively on validation.

## Interpretation and drift
The strongest validation permutation signals include
{', '.join(importance.nlargest(3, 'mean_ap_decrease').feature)}.
Permutation importance is predictive association; correlated engineered features share signal.
The largest train-to-test feature PSI is {drift.iloc[0].psi:.3f} for `{drift.iloc[0].feature}`.
PSI describes binned distribution differences and is not a statistical significance test.
No calibration mapping was fitted; inspect the reliability plot and Brier/log-loss metrics
before treating raw scores as decision-grade probabilities.

![Validation permutation importance](figures/feature_importance.png)
![Daily cancellation rates](figures/daily_cancellations.png)
![Raw missingness](figures/missingness.png)

## Leakage controls and limitations
The feature allowlist uses order/allocation times and first/last-mile distances only.
`pickup_time`, `delivered_time`, and `cancelled_time` are outcomes and excluded from features.
`accept_time` and reassignment fields may occur after first allocation and are excluded.
Counters, session time, and lifetime counts have undocumented snapshot timing and are excluded.
Outcome timestamps are used only for eligibility and boundary purging. IDs are used only for joins.
There is no location, weather, traffic, capacity, intervention, demographic, or cost ground truth.
The observation window is short; riders may repeat across periods. This evaluates later orders
in the same service, not generalization to unseen riders, cities, seasons, or years.
The supplied labels describe order cancellation, not proven rider misconduct.
Do not use the score alone for rider penalties or employment decisions.

## Artifacts and next steps
`models/evaluation_model.joblib` preserves the model used for the holdout numbers.
`models/model.joblib` refits the selected architecture on {result['production_refit_rows']:,}
labels resolved before {result['production_label_cutoff']}, retaining the validation threshold.
The refit model has no separate labeled evaluation; holdout results describe the frozen evaluation model.
`submission.csv` follows the binary-valued supplied template. The actual competition scoring rule
was not supplied; use `test_probabilities.csv` if the scoring system expects probabilities.
Test labels are unavailable, so there is no measured test-set score.

Before deployment, verify feature timestamps and distance units, expand the time window,
run rolling temporal backtests and rider-cohort checks, recalibrate with a separate recent window,
agree on review capacity and costs, then run a prospective monitored experiment.
Monitor schema failures, missingness, score distributions, queue volume, delayed-label AP,
precision/recall, calibration, and operational outcomes with an explicit rollback owner.

Run environment: {json.dumps(versions)}. Full provenance is in `run_manifest.json`.
"""
    (reports / "findings.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
