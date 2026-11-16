"""Data checks and forward-in-time splits with label-availability purging."""
import pandas as pd

from .features import AllocationFeatures


def validate_training(raw):
    required = {"order_id", "cancelled", "delivered_time", "cancelled_time"}
    if not required.issubset(raw):
        raise ValueError(f"Missing training fields: {sorted(required - set(raw))}")
    if raw.order_id.isna().any() or not raw.cancelled.isin([0, 1]).all():
        raise ValueError("Order IDs must be present and target must be binary")
    deduplicated = raw.drop_duplicates().copy()
    if deduplicated.order_id.duplicated().any():
        raise ValueError("Conflicting records for the same order_id require review")
    AllocationFeatures().transform(deduplicated)
    allocation = pd.to_datetime(deduplicated.allot_time)
    outcome = pd.to_datetime(deduplicated.delivered_time.where(
        deduplicated.cancelled.eq(0), deduplicated.cancelled_time), errors="coerce")
    if outcome.isna().any():
        raise ValueError("Every labeled row needs an outcome timestamp")
    eligible = outcome > allocation
    clean = deduplicated.loc[eligible].copy()
    return clean, {
        "raw_rows": len(raw), "exact_duplicates_removed": len(raw) - len(deduplicated),
        "outcome_already_known_at_allocation_removed": int((~eligible).sum()),
        "eligible_rows": len(clean), "cancellation_rate": float(clean.cancelled.mean()),
    }


def chronological_split(frame):
    allocation = pd.to_datetime(frame.allot_time)
    days = pd.DatetimeIndex(allocation.dt.normalize().unique()).sort_values()
    if len(days) < 5:
        raise ValueError("At least five allocation dates required for temporal evaluation")
    val_start = days[int(len(days) * .6)]
    holdout_start = days[int(len(days) * .8)]
    outcome = pd.to_datetime(frame.delivered_time.where(
        frame.cancelled.eq(0), frame.cancelled_time))
    train_window = allocation < val_start
    val_window = (allocation >= val_start) & (allocation < holdout_start)
    masks = [train_window & (outcome < val_start),
             val_window & (outcome < holdout_start), allocation >= holdout_start]
    splits = [frame.loc[mask].copy() for mask in masks]
    if any(part.cancelled.nunique() != 2 for part in splits):
        raise ValueError("Each chronological split must contain both target classes")
    metadata = {"validation_start": str(val_start), "holdout_start": str(holdout_start),
                "train_labels_purged": int((train_window & (outcome >= val_start)).sum()),
                "validation_labels_purged": int((val_window & (outcome >= holdout_start)).sum())}
    for name, part in zip(["train", "validation", "holdout"], splits):
        metadata[name] = {"rows": len(part), "cancellations": int(part.cancelled.sum()),
                          "cancellation_rate": float(part.cancelled.mean()),
                          "allocation_min": str(pd.to_datetime(part.allot_time).min()),
                          "allocation_max": str(pd.to_datetime(part.allot_time).max())}
    return (*splits, metadata)
