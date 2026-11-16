import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from rider_cancellation.data import chronological_split, validate_training
from rider_cancellation.evaluation import choose_threshold, metrics
from rider_cancellation.features import AllocationFeatures
from rider_cancellation.predict import predict_frame


def orders(n=60):
    created = pd.date_range("2021-01-01", periods=n, freq="4h")
    y = np.arange(n) % 2
    allocation = created + pd.Timedelta(minutes=5)
    resolved = allocation + pd.Timedelta(minutes=30)
    return pd.DataFrame({"order_id": np.arange(n), "order_time": created,
        "allot_time": allocation, "first_mile_distance": np.ones(n),
        "last_mile_distance": np.ones(n) * 2, "cancelled": y,
        "delivered_time": pd.Series(resolved).where(y == 0),
        "cancelled_time": pd.Series(resolved).where(y == 1)})


def test_future_outcomes_and_identifiers_cannot_change_features():
    raw = orders()
    expected = AllocationFeatures().fit_transform(raw)
    raw["cancelled"] = 1 - raw.cancelled
    for col in ["pickup_time", "delivered_time", "cancelled_time", "accept_time",
                "reassignment_reason", "session_time", "delivered_orders", "rider_id", "order_id"]:
        raw[col] = "future information"
    pd.testing.assert_frame_equal(expected, AllocationFeatures().transform(raw))


def test_feature_contract_and_zero_distance():
    raw = orders()
    raw.loc[0, ["first_mile_distance", "last_mile_distance"]] = 0
    assert np.isnan(AllocationFeatures().transform(raw).loc[0, "first_mile_share"])
    with pytest.raises(ValueError, match="Missing required"):
        AllocationFeatures().transform(raw.drop(columns="allot_time"))
    raw.loc[0, "allot_time"] = pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError, match="precedes"):
        AllocationFeatures().transform(raw)


def test_dedup_conflict_and_resolved_outcomes():
    raw = orders()
    clean, audit = validate_training(pd.concat([raw, raw.iloc[[0]]]))
    assert len(clean) == len(raw) and audit["exact_duplicates_removed"] == 1
    conflict = raw.iloc[[0]].copy()
    conflict["first_mile_distance"] = 4
    with pytest.raises(ValueError, match="Conflicting"):
        validate_training(pd.concat([raw, conflict]))
    raw.loc[0, "delivered_time"] = raw.loc[0, "allot_time"]
    clean, audit = validate_training(raw)
    assert audit["outcome_already_known_at_allocation_removed"] == 1


def test_time_boundaries_and_label_purging():
    raw = orders(72)
    _, _, _, initial = chronological_split(raw)
    boundary = pd.Timestamp(initial["validation_start"])
    idx = raw.index[pd.to_datetime(raw.allot_time) < boundary][-1]
    outcome_col = "cancelled_time" if raw.loc[idx, "cancelled"] else "delivered_time"
    raw.loc[idx, outcome_col] = boundary + pd.Timedelta(hours=1)
    train, val, holdout, meta = chronological_split(raw)
    assert meta["train_labels_purged"] == 1
    assert idx not in train.index
    assert pd.to_datetime(train.allot_time).max() < pd.to_datetime(val.allot_time).min()
    assert pd.to_datetime(val.allot_time).max() < pd.to_datetime(holdout.allot_time).min()
    assert not set(train.order_id) & set(holdout.order_id)


def test_cost_threshold_matches_brute_force_including_ties():
    y = np.array([0, 1, 0, 1, 0, 0])
    p = np.array([.1, .2, .2, .8, .9, .9])
    selected = choose_threshold(y, p, 5, 1)
    def cost(t):
        return 5 * ((y == 1) & (p < t)).sum() + ((y == 0) & (p >= t)).sum()
    assert cost(selected) == min(cost(t) for t in [*np.unique(p), np.nextafter(1., 2.)])
    with pytest.raises(ValueError):
        choose_threshold(y, p, 0, 1)


def test_serialized_model_scores_same_and_preserves_order(tmp_path):
    raw = orders()
    pipeline = Pipeline([("features", AllocationFeatures()), ("model", DummyClassifier())])
    pipeline.fit(raw, raw.cancelled)
    bundle = {"pipeline": pipeline, "threshold": .4}
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    data = raw.iloc[::-1].drop(columns=["cancelled", "delivered_time", "cancelled_time"])
    first = predict_frame(bundle, data)
    second = predict_frame(joblib.load(path), data)
    pd.testing.assert_frame_equal(first, second)
    assert first.order_id.tolist() == data.order_id.tolist()
    with pytest.raises(ValueError, match="unique"):
        predict_frame(bundle, pd.concat([data, data.iloc[[0]]]))


def test_metrics_match_expected_confusion_counts():
    m = metrics([0, 1, 0, 1], [.1, .8, .7, .2], .5)
    assert [m[k] for k in ["tn", "fp", "fn", "tp"]] == [1, 1, 1, 1]
    assert m["precision"] == .5 and m["recall"] == .5
