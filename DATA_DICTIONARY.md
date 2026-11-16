# Data contract and field decisions

Prediction time: immediately after the allocation represented by `allot_time`.
Verify that this is the first allocation event and that distances are already known.
Unknown units and timezone must be confirmed before production use.

| Raw field | Meaning inferred from name | Treatment |
|---|---|---|
| order_id | Order identifier | Audit, deduplication, output joins; never a feature |
| order_time | Order creation timestamp | Feature: creation hour; allocation delay |
| order_date | Redundant date | Excluded; derive from timestamp |
| allot_time | Allocation timestamp | Prediction timestamp; cyclic hour, weekday, weekend, delay |
| accept_time | Rider acceptance timestamp | Excluded: may occur after prediction |
| pickup_time | Pickup timestamp; train only | Excluded: downstream event |
| delivered_time | Delivery timestamp; train only | Label-resolution audit and purge only |
| cancelled_time | Cancellation timestamp; train only | Label-resolution audit and purge only |
| rider_id | Rider identifier | Excluded: identity and cold-start dependence |
| first_mile_distance | Distance to pickup (inferred) | Feature; nonnegative; units unconfirmed |
| last_mile_distance | Delivery-leg distance (inferred) | Feature; nonnegative; units unconfirmed |
| alloted_orders | Allocation count | Excluded: aggregation window and snapshot timing unknown |
| delivered_orders | Delivery count | Excluded: aggregation window and snapshot timing unknown |
| undelivered_orders | Undelivered count | Excluded: aggregation window and snapshot timing unknown |
| lifetime_order_count | Rider lifetime count (inferred) | Excluded: snapshot timing unknown |
| reassignment_method | Reassignment process | Excluded: may be downstream of allocation |
| reassignment_reason | Reassignment explanation | Excluded: may reveal cancellation workflow |
| reassigned_order | Reassignment flag | Excluded: timing ambiguous |
| session_time | Session duration (inferred) | Excluded: units and snapshot timing unknown |
| cancelled | Binary target | Training/evaluation label only |

Engineered numeric features: first/last/total distance, first-mile share, allocation
delay in minutes and log delay, allocation hour sine/cosine, allocation weekday,
weekend indicator, and order hour. Missing distances remain missing for histogram
boosting; logistic regression imputes medians learned on training only. A zero
total distance creates a missing share rather than division by zero.

Invalid/missing timestamps, negative distances, allocation before creation,
nonbinary labels, conflicting duplicate orders, and missing resolution timestamps
fail training. Exact duplicates are removed. Outcomes already resolved at allocation
are excluded from this prediction population. Inference rejects duplicate/missing IDs.

Only the four feature input fields plus `order_id` are required for scoring. Extra
columns are ignored by the explicit feature allowlist. Original raw data is preserved.
