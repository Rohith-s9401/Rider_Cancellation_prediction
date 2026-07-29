# Shadowfax Rider-Cancellation Prediction — Model Summary

Only `train.csv` was uploaded (no `test.csv`), so I built and validated the model
using 5-fold cross-validation on the 450,000-row training set. The scoring script
is ready to run the moment you upload a `test.csv` in the same schema.

## 1. Key finding: several columns leak the answer

Before modeling, I checked every column against the target and found three that
directly encode the outcome and must be excluded:

| Column | Issue |
|---|---|
| `delivered_time` | Missing **iff** `cancelled == 1` (100% correlation — it's the outcome) |
| `cancelled_time` | Present **iff** `cancelled == 1` (literally a timestamp of the event we're predicting) |
| `pickup_time` | Missing only for cancelled orders (partial leakage — cancellation sometimes happens pre-pickup) |

These were dropped entirely. I also flagged `accept_time` as borderline: its
absence maps almost perfectly to cancellation (157/157 missing rows are
cancelled), but a rider simply hasn't accepted yet isn't really "cancelled" —
it's a live, unresolved state at prediction time. Using it risks overstating
real-world performance, so I trained two versions:

- **Conservative model** — usable right at order allotment, before the rider
  has necessarily accepted. Excludes anything accept-time-derived.
- **Full model** — usable once the rider has accepted, includes accept-delay
  and an "accept still missing" flag.

## 2. Features used (all known at or before the prediction point)

- Order timing: hour of day, day of week, allotment delay
- Route: first-mile / last-mile distance, ratio, total distance
- Rider history: lifetime order count, session time, rolling delivered/undelivered
  counts and rates
- Reassignment context: whether this allotment is itself a reassignment, method, reason flag
- *(Full model only)* accept delay, accept-still-missing flag

**Caveat on rider running-total columns** (`alloted_orders`, `delivered_orders`,
`undelivered_orders`): it's ambiguous from the data whether these are snapshotted
*before* or *after* the current order's outcome is folded in. If they include the
current order, that's a mild leak. I tested a stripped-down model without them
(see results below) so you can see the performance gap this explains.

## 3. Results (5-fold stratified CV, AUC-ROC)

| Model | OOF AUC | Notes |
|---|---|---|
| Full (with accept-time info) | **0.7485** | Highest signal, but only usable after acceptance |
| Conservative (no accept-time info) | **0.7401** | Usable immediately at allotment — recommended for production |
| Safe/minimal (no rider running-totals either) | **0.6725** | Lower bound if running-total columns are excluded out of caution |

Target is heavily imbalanced (1.16% cancelled), so I used `scale_pos_weight` in
LightGBM rather than resampling, and evaluated with AUC-ROC per the competition spec.

Top predictors (full model): rider's delivered-order rate, whether this
allotment is a reassignment, lifetime order count, reassignment method, and
allotment delay.

## 4. Files

- `features.py` — leakage-aware feature engineering (shared by all scripts)
- `train_model.py` — reproduces the CV experiment above (full / conservative / prints importances)
- `final_train_and_save.py` — fits both models on 100% of train.csv and saves them; also has `predict_on_test()` for scoring a future test.csv
- `model_conservative.txt`, `model_full.txt` — trained LightGBM boosters, ready to load
- `results.json` — fold-by-fold AUCs and feature importances

## 5. To score a test set once you have one

```python
from final_train_and_save import predict_on_test, CONSERVATIVE_COLS
predict_on_test('test.csv', 'model_conservative.txt', CONSERVATIVE_COLS, 'submission.csv')
```
This writes a `submission.csv` with `order_id, cancelled` (predicted probability),
matching the required submission format.

## 6. Recommendation

Use the **conservative model** for production — it doesn't depend on accept-time
being resolved, so it can trigger a reassignment recommendation the moment an
order is allotted, which is what actually gives you time to act. Treat the
0.75 AUC full-model number as an upper bound rather than a deployment target,
since some of its lift comes from a feature (`accept_missing`) that's only
knowable once the failure is already underway.
