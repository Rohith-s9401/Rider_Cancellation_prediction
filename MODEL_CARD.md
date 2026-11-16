# Model card

**Purpose:** rank orders by cancellation risk immediately after allocation, for
optional operations review. Cancellation initiator and causality are unknown.

**Model:** selected by validation average precision from a prior baseline, logistic
regression, and two histogram gradient-boosting configurations. See
`reports/metrics.json` for the actual selected model and measured results.

**Training:** January–February 2021 supplied CSVs, exact deduplication, explicit
feature allowlist, forward date splits, and resolution-time purging. All transforms
live in the serialized sklearn pipeline. No target encoding or random oversampling.

**Operating threshold:** selected on validation under illustrative costs. Cost
values are not established business facts. Reconfirm capacity, probability quality,
and threshold after any refit. Calibration is diagnosed but not corrected.

**Evaluation:** frozen `evaluation_model.joblib` on later labeled orders. Production
`model.joblib` is refit on all labels resolved before the earliest test allocation.
Holdout numbers do not constitute independent evaluation of that refit artifact.
Test labels are absent. No leaderboard score or realized business benefit is claimed.

**Limitations:** short time window, uncertain field definitions and licensing,
unverified distance availability, temporal changes, repeat riders, missing location
and operational context. No causal interpretation, subgroup fairness conclusion,
or assurance for unseen riders/cities/seasons. Not suitable for punitive rider decisions.

**Use conditions:** validate the upstream event contract and data permissions;
extend temporal backtesting; estimate intervention economics; prospectively test
with human oversight; monitor delayed-label ranking and probability quality.

**Persistence:** joblib is unsafe for untrusted inputs. Load only artifacts built
from trusted code. Reproduce with the supplied runtime version lock; sklearn
artifacts are not guaranteed portable between package versions.
