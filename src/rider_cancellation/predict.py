"""Batch scoring: python -m rider_cancellation.predict --help."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def predict_frame(bundle, data):
    if "order_id" not in data or data.order_id.isna().any() or data.order_id.duplicated().any():
        raise ValueError("Scoring requires unique, nonmissing order_id values")
    p = bundle["pipeline"].predict_proba(data)[:, 1]
    if not np.isfinite(p).all():
        raise ValueError("Model returned nonfinite probabilities")
    return pd.DataFrame({"order_id": data.order_id.to_numpy(),
                         "cancellation_probability": p,
                         "cancelled": (p >= bundle["threshold"]).astype(int)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Only load trusted local artifacts: joblib/pickle can execute code.
    bundle = joblib.load(args.model)
    result = predict_frame(bundle, pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result):,} orders -> {args.output}")


if __name__ == "__main__":
    main()
