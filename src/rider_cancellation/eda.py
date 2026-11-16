"""Descriptive cohorts with sample counts and Wilson intervals, not causal effects."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cohort_table(frame, bucket):
    table = frame.groupby(bucket, observed=True).cancelled.agg(rows="size", cancellations="sum")
    n, events = table.rows, table.cancellations
    rate, z = events / n, 1.96
    center = (rate + z*z/(2*n)) / (1+z*z/n)
    half = z*np.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1+z*z/n)
    table["cancellation_rate"] = rate
    table["wilson_lower"] = center-half
    table["wilson_upper"] = center+half
    return table


def generate_cohorts(frame, reports):
    from pathlib import Path
    reports = Path(reports)
    delay = (pd.to_datetime(frame.allot_time) - pd.to_datetime(frame.order_time)).dt.total_seconds()/60
    groups = {
        "allocation_delay": pd.cut(delay, [-.001, .5, 1, 2, 5, 10, 30, np.inf]),
        "first_mile_distance": pd.cut(frame.first_mile_distance, [-.001, .5, 1, 2, 3, 5, np.inf]),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, bucket) in zip(axes, groups.items()):
        table = cohort_table(frame, bucket)
        table.to_csv(reports / f"cohort_{name}.csv")
        y = table.cancellation_rate.to_numpy()*100
        lower = table.wilson_lower.to_numpy()*100
        upper = table.wilson_upper.to_numpy()*100
        x = np.arange(len(table))
        ax.errorbar(x, y, yerr=np.vstack([np.maximum(y-lower, 0), np.maximum(upper-y, 0)]),
                    fmt="o", capsize=3, color="#166c89")
        ax.set_xticks(x, [str(v) for v in table.index], rotation=35, ha="right")
        ax.set(title=name.replace("_", " ").title(), ylabel="Cancellation rate (%)")
    fig.suptitle("Descriptive cohorts: 95% Wilson intervals; no adjustment for time/rider clustering")
    fig.tight_layout(); fig.savefig(reports / "figures/cohort_rates.png", dpi=130); plt.close(fig)
