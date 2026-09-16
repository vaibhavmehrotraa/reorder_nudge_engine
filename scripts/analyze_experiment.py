"""
Experiment analysis for the Reorder Nudge Engine project (Swiggy PRD).

Reads the outputs of generate_synthetic_swiggy_data.py and simulate_trigger_engine.py
and produces the numbers a real experiment readout + dashboard would need:

  1. The nudge funnel (trigger fired -> sent -> opened -> incremental order),
     overall and by trigger type, plus WHY nudges got suppressed before sending
     (frequency cap, inactive restaurant, already ordered today) — these are
     the guardrail/operational numbers, not just the funnel.
  2. The incrementality experiment: treatment vs. holdout orders-per-user,
     measured from each user's own eligibility date forward (never counting
     pre-randomization history as an outcome), with a Welch's t-test, a
     bootstrap 95% CI on the mean difference, and Cohen's d for effect size.
  3. Cohort cuts by frequency segment x experiment group.

Writes analysis_summary.json (small, dashboard-ready) and analysis_summary.txt
(human-readable).

Usage:
  python analyze_experiment.py --data-dir ./synthetic_data_sample --trigger-dir ./trigger_engine_sample
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


def load_inputs(data_dir, trigger_dir):
    users = pd.read_csv(os.path.join(data_dir, "users.csv"))
    events = pd.read_csv(os.path.join(trigger_dir, "events.csv"), parse_dates=["event_date"])
    orders = pd.read_csv(os.path.join(trigger_dir, "orders_with_nudges.csv"), parse_dates=["order_datetime"])
    orders["order_date"] = orders["order_datetime"].dt.date
    events["event_date"] = events["event_date"].dt.date
    return users, events, orders


def compute_funnel(events):
    fired = events[(events["event_type"] == "trigger_evaluated") & (events["note"] == "fired")]
    sent = events[events["event_type"] == "nudge_sent"]
    opened = events[events["event_type"] == "nudge_opened"]
    placed = events[events["event_type"] == "reorder_placed"]

    def by_trigger(df):
        return df.groupby("trigger_type").size().to_dict()

    overall = {
        "trigger_fired": int(len(fired)),
        "nudge_sent": int(len(sent)),
        "nudge_opened": int(len(opened)),
        "reorder_placed": int(len(placed)),
    }
    by_type = {}
    for t in sorted(events["trigger_type"].dropna().unique()):
        by_type[t] = {
            "trigger_fired": int((fired["trigger_type"] == t).sum()),
            "nudge_sent": int((sent["trigger_type"] == t).sum()),
            "nudge_opened": int((opened["trigger_type"] == t).sum()),
            "reorder_placed": int((placed["trigger_type"] == t).sum()),
        }

    suppressed = events[(events["event_type"] == "trigger_evaluated") & (events["note"] != "fired")]
    suppression_reasons = (
        suppressed["note"].str.replace("suppressed:", "", regex=False).value_counts().to_dict()
    )
    suppression_reasons = {k: int(v) for k, v in suppression_reasons.items()}

    return {"overall": overall, "by_trigger_type": by_type, "suppression_reasons": suppression_reasons}


def compute_experiment(users, events, orders, n_bootstrap=5000, seed=11):
    eligible_since = events.groupby("user_id")["event_date"].min().rename("eligible_since")
    group = events.groupby("user_id")["experiment_group"].first().rename("experiment_group")
    exp_users = pd.concat([eligible_since, group], axis=1).reset_index()
    exp_users = exp_users.merge(users[["user_id", "frequency_segment"]], on="user_id", how="left")

    orders_idx = orders.merge(exp_users[["user_id", "eligible_since"]], on="user_id", how="inner")
    orders_post = orders_idx[orders_idx["order_date"] >= orders_idx["eligible_since"]]

    per_user_orders = orders_post.groupby("user_id").size().rename("total_orders")
    per_user_organic = (
        orders_post[orders_post["source"] == "organic"].groupby("user_id").size().rename("organic_orders")
    )
    per_user_incremental = (
        orders_post[orders_post["source"] == "nudge_incremental"].groupby("user_id").size().rename("incremental_orders")
    )

    df = exp_users.set_index("user_id").join([per_user_orders, per_user_organic, per_user_incremental])
    df[["total_orders", "organic_orders", "incremental_orders"]] = df[
        ["total_orders", "organic_orders", "incremental_orders"]
    ].fillna(0)

    treatment = df[df["experiment_group"] == "treatment"]["total_orders"].to_numpy()
    holdout = df[df["experiment_group"] == "holdout"]["total_orders"].to_numpy()

    t_stat, p_value = stats.ttest_ind(treatment, holdout, equal_var=False)
    mean_diff = float(treatment.mean() - holdout.mean())

    pooled_std = np.sqrt(((treatment.std(ddof=1) ** 2) + (holdout.std(ddof=1) ** 2)) / 2)
    cohens_d = float(mean_diff / pooled_std) if pooled_std > 0 else 0.0

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        t_sample = rng.choice(treatment, size=len(treatment), replace=True)
        h_sample = rng.choice(holdout, size=len(holdout), replace=True)
        boot_diffs[i] = t_sample.mean() - h_sample.mean()
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    cohort = (
        df.groupby(["frequency_segment", "experiment_group"])["total_orders"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_total_orders", "count": "n_users"})
    )
    cohort["mean_total_orders"] = cohort["mean_total_orders"].round(3)

    return {
        "n_treatment": int(len(treatment)),
        "n_holdout": int(len(holdout)),
        "mean_orders_treatment": round(float(treatment.mean()), 3),
        "mean_orders_holdout": round(float(holdout.mean()), 3),
        "mean_incremental_orders_treatment": round(float(df[df["experiment_group"] == "treatment"]["incremental_orders"].mean()), 3),
        "measured_lift": round(mean_diff, 3),
        "relative_lift_pct": round(100 * mean_diff / holdout.mean(), 2) if holdout.mean() > 0 else None,
        "t_stat": round(float(t_stat), 3),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
        "cohens_d": round(cohens_d, 3),
        "bootstrap_ci_95": [round(float(ci_low), 3), round(float(ci_high), 3)],
        "cohort_breakdown": cohort.to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--trigger-dir", required=True)
    parser.add_argument("--output-dir", default="./analysis_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    users, events, orders = load_inputs(args.data_dir, args.trigger_dir)

    funnel = compute_funnel(events)
    experiment = compute_experiment(users, events, orders)

    sim_start = str(orders["order_date"].min())
    sim_end = str(orders["order_date"].max())

    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "simulation_window": {"start": sim_start, "end": sim_end},
        "funnel": funnel,
        "experiment": experiment,
    }

    with open(os.path.join(args.output_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append(f"Simulation window: {sim_start} to {sim_end}")
    lines.append("")
    lines.append("FUNNEL (overall):")
    for k, v in funnel["overall"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("FUNNEL by trigger type:")
    for t, stats_ in funnel["by_trigger_type"].items():
        lines.append(f"  {t}: {stats_}")
    lines.append("")
    lines.append("Suppression reasons (nudges evaluated but not sent):")
    for k, v in funnel["suppression_reasons"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("EXPERIMENT (post-eligibility window):")
    for k, v in experiment.items():
        if k != "cohort_breakdown":
            lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Cohort breakdown (frequency segment x group):")
    for row in experiment["cohort_breakdown"]:
        lines.append(f"  {row}")

    text = "\n".join(lines)
    with open(os.path.join(args.output_dir, "analysis_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
