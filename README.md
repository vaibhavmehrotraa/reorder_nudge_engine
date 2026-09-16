# Reorder Nudge Engine — a Swiggy-style product + analytics case study

A self-contained project that walks a reorder-nudge feature end to end: PRD → synthetic data →
rule-based trigger engine with event logging → a holdout-controlled incrementality experiment → an
analysis dashboard. On a feature modeled after how Swiggy's real order-frequency mechanics work (no real Swiggy data was used).

**[Live dashboard](./dashboard/index.html)** — open `dashboard/index.html` directly, or enable GitHub
Pages on this repo (Settings → Pages → deploy from `/dashboard`) for a shareable URL.

## The idea

Repeat orders — the same user ordering the same dish from the same restaurant on a predictable occasion —
are Swiggy's cheapest source of order frequency, but the app makes every order feel like a fresh discovery
journey. This project specs and simulates a rule-based **reorder nudge**: a feature that detects a
habitual or contextually-likely reorder moment (day-of-week pattern, overdue cadence, or a rainy-evening
comfort-food craving) and offers a one-tap path back to that order.

Because there's no access to real Swiggy data, the project builds its own realistic substitute: a
synthetic order-history generator calibrated to the publicly documented shape of the
[Instacart Market Basket Analysis](https://www.kaggle.com/datasets/lakshaymalhotra/instacart-market-basket-analysis)
dataset (reorder cadence, weekly periodicity) and Kaggle food-delivery datasets (cuisine mix, order
values). Every downstream number — the funnel, the experiment result, the dashboard — is computed the
same way it would be from real production data; only the underlying order history is simulated.

## What's in here

| Path | What it is |
| --- | --- |
| `PRD.md` | The product spec: problem, goals & metrics tree, scope, trigger logic, functional requirements, experiment design, risks. |
| `scripts/generate_synthetic_swiggy_data.py` | Generates a calibrated synthetic order-history dataset (users, restaurants, orders). |
| `scripts/simulate_trigger_engine.py` | Runs the rule-based trigger engine day by day over that history, splits users into treatment/holdout, and logs the full event stream (`trigger_evaluated` → `nudge_sent` → `nudge_opened` → `reorder_placed`). |
| `scripts/analyze_experiment.py` | Computes the funnel, the incrementality test (Welch's t-test + bootstrap CI), and cohort cuts from the event log. |
| `dashboard/index.html` | Self-contained HTML dashboard — the experiment readout, funnel, trigger-type breakdown, and cohort lift. No build step; open the file or host it as a static page. |
| `sample_data/` | Small trimmed samples of each pipeline stage's output, so you can see the data shape without running anything. |
| `analysis/` | The computed `analysis_summary.json` / `.txt` behind the dashboard, from one full run. |

## Running it yourself

```bash
pip install -r requirements.txt

# 1. Generate synthetic order history
python scripts/generate_synthetic_swiggy_data.py --n-users 5000 --n-restaurants 300 --sim-days 180 \
  --output-dir ./synthetic_data

# 2. Run the trigger engine + event logger over it
python scripts/simulate_trigger_engine.py --data-dir ./synthetic_data --output-dir ./trigger_engine_output

# 3. Compute the experiment readout
python scripts/analyze_experiment.py --data-dir ./synthetic_data --trigger-dir ./trigger_engine_output \
  --output-dir ./analysis_output
```

Then paste the new `analysis_output/analysis_summary.json` into `dashboard/index.html`'s `const DATA = ...`
line to refresh the dashboard with your own run.

## What the current run found

On a 5,000-user, 180-day simulation: treatment users averaged **11.20 orders/user** post-eligibility vs.
**10.27 for holdout** (+9.1% relative lift), but the effect landed at **p = 0.051** — just above the
conventional significance threshold, with a bootstrap 95% CI of **[0.02, 1.89]**. That's an honest
"directionally positive, not yet proven" result, not a clean win, which is the more realistic outcome to
practice interpreting than a manufactured slam-dunk. The contextual (rainy-day comfort food) trigger
converted best per open (24.5%) despite the cadence-deviation trigger sending the most volume — a
reach-vs-quality tradeoff worth prioritizing around. Full breakdown in the dashboard.

## Why this project

Built as a personal skill-building exercise to practice product development (PRD writing, scoping,
trigger/flow design) and product analytics strategy (metrics trees, instrumentation, holdout-controlled
experimentation, incrementality vs. cannibalization, cohort analysis) on the same feature, rather than
treating them as separate skills.
