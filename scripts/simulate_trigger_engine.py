"""
Trigger engine + event logger for the Reorder Nudge Engine project (Swiggy PRD).

Reads the synthetic order history produced by generate_synthetic_swiggy_data.py and
simulates, day by day, what the PRD's rule-based trigger engine (Functional
Requirements #1-#7) would actually do:

  1. Evaluate the 3 trigger types from the PRD's "User Flow & Trigger Logic" section
     using ONLY observable order history up to that day (never hidden ground-truth
     simulation parameters — the engine has to infer patterns from data, same as a
     real one would).
  2. Apply eligibility: >=2 orders in trailing 60 days, restaurant active, not
     already ordered today, per-user frequency cap.
  3. Split users into treatment / holdout at first eligibility (PRD requirement #7)
     and keep that assignment for the whole simulation.
  4. For treatment users only, simulate whether the nudge is opened and whether it
     causes an INCREMENTAL order (one that would not have happened organically).
     A true incremental lift is baked in per trigger type so you can later check
     whether your own experiment analysis (Notebook 3 / dashboard) recovers it —
     that's a stronger portfolio signal than reporting a p-value with no ground
     truth to check it against.
  5. Holdout users are evaluated identically (so you have a clean "what would have
     fired" log for them) but never receive a nudge and never get an incremental
     order — their order stream is exactly the organic baseline.

Outputs (to --output-dir):
  events.csv              one row per event: trigger_evaluated / nudge_sent /
                           nudge_opened / reorder_placed / nudge_dismissed
  orders_with_nudges.csv  the organic orders.csv PLUS incremental orders caused by
                           nudges, tagged by `source` (organic / nudge_incremental)
  experiment_readout.txt  treatment vs. holdout orders/user, compared against the
                           true baked-in lift, as a sanity check on the whole sim

Usage:
  python simulate_trigger_engine.py --data-dir ./synthetic_data_sample
"""

import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

COMFORT_CUISINES = {"Biryani", "Chinese", "Momos", "Fast Food"}
MONSOON_MONTHS = {6, 7, 8, 9}
RAINY_PROB_MONSOON = 0.40
RAINY_PROB_OTHER = 0.08

# ---------------------------------------------------------------------------
# Tunable trigger-engine parameters (PRD "Functional Requirements": configurable
# without a redeploy — these constants are that config, just not wired to a UI).
# ---------------------------------------------------------------------------

MIN_ORDERS_TRAILING_60D = 2         # PRD "Target Users": eligibility floor
FREQUENCY_CAP_PER_WEEK = 2          # PRD FR #2

DOW_LOOKBACK_ORDERS = 4             # trigger A: look at last N orders to a restaurant
DOW_MATCH_THRESHOLD = 3             # ...fire if >= this many share today's weekday
DOW_REFIRE_COOLDOWN_DAYS = 3        # don't re-fire for the same restaurant too often

CADENCE_OVERDUE_DAYS = 2            # trigger B: fire once this many days past the
                                     # user's own typical gap
CADENCE_REFIRE_COOLDOWN_DAYS = 5

CONTEXTUAL_MIN_RAINY_ORDERS = 3     # trigger C: need this many historical rainy
                                     # orders before we trust the comfort-food signal
CONTEXTUAL_COMFORT_SHARE_THRESHOLD = 0.5
CONTEXTUAL_REFIRE_COOLDOWN_DAYS = 3

HOLDOUT_FRACTION = 0.5

# Response model: open rate and TRUE incremental-order probability (given opened),
# by trigger type. These are the "ground truth" causal effects this simulation
# bakes in — your experiment analysis should recover numbers close to these.
P_OPEN = {"day_of_week": 0.35, "cadence": 0.22, "contextual": 0.40}
P_INCREMENTAL_ORDER_GIVEN_OPEN = {"day_of_week": 0.18, "cadence": 0.10, "contextual": 0.25}


def load_data(data_dir):
    users = pd.read_csv(os.path.join(data_dir, "users.csv"))
    restaurants = pd.read_csv(os.path.join(data_dir, "restaurants.csv"))
    orders = pd.read_csv(os.path.join(data_dir, "orders.csv"), parse_dates=["order_datetime"])

    for col in ["is_active"]:
        restaurants[col] = restaurants[col].map({"True": True, "False": False}).fillna(restaurants[col])
    for col in ["is_weekend", "is_rainy", "is_regular_restaurant_for_user", "restaurant_was_active"]:
        orders[col] = orders[col].map({"True": True, "False": False}).fillna(orders[col])

    orders["order_date"] = orders["order_datetime"].dt.date
    orders = orders.sort_values("order_datetime").reset_index(drop=True)
    return users, restaurants, orders


def generate_weather_calendar(cities, start_date, end_date, seed=123):
    """
    Independent day-by-day 'actual weather' the trigger engine observes at
    evaluation time. Not derived from orders.csv (which only tells us the weather
    on days a given user happened to order) — this is a full daily calendar so
    triggers can fire on days with no organic order too.
    """
    rng = np.random.default_rng(seed)
    calendar = {}
    d = start_date
    while d <= end_date:
        is_monsoon = d.month in MONSOON_MONTHS
        p = RAINY_PROB_MONSOON if is_monsoon else RAINY_PROB_OTHER
        for city in cities:
            calendar[(d, city)] = bool(rng.random() < p)
        d += timedelta(days=1)
    return calendar


def restaurant_most_common_cuisine_lookup(restaurants):
    return restaurants.set_index("restaurant_id")[["cuisine", "is_active", "city", "name"]].to_dict("index")


class UserSimState:
    """Rolling, mutable per-user state as the simulation walks forward day by day."""

    __slots__ = [
        "user_id", "city", "history", "experiment_group", "eligible_since",
        "nudges_sent_dates", "last_dow_fire", "last_cadence_fire", "last_contextual_fire",
    ]

    def __init__(self, user_id, city):
        self.user_id = user_id
        self.city = city
        self.history = []  # list of dicts: {date, restaurant_id, cuisine, is_rainy, source}
        self.experiment_group = None
        self.eligible_since = None
        self.nudges_sent_dates = []
        self.last_dow_fire = {}
        self.last_cadence_fire = None
        self.last_contextual_fire = None

    def orders_in_trailing(self, today, days):
        cutoff = today - timedelta(days=days)
        return [o for o in self.history if cutoff <= o["date"] <= today]

    def nudges_in_trailing_week(self, today):
        cutoff = today - timedelta(days=7)
        return [d for d in self.nudges_sent_dates if cutoff <= d <= today]


def evaluate_day_of_week_trigger(state, today, restaurant_lookup):
    """Trigger A: has this user ordered from restaurant X on today's weekday in
    >= DOW_MATCH_THRESHOLD of their last DOW_LOOKBACK_ORDERS orders to X?"""
    by_restaurant = defaultdict(list)
    for o in state.history:
        by_restaurant[o["restaurant_id"]].append(o)

    for restaurant_id, orders in by_restaurant.items():
        if len(orders) < DOW_MATCH_THRESHOLD:
            continue
        recent = sorted(orders, key=lambda o: o["date"])[-DOW_LOOKBACK_ORDERS:]
        matches = sum(1 for o in recent if o["date"].weekday() == today.weekday())
        if matches < DOW_MATCH_THRESHOLD:
            continue
        already_today = any(o["date"] == today and o["restaurant_id"] == restaurant_id for o in state.history)
        if already_today:
            continue
        last_fire = state.last_dow_fire.get(restaurant_id)
        if last_fire and (today - last_fire).days < DOW_REFIRE_COOLDOWN_DAYS:
            continue
        meta = restaurant_lookup.get(restaurant_id)
        if meta is None or not meta["is_active"]:
            return {"fired": True, "eligible": False, "restaurant_id": restaurant_id,
                    "suppression_reason": "restaurant_inactive"}
        return {"fired": True, "eligible": True, "restaurant_id": restaurant_id,
                "suppression_reason": None}
    return {"fired": False, "eligible": False, "restaurant_id": None, "suppression_reason": None}


def evaluate_cadence_trigger(state, today):
    """Trigger B: user is overdue relative to their own typical inter-order gap."""
    if len(state.history) < 3:
        return {"fired": False, "eligible": False, "restaurant_id": None, "suppression_reason": None}

    dates = sorted(o["date"] for o in state.history)
    gaps = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
    typical_gap = float(np.median(gaps))
    last_order_date = dates[-1]
    days_since = (today - last_order_date).days

    if days_since < typical_gap + CADENCE_OVERDUE_DAYS:
        return {"fired": False, "eligible": False, "restaurant_id": None, "suppression_reason": None}
    if state.last_cadence_fire and (today - state.last_cadence_fire).days < CADENCE_REFIRE_COOLDOWN_DAYS:
        return {"fired": False, "eligible": False, "restaurant_id": None, "suppression_reason": None}

    counts = defaultdict(int)
    for o in state.history:
        counts[o["restaurant_id"]] += 1
    top_restaurant = max(counts, key=counts.get)
    return {"fired": True, "eligible": True, "restaurant_id": top_restaurant, "suppression_reason": None}


def evaluate_contextual_trigger(state, today, is_rainy_today, restaurant_lookup):
    """Trigger C: it's raining today AND this user's history shows an elevated
    tendency to order comfort food on rainy days."""
    if not is_rainy_today:
        return {"fired": False, "eligible": False, "restaurant_id": None, "suppression_reason": None}

    rainy_orders = [o for o in state.history if o["is_rainy"]]
    if len(rainy_orders) < CONTEXTUAL_MIN_RAINY_ORDERS:
        return {"fired": False, "eligible": False, "restaurant_id": None, "suppression_reason": None}

    comfort_rainy = [o for o in rainy_orders if o["cuisine"] in COMFORT_CUISINES]
    if len(comfort_rainy) / len(rainy_orders) < CONTEXTUAL_COMFORT_SHARE_THRESHOLD:
        return {"fired": False, "eligible": False, "restaurant_id": None, "suppression_reason": None}

    already_today = any(o["date"] == today for o in state.history)
    if already_today:
        return {"fired": True, "eligible": False, "restaurant_id": None, "suppression_reason": "already_ordered_today"}
    if state.last_contextual_fire and (today - state.last_contextual_fire).days < CONTEXTUAL_REFIRE_COOLDOWN_DAYS:
        return {"fired": False, "eligible": False, "restaurant_id": None, "suppression_reason": None}

    counts = defaultdict(int)
    for o in comfort_rainy:
        counts[o["restaurant_id"]] += 1
    top_comfort_restaurant = max(counts, key=counts.get)
    meta = restaurant_lookup.get(top_comfort_restaurant)
    if meta is None or not meta["is_active"]:
        return {"fired": True, "eligible": False, "restaurant_id": top_comfort_restaurant,
                "suppression_reason": "restaurant_inactive"}
    return {"fired": True, "eligible": True, "restaurant_id": top_comfort_restaurant, "suppression_reason": None}


def run_simulation(users, restaurants, orders, weather_calendar, rng, holdout_fraction, freq_cap):
    restaurant_lookup = restaurant_most_common_cuisine_lookup(restaurants)
    states = {row.user_id: UserSimState(row.user_id, row.city) for row in users.itertuples()}

    orders_by_date = defaultdict(list)
    for o in orders.itertuples():
        orders_by_date[o.order_date].append(o)

    all_dates = sorted(orders_by_date.keys())
    if not all_dates:
        raise ValueError("orders.csv has no rows")
    sim_start, sim_end = all_dates[0], all_dates[-1]

    events = []
    incremental_orders = []
    incr_order_counter = 0
    event_id_counter = 0

    def log_event(day, user_id, group, trigger_type, restaurant_id, event_type, note=""):
        nonlocal event_id_counter
        events.append({
            "event_id": f"E{event_id_counter:08d}",
            "event_date": day,
            "user_id": user_id,
            "experiment_group": group,
            "trigger_type": trigger_type,
            "restaurant_id": restaurant_id,
            "event_type": event_type,
            "note": note,
        })
        event_id_counter += 1

    day = sim_start
    while day <= sim_end:
        # 1. Ingest today's organic orders into each user's rolling history.
        for o in orders_by_date.get(day, []):
            st = states[o.user_id]
            st.history.append({
                "date": day, "restaurant_id": o.restaurant_id, "cuisine": o.cuisine,
                "is_rainy": bool(o.is_rainy), "source": "organic",
            })

        # 2. Evaluate every user for eligibility + all three triggers.
        for user_id, st in states.items():
            trailing_60 = st.orders_in_trailing(day, 60)
            is_eligible_user = len(trailing_60) >= MIN_ORDERS_TRAILING_60D

            if is_eligible_user and st.experiment_group is None:
                st.experiment_group = "holdout" if rng.random() < holdout_fraction else "treatment"
                st.eligible_since = day

            if not is_eligible_user or st.experiment_group is None:
                continue

            city_rainy_today = weather_calendar.get((day, st.city), False)
            trigger_results = {
                "day_of_week": evaluate_day_of_week_trigger(st, day, restaurant_lookup),
                "cadence": evaluate_cadence_trigger(st, day),
                "contextual": evaluate_contextual_trigger(st, day, city_rainy_today, restaurant_lookup),
            }

            for trigger_type, result in trigger_results.items():
                if not result["fired"]:
                    continue
                log_event(day, user_id, st.experiment_group, trigger_type,
                          result["restaurant_id"], "trigger_evaluated",
                          note="fired" if result["eligible"] else f"suppressed:{result['suppression_reason']}")

                if not result["eligible"]:
                    continue

                recent_nudges = st.nudges_in_trailing_week(day)
                if len(recent_nudges) >= freq_cap:
                    events[-1]["note"] = "suppressed:frequency_cap"
                    continue

                if st.experiment_group == "holdout":
                    # Evaluated and would have been sent, but holdout never gets nudged.
                    continue

                # --- Treatment path: send the nudge and simulate the response. ---
                st.nudges_sent_dates.append(day)
                if trigger_type == "day_of_week":
                    st.last_dow_fire[result["restaurant_id"]] = day
                elif trigger_type == "cadence":
                    st.last_cadence_fire = day
                elif trigger_type == "contextual":
                    st.last_contextual_fire = day

                log_event(day, user_id, st.experiment_group, trigger_type, result["restaurant_id"], "nudge_sent")

                opened = rng.random() < P_OPEN[trigger_type]
                if not opened:
                    log_event(day, user_id, st.experiment_group, trigger_type, result["restaurant_id"],
                              "nudge_dismissed", note="not_opened")
                    continue

                log_event(day, user_id, st.experiment_group, trigger_type, result["restaurant_id"], "nudge_opened")

                incremental = rng.random() < P_INCREMENTAL_ORDER_GIVEN_OPEN[trigger_type]
                if not incremental:
                    log_event(day, user_id, st.experiment_group, trigger_type, result["restaurant_id"],
                              "nudge_dismissed", note="opened_no_order")
                    continue

                log_event(day, user_id, st.experiment_group, trigger_type, result["restaurant_id"], "reorder_placed",
                          note="incremental")
                meta = restaurant_lookup[result["restaurant_id"]]
                incremental_orders.append({
                    "order_id": f"N{incr_order_counter:07d}",
                    "user_id": user_id,
                    "restaurant_id": result["restaurant_id"],
                    "cuisine": meta["cuisine"],
                    "order_datetime": datetime.combine(day, datetime.min.time()) + timedelta(hours=20),
                    "order_dow": day.weekday(),
                    "is_rainy": city_rainy_today,
                    "source": "nudge_incremental",
                    "trigger_type": trigger_type,
                })
                incr_order_counter += 1
                # This incremental order becomes part of the user's real history,
                # so tomorrow's cadence/day-of-week evaluation sees it too.
                st.history.append({
                    "date": day, "restaurant_id": result["restaurant_id"], "cuisine": meta["cuisine"],
                    "is_rainy": city_rainy_today, "source": "nudge_incremental",
                })

        day += timedelta(days=1)

    return pd.DataFrame(events), pd.DataFrame(incremental_orders), states


def write_experiment_readout(states, orders, incremental_orders_df, output_dir):
    rows = []
    for user_id, st in states.items():
        if st.experiment_group is None:
            continue
        organic_count = sum(1 for o in st.history if o["source"] == "organic")
        incremental_count = sum(1 for o in st.history if o["source"] == "nudge_incremental")
        rows.append({"user_id": user_id, "group": st.experiment_group,
                     "organic_orders": organic_count, "incremental_orders": incremental_count,
                     "total_orders": organic_count + incremental_count})
    df = pd.DataFrame(rows)

    summary = df.groupby("group")[["organic_orders", "incremental_orders", "total_orders"]].mean().round(3)

    lines = []
    lines.append(f"Eligible & assigned users: {len(df)} ({(df['group'] == 'treatment').sum()} treatment / "
                 f"{(df['group'] == 'holdout').sum()} holdout)")
    lines.append("")
    lines.append("Mean orders per user, by experiment group:")
    lines.append(summary.to_string())
    lines.append("")
    treatment_mean = summary.loc["treatment", "total_orders"] if "treatment" in summary.index else float("nan")
    holdout_mean = summary.loc["holdout", "total_orders"] if "holdout" in summary.index else float("nan")
    measured_lift = treatment_mean - holdout_mean
    lines.append(f"Measured incremental orders/user (treatment total − holdout total): {measured_lift:.3f}")
    lines.append("This should be in the neighborhood of the incremental orders the treatment group actually")
    lines.append("received (see 'incremental_orders' column above) — that's the number a real experiment")
    lines.append("readout would report as the causal effect. Compare it against P_INCREMENTAL_ORDER_GIVEN_OPEN")
    lines.append("x P_OPEN x (trigger fire rate) in the script to sanity-check the simulation end to end.")
    lines.append("")
    lines.append("Incremental orders by trigger type:")
    if not incremental_orders_df.empty:
        lines.append(incremental_orders_df["trigger_type"].value_counts().to_string())
    else:
        lines.append("(none fired — try a larger user count or longer sim window)")

    text = "\n".join(lines)
    with open(os.path.join(output_dir, "experiment_readout.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, required=True, help="output dir from generate_synthetic_swiggy_data.py")
    parser.add_argument("--output-dir", type=str, default="./trigger_engine_output")
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument("--frequency-cap-per-week", type=int, default=FREQUENCY_CAP_PER_WEEK)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading synthetic order history...")
    users, restaurants, orders = load_data(args.data_dir)

    sim_start = orders["order_date"].min()
    sim_end = orders["order_date"].max()
    print(f"Simulation window: {sim_start} to {sim_end} ({(sim_end - sim_start).days} days)")

    print("Building weather calendar...")
    weather_calendar = generate_weather_calendar(users["city"].unique().tolist(), sim_start, sim_end)

    print("Running trigger engine day by day (this is the slow step)...")
    events_df, incremental_orders_df, states = run_simulation(
        users, restaurants, orders, weather_calendar, rng,
        args.holdout_fraction, args.frequency_cap_per_week,
    )

    events_df.to_csv(os.path.join(args.output_dir, "events.csv"), index=False)

    orders_export = orders[["order_id", "user_id", "restaurant_id", "cuisine", "order_datetime",
                             "order_dow", "is_rainy"]].copy()
    orders_export["source"] = "organic"
    orders_export["trigger_type"] = None
    combined = pd.concat([orders_export, incremental_orders_df], ignore_index=True, sort=False)
    combined = combined.sort_values("order_datetime").reset_index(drop=True)
    combined.to_csv(os.path.join(args.output_dir, "orders_with_nudges.csv"), index=False)

    print(f"\nWrote events.csv ({len(events_df)} rows) and orders_with_nudges.csv "
          f"({len(combined)} rows, {len(incremental_orders_df)} incremental) to {args.output_dir}\n")

    write_experiment_readout(states, orders, incremental_orders_df, args.output_dir)


if __name__ == "__main__":
    main()
