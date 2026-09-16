"""
Synthetic order-history generator for the Reorder Nudge Engine project (Swiggy PRD).

Why synthetic: we don't have access to real Swiggy logs. Instead of inventing numbers,
this generator is CALIBRATED to the general, publicly-documented shape of two real
datasets:

1. Instacart Market Basket Analysis (Kaggle) — the closest public analogue for
   recurring-purchase cadence. Its published summary stats show:
     - days_since_prior_order is capped at 30 and has strong weekly periodicity:
       visible spikes at 7, 14, 21, 28/30 days, not a smooth decay.
     - a large majority of users have at least one reorder; reorder behavior is
       concentrated on a small set of "regular" items/vendors per user, not spread evenly.
   We use this to shape inter-order GAP DAYS and per-user REPEAT CONCENTRATION below.
   Dataset: https://www.kaggle.com/datasets/lakshaymalhotra/instacart-market-basket-analysis

2. Zomato Delivery Operations Analytics / Swiggy-Zomato Order Information (Kaggle) —
   used only for plausible food-delivery texture: cuisine mix, order value ranges,
   prep times. Dataset:
   https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset
   https://www.kaggle.com/datasets/cbhavik/swiggyzomato-order-information

IMPORTANT — these are DEFAULT, approximate calibrations based on the widely-reported
shape of those datasets, not a re-fit against the raw files (we don't have Kaggle
access in this environment). If you download the real Instacart `orders.csv`, pass
its path via --instacart-orders-csv and this script will refit the gap-days
distribution to the ACTUAL empirical histogram instead of the hardcoded default —
this is the "calibrate against real data" step called out in the PRD's data strategy.

Outputs (to --output-dir, default ./synthetic_data):
  users.csv        one row per synthetic user
  restaurants.csv  one row per synthetic restaurant
  orders.csv       one row per order, with the fields the trigger engine will need
  validation_summary.txt  sanity-check stats so you can eyeball the calibration

Usage:
  python generate_synthetic_swiggy_data.py --n-users 5000 --n-restaurants 300 --sim-days 180
  python generate_synthetic_swiggy_data.py --instacart-orders-csv path/to/orders.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Calibration constants (defaults — overridden by --instacart-orders-csv if given)
# ---------------------------------------------------------------------------

CITIES = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai"]

CUISINES = [
    "North Indian", "South Indian", "Biryani", "Chinese", "Fast Food",
    "Momos", "Desserts", "Pizza", "Rolls & Wraps", "Beverages",
]

# Cuisines treated as "comfort food" for the contextual (rainy-evening) trigger.
COMFORT_CUISINES = {"Biryani", "Chinese", "Momos", "Fast Food"}

# Monsoon months get an elevated chance of a "rainy" flag on any given day.
MONSOON_MONTHS = {6, 7, 8, 9}
RAINY_PROB_MONSOON = 0.40
RAINY_PROB_OTHER = 0.08

# User frequency segments: share of the population and mean inter-order gap (days).
# Shaped to mirror Instacart-style skew: a small power-user tail, a large
# occasional/dormant base — NOT a uniform or normal distribution.
SEGMENTS = {
    "power":      {"share": 0.12, "mean_gap": 3.5, "regular_set_size": (3, 5)},
    "regular":    {"share": 0.33, "mean_gap": 7.0,  "regular_set_size": (2, 4)},
    "occasional": {"share": 0.35, "mean_gap": 16.0, "regular_set_size": (1, 2)},
    "dormant":    {"share": 0.20, "mean_gap": 35.0, "regular_set_size": (1, 1)},
}

# Probability an order goes to one of the user's "regular" restaurants rather than
# a fresh one. Instacart's item-level reorder rate is ~59%; restaurant-level repeat
# in food delivery tends to run higher than item-level repeat in grocery, so we set
# this a bit above that as a reasoned assumption — tune this constant if you have
# a better number.
REGULAR_RESTAURANT_ORDER_PROB = 0.68

# Probability that, for a REGULAR-restaurant order, we snap the date to the user's
# personal "habit weekday" (e.g. always orders biryani on Friday). This is what
# makes trigger #1 (day-of-week pattern) in the PRD detectable in the data at all —
# without an intentionally seeded pattern, a trigger engine has nothing real to find.
HABIT_WEEKDAY_SNAP_PROB = 0.55

DISH_NAMES_BY_CUISINE = {
    "North Indian": ["Butter Chicken", "Paneer Tikka", "Dal Makhani", "Naan Combo"],
    "South Indian": ["Masala Dosa", "Idli Sambar", "Curd Rice", "Uttapam"],
    "Biryani": ["Chicken Biryani", "Mutton Biryani", "Veg Biryani", "Egg Biryani"],
    "Chinese": ["Hakka Noodles", "Manchurian", "Fried Rice", "Chilli Chicken"],
    "Fast Food": ["Burger Combo", "Loaded Fries", "Chicken Wrap", "Cheese Sandwich"],
    "Momos": ["Steamed Momos", "Fried Momos", "Tandoori Momos"],
    "Desserts": ["Brownie", "Gulab Jamun", "Ice Cream Tub", "Cheesecake Slice"],
    "Pizza": ["Margherita", "Farmhouse Pizza", "Pepperoni Pizza"],
    "Rolls & Wraps": ["Chicken Roll", "Paneer Roll", "Egg Roll"],
    "Beverages": ["Cold Coffee", "Fresh Juice", "Milkshake"],
}


def load_instacart_gap_distribution(csv_path):
    """
    Optional real-data calibration. Expects Instacart's orders.csv (columns include
    'days_since_prior_order'). Returns an empirical (values, probabilities) pair to
    sample from, replacing the hardcoded gamma-mixture default.
    """
    df = pd.read_csv(csv_path, usecols=["days_since_prior_order"])
    gaps = df["days_since_prior_order"].dropna()
    gaps = gaps[gaps > 0]
    counts = gaps.value_counts().sort_index()
    values = counts.index.to_numpy()
    probs = (counts / counts.sum()).to_numpy()
    print(f"[calibration] loaded empirical gap-day distribution from {csv_path} "
          f"({len(gaps)} rows, {len(values)} distinct gap values)")
    return values, probs


def sample_gap_days(rng, mean_gap, empirical_gap_dist=None):
    """
    Draw an inter-order gap in days. If an empirical distribution (from real
    Instacart data) was loaded, sample from it directly, scaled by the user's
    segment speed. Otherwise use a weekly-periodicity mixture: most gaps cluster
    on multiples of 7 (habitual weekly reordering), with continuous variation
    elsewhere.
    """
    if empirical_gap_dist is not None:
        values, probs = empirical_gap_dist
        base = rng.choice(values, p=probs)
        # Rescale the Instacart-shaped draw (which centers around grocery cadence)
        # to this user's segment speed while keeping the same relative "peakiness".
        scale = mean_gap / 12.0  # ~12 days is Instacart's rough central tendency
        gap = max(1, int(round(base * scale)))
        return gap

    if rng.random() < 0.55:
        weekly_multiples = np.array([7, 14, 21, 28, 30])
        weights = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        base = rng.choice(weekly_multiples, p=weights)
        jitter = rng.integers(-1, 2)
        gap = max(1, int(base * (mean_gap / 7.0) + jitter))
    else:
        gap = max(1, int(rng.gamma(shape=2.0, scale=mean_gap / 2.0)))
    return gap


def generate_restaurants(n, rng):
    rows = []
    for i in range(n):
        cuisine = rng.choice(CUISINES)
        rows.append({
            "restaurant_id": f"R{i:05d}",
            "name": f"{cuisine.split()[0]} Kitchen {i}",
            "city": rng.choice(CITIES),
            "cuisine": cuisine,
            "avg_prep_time_min": int(rng.integers(15, 45)),
            "price_tier": rng.choice(["budget", "mid", "premium"], p=[0.5, 0.4, 0.1]),
            "rating": round(float(rng.uniform(3.2, 4.8)), 1),
            # A small share of restaurants are "closed" at generation time, so the
            # trigger engine's fire-time eligibility check (PRD requirement #3) has
            # something real to filter out.
            "is_active": bool(rng.random() > 0.06),
        })
    return pd.DataFrame(rows)


def generate_users(n, rng):
    segment_names = list(SEGMENTS.keys())
    segment_shares = [SEGMENTS[s]["share"] for s in segment_names]
    rows = []
    for i in range(n):
        segment = rng.choice(segment_names, p=segment_shares)
        rows.append({
            "user_id": f"U{i:06d}",
            "city": rng.choice(CITIES),
            "signup_date": (datetime(2024, 1, 1) + timedelta(days=int(rng.integers(0, 400)))).date(),
            "frequency_segment": segment,
            "cuisine_preference": rng.choice(CUISINES, size=2, replace=False).tolist(),
            # Habit weekday: 0=Mon .. 6=Sun. Drives the day-of-week trigger.
            # Weighted toward Thu-Sun so the population-level distribution skews
            # toward weekends (a well-known food-delivery pattern), while each user
            # still has one specific idiosyncratic day the trigger can latch onto.
            "habit_weekday": int(rng.choice(
                [0, 1, 2, 3, 4, 5, 6],
                p=[0.08, 0.08, 0.09, 0.10, 0.20, 0.23, 0.22],
            )),
            "orders_comfort_food_when_rainy": bool(rng.random() < 0.45),
        })
    return pd.DataFrame(rows)


def snap_to_weekday(d, target_weekday):
    """Shift date d by at most 3 days to land on target_weekday."""
    delta = (target_weekday - d.weekday()) % 7
    if delta > 3:
        delta -= 7
    return d + timedelta(days=int(delta))


def generate_orders(users_df, restaurants_df, sim_days, rng, empirical_gap_dist=None):
    end_date = datetime(2026, 9, 16)
    start_date = end_date - timedelta(days=sim_days)

    restaurants_by_city = {
        city: restaurants_df[restaurants_df["city"] == city]
        for city in CITIES
    }

    order_rows = []
    order_id = 0

    for user in users_df.itertuples():
        segment = SEGMENTS[user.frequency_segment]
        mean_gap = segment["mean_gap"]
        city_restaurants = restaurants_by_city[user.city]
        if city_restaurants.empty:
            continue

        # Build this user's "regular set" — a small handful of go-to restaurants,
        # preferring their favorite cuisines. This is what creates real repeat
        # concentration for the reorder-nudge trigger to detect.
        pref_cuisine_pool = city_restaurants[city_restaurants["cuisine"].isin(user.cuisine_preference)]
        pool = pref_cuisine_pool if len(pref_cuisine_pool) >= 2 else city_restaurants
        lo, hi = segment["regular_set_size"]
        set_size = min(len(pool), int(rng.integers(lo, hi + 1)))
        regular_set = pool.sample(n=set_size, random_state=int(rng.integers(0, 1_000_000)))

        current_date = start_date + timedelta(days=int(rng.integers(0, 14)))
        while current_date < end_date:
            gap = sample_gap_days(rng, mean_gap, empirical_gap_dist)
            current_date = current_date + timedelta(days=gap)
            if current_date >= end_date:
                break

            use_regular = rng.random() < REGULAR_RESTAURANT_ORDER_PROB
            if use_regular and not regular_set.empty:
                restaurant = regular_set.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
                if rng.random() < HABIT_WEEKDAY_SNAP_PROB:
                    current_date = snap_to_weekday(current_date, user.habit_weekday)
            else:
                restaurant = city_restaurants.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]

            is_monsoon = current_date.month in MONSOON_MONTHS
            rainy = rng.random() < (RAINY_PROB_MONSOON if is_monsoon else RAINY_PROB_OTHER)

            # Contextual trigger seed: on rainy evenings, comfort-food-prone users
            # skew toward comfort cuisines even outside their normal regular set.
            cuisine = restaurant["cuisine"]
            if rainy and user.orders_comfort_food_when_rainy and rng.random() < 0.5:
                comfort_pool = city_restaurants[city_restaurants["cuisine"].isin(COMFORT_CUISINES)]
                if not comfort_pool.empty:
                    restaurant = comfort_pool.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
                    cuisine = restaurant["cuisine"]

            dish_pool = DISH_NAMES_BY_CUISINE.get(cuisine, ["Chef's Special"])
            item_count = int(rng.integers(1, 4))
            items = rng.choice(dish_pool, size=min(item_count, len(dish_pool)), replace=False).tolist()

            price_tier_multiplier = {"budget": 1.0, "mid": 1.6, "premium": 2.4}[restaurant["price_tier"]]
            cart_value = round(float(rng.uniform(120, 220) * price_tier_multiplier * len(items) / 2), 2)

            hour = int(np.clip(rng.normal(20, 2.5), 8, 23))  # dinner-skewed

            order_rows.append({
                "order_id": f"O{order_id:07d}",
                "user_id": user.user_id,
                "restaurant_id": restaurant["restaurant_id"],
                "cuisine": cuisine,
                "order_datetime": datetime.combine(current_date.date(), datetime.min.time()) + timedelta(hours=hour),
                "order_dow": current_date.weekday(),
                "order_hour": hour,
                "is_weekend": current_date.weekday() >= 4,
                "is_rainy": rainy,
                "is_regular_restaurant_for_user": bool(use_regular),
                "gap_days_since_prev_order": gap,
                "item_count": len(items),
                "items": "|".join(items),
                "cart_value_inr": cart_value,
                "restaurant_was_active": bool(restaurant["is_active"]),
            })
            order_id += 1

    return pd.DataFrame(order_rows)


def write_validation_summary(users_df, restaurants_df, orders_df, output_dir):
    lines = []
    lines.append(f"Users: {len(users_df)} | Restaurants: {len(restaurants_df)} | Orders: {len(orders_df)}")
    lines.append("")
    lines.append("Orders per user by frequency segment (mean):")
    merged = orders_df.merge(users_df[["user_id", "frequency_segment"]], on="user_id")
    lines.append(merged.groupby("frequency_segment").size().div(
        users_df.groupby("frequency_segment").size()
    ).round(1).to_string())
    lines.append("")
    lines.append("Share of orders going to a 'regular' restaurant (repeat concentration):")
    lines.append(f"  {orders_df['is_regular_restaurant_for_user'].mean():.2%}")
    lines.append("")
    lines.append("Gap-days histogram (bucketed) — should show weekly spikes at 7/14/21/28-30:")
    bucket_counts = pd.cut(
        orders_df["gap_days_since_prev_order"],
        bins=[0, 3, 6, 8, 13, 15, 20, 22, 27, 31, 1000],
        labels=["1-3", "4-6", "7", "8-13", "14", "15-20", "21", "22-27", "28-30", "30+"],
    ).value_counts().sort_index()
    lines.append(bucket_counts.to_string())
    lines.append("")
    lines.append("Day-of-week order distribution (0=Mon..6=Sun) — should skew toward Thu-Sun:")
    lines.append(orders_df["order_dow"].value_counts().sort_index().to_string())
    lines.append("")
    lines.append("Share of orders flagged rainy:")
    lines.append(f"  {orders_df['is_rainy'].mean():.2%}")
    lines.append("")
    lines.append("Restaurants flagged inactive (for eligibility-check testing):")
    lines.append(f"  {(~restaurants_df['is_active']).sum()} of {len(restaurants_df)} "
                 f"({(~restaurants_df['is_active']).mean():.1%})")

    text = "\n".join(lines)
    with open(os.path.join(output_dir, "validation_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-users", type=int, default=5000)
    parser.add_argument("--n-restaurants", type=int, default=300)
    parser.add_argument("--sim-days", type=int, default=180, help="how many days of history to simulate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./synthetic_data")
    parser.add_argument("--instacart-orders-csv", type=str, default=None,
                         help="path to a real Instacart orders.csv to calibrate gap-days from")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    empirical_gap_dist = None
    if args.instacart_orders_csv:
        empirical_gap_dist = load_instacart_gap_distribution(args.instacart_orders_csv)

    print("Generating restaurants...")
    restaurants_df = generate_restaurants(args.n_restaurants, rng)

    print("Generating users...")
    users_df = generate_users(args.n_users, rng)

    print("Generating order history (this is the slow step)...")
    orders_df = generate_orders(users_df, restaurants_df, args.sim_days, rng, empirical_gap_dist)

    users_df.to_csv(os.path.join(args.output_dir, "users.csv"), index=False)
    restaurants_df.to_csv(os.path.join(args.output_dir, "restaurants.csv"), index=False)
    orders_df.to_csv(os.path.join(args.output_dir, "orders.csv"), index=False)

    print(f"\nWrote users.csv, restaurants.csv, orders.csv to {args.output_dir}\n")
    write_validation_summary(users_df, restaurants_df, orders_df, args.output_dir)


if __name__ == "__main__":
    main()
