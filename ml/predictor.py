"""
ml/predictor.py
Monthly spending forecaster.

Why "baseline + deviation"
-------------------------
Naively extrapolating a partial month (spend * days_in_month / days_elapsed)
over-counts lumpy costs: e.g. rent lands on day 1, so a mid-month projection
triples it, which then creates tons of false "this category is up 200%!"
alerts.

Instead, for each category:

    projected = baseline + (so_far - expected_by_now)

* `baseline`      = that category's own recent full-month average.
* `expected_by_now` = baseline * (days_elapsed / days_in_month) for
  "streamed" spend (dining, groceries…), which accrues evenly; but the *full*
  baseline for "fixed"/lumpy spend (rent, subscriptions) that is essentially
  settled for the month.

So when you're on pace, projected == baseline (stable, no noise). When you
genuinely overspend, projected rises by exactly the deviation. This gives a
calm, believable month-end forecast and per-category split with no heavy
external dependencies.
"""
from __future__ import annotations

import calendar
from datetime import date

import numpy as np
import pandas as pd

# Lump payments that effectively settle once per month, near its start.
FIXED_CATEGORIES = {"Housing", "Subscriptions"}


def _to_frame(expenses):
    if not expenses:
        return pd.DataFrame(columns=["date", "amount", "category"])
    df = pd.DataFrame(expenses)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["category"] = df.get("category", pd.Series(["Other"] * len(df)))
    return df


def _month_bounds(today):
    month_start = pd.Timestamp(today.replace(day=1))
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    days_elapsed = today.day
    frac = days_elapsed / days_in_month
    return month_start, days_elapsed, days_in_month, frac


def _monthly_totals(df, category=None):
    sub = df if category is None else df[df["category"] == category]
    if sub.empty:
        return pd.Series(dtype="float64")
    return sub.set_index("date")["amount"].resample("MS").sum()


def _hist_avg(series, today, months=3):
    if series is None or series.empty:
        return 0.0
    series = series[series.index < pd.Timestamp(today.replace(day=1))]
    completed = series[series > 0]
    if completed.empty:
        return 0.0
    return float(completed.tail(months).mean())


def _daily_increments(sub_df, month_start, days_elapsed):
    """Daily spend amounts (length == days_elapsed) within the current month."""
    day_amounts = np.zeros(days_elapsed)
    for _, row in sub_df.iterrows():
        idx = (row["date"] - month_start).days
        if 0 <= idx < days_elapsed:
            day_amounts[idx] += row["amount"]
    return day_amounts


def _project_category(so_far, hist_avg, fixed, frac, day_amounts,
                      days_in_month, days_elapsed):
    """Return (projected, lower, upper) for one category."""
    if hist_avg > 0:
        expected_by_now = hist_avg if fixed else hist_avg * frac
        projected = hist_avg + (so_far - expected_by_now)
    else:
        # No history: fall back to a pro-rata pace projection.
        projected = so_far / frac if frac > 0 else so_far

    # Never forecast below what's actually already been spent.
    projected = max(projected, so_far)

    # Uncertainty band.
    if fixed:
        err = 0.05 * projected
    elif days_elapsed >= 2 and day_amounts.any():
        std_daily = float(np.std(day_amounts))
        remaining = max(days_in_month - days_elapsed, 0)
        err = 1.28 * std_daily * np.sqrt(remaining) if remaining else 0.05 * projected
    else:
        err = 0.15 * projected

    lower = max(so_far, projected - err)
    upper = projected + err
    return round(projected, 2), round(lower, 2), round(upper, 2)


def forecast(expenses) -> dict:
    today = date.today()
    df = _to_frame(expenses)

    if df.empty:
        return {
            "has_data": False,
            "current_month": {"projected_total": 0.0, "so_far": 0.0},
            "by_category": [],
            "monthly_history": [],
            "next_month": 0.0,
        }

    month_start, days_elapsed, days_in_month, frac = _month_bounds(today)
    current = df[(df["date"].dt.year == today.year) &
                 (df["date"].dt.month == today.month)]

    cat_forecasts = []
    total_projected = 0.0
    total_so_far = 0.0
    for cat in sorted(df["category"].unique()):
        sub = current[current["category"] == cat]
        so_far = round(float(sub["amount"].sum()), 2)
        hist = _hist_avg(_monthly_totals(df, cat), today)
        fixed = cat in FIXED_CATEGORIES
        day_amounts = _daily_increments(sub, month_start, days_elapsed)
        projected, lower, upper = _project_category(
            so_far, hist, fixed, frac, day_amounts, days_in_month, days_elapsed)
        cat_forecasts.append({
            "category": cat,
            "so_far": so_far,
            "projected_total": projected,
            "lower": lower,
            "upper": upper,
            "fixed": fixed,
        })
        total_projected += projected
        total_so_far += so_far

    cat_forecasts.sort(key=lambda x: x["projected_total"], reverse=True)

    # Total-level band from the combined daily curve.
    total_day = _daily_increments(current, month_start, days_elapsed)
    remaining = max(days_in_month - days_elapsed, 0)
    if days_elapsed >= 2 and total_day.any():
        total_err = 1.28 * float(np.std(total_day)) * np.sqrt(remaining)
    else:
        total_err = 0.15 * total_projected

    result = {
        "projected_total": round(total_projected, 2),
        "so_far": round(total_so_far, 2),
        "upper": round(total_projected + total_err, 2),
        "lower": round(max(total_so_far, total_projected - total_err), 2),
        "days_elapsed": days_elapsed,
        "days_in_month": days_in_month,
        "days_remaining": remaining,
        "daily_burn": round(total_so_far / max(days_elapsed, 1), 2),
    }

    monthly = df.set_index("date")["amount"].resample("MS").sum()
    history = [
        {"month": idx.strftime("%b %Y"), "total": round(float(v), 2)}
        for idx, v in monthly.round(2).sort_index().items()
    ]

    completed = monthly[monthly.index < month_start]
    completed = completed[completed > 0]
    next_month = 0.0
    if len(completed) >= 2:
        x = np.arange(len(completed))
        slope, intercept = np.polyfit(x, completed.values, 1)
        next_month = float(max(slope, 0.0) * len(completed) + intercept)
    elif len(completed) == 1:
        next_month = float(completed.iloc[-1])

    return {
        "has_data": True,
        "as_of": today.isoformat(),
        "current_month": result,
        "by_category": cat_forecasts,
        "monthly_history": history,
        "next_month": round(next_month, 2),
    }