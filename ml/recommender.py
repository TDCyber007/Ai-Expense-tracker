"""
ml/recommender.py
Savings recommender.

Turns the user's spending patterns + the forecast into concrete, actionable
advice.  Each suggestion carries a realistic, quantified `potential_saving`
so the UI can show "acting on these tips saves ~$X this month".

Rules (pure logic, fully offline):
  * Rising category      -> trim back to a sane target.
  * Top discretionary    -> biggest lever to cut.
  * Subscriptions        -> look for unused recurring plans.
  * Budget overrun       -> alert against user-set per-category budgets.
  * On-track             -> positive reinforcement.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

# Categories where cutting back is usually realistic and low-pain.
DISCRETIONARY = {
    "Food & Dining", "Entertainment", "Shopping", "Subscriptions",
    "Travel", "Personal Care",
}

INCREASE_THRESHOLD = 0.15     # 15% vs historical average
MIN_AMOUNT = 12.0             # ignore tiny categories
DISCRETIONARY_CUT = 0.20      # suggested trim for top discretionary category


def _month_totals_by_category(df, today):
    """Return {category: Series indexed by month-start} of spend totals."""
    out = {}
    for cat, sub in df.groupby("category"):
        out[cat] = sub.set_index("date")["amount"].resample("MS").sum()
    return out


def _completed_avg(series, today, months=3):
    series = series[series.index < pd.Timestamp(today.replace(day=1))]
    completed = series[series > 0]
    if completed.empty:
        return 0.0
    return float(completed.tail(months).mean())


def _last_month_total(series, today):
    prev_start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    series = series[series.index == pd.Timestamp(prev_start)]
    return float(series.iloc[0]) if len(series) else 0.0


def recommend(expenses, forecast, budgets=None) -> dict:
    budgets = budgets or {}
    today = date.today()

    if not expenses:
        return {
            "tips": [],
            "total_potential_saving": 0.0,
            "headline": "Start logging daily expenses to unlock insights.",
        }

    df = pd.DataFrame(expenses)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    cat_series = _month_totals_by_category(df, today)
    proj_by_cat = {c["category"]: c for c in forecast.get("by_category", [])}

    tips = []
    savings_by_cat = {}

    # ---- Rule A: rising categories ---------------------------------------
    for cat, series in cat_series.items():
        projected = float(proj_by_cat.get(cat, {}).get("projected_total", 0.0))
        avg3 = _completed_avg(series, today)
        last = _last_month_total(series, today)
        if projected < MIN_AMOUNT or avg3 <= 0:
            continue
        if projected > avg3 * (1 + INCREASE_THRESHOLD):
            pct = int(round((projected / avg3 - 1) * 100))
            save = round(projected * 0.15, 2)  # trim back ~15%
            tips.append({
                "type": "rising",
                "severity": "warn",
                "category": cat,
                "title": f"{cat} is up {pct}% vs your 3-month average",
                "detail": (
                    f"You're on pace for ${projected:,.2f} this month versus "
                    f"a ${avg3:,.2f} average. Cutting back ~15% keeps the "
                    f"habit without overspending."
                ),
                "potential_saving": save,
            })
            savings_by_cat[cat] = max(savings_by_cat.get(cat, 0), save)

    # ---- Rule B: top discretionary category ------------------------------
    disc_proj = [(c, proj_by_cat[c]["projected_total"])
                 for c in proj_by_cat if c in DISCRETIONARY]
    disc_proj = [(c, v) for c, v in disc_proj if v >= MIN_AMOUNT]
    if disc_proj:
        top_cat, top_val = max(disc_proj, key=lambda x: x[1])
        save = round(top_val * DISCRETIONARY_CUT, 2)
        tips.append({
            "type": "discretionary",
            "severity": "info",
            "category": top_cat,
            "title": f"{top_cat} is your biggest discretionary lever",
            "detail": (
                f"At ${top_val:,.2f} projected, trimming 20% of "
                f"{top_cat} (fewer impulse buys / one less outing a week) is "
                f"the fastest way to bank a saving this month."
            ),
            "potential_saving": save,
        })
        savings_by_cat[top_cat] = max(savings_by_cat.get(top_cat, 0), save)

    # ---- Rule C: subscriptions -------------------------------------------
    sub_proj = float(proj_by_cat.get("Subscriptions", {}).get(
        "projected_total", 0.0))
    if sub_proj >= MIN_AMOUNT:
        est_count = max(1, int(round(sub_proj / 10.0)))
        tips.append({
            "type": "subscriptions",
            "severity": "info",
            "category": "Subscriptions",
            "title": f"~${sub_proj:,.2f}/mo going to subscriptions",
            "detail": (
                f"That's roughly {est_count} recurring plan(s). Audit for "
                f"anything unused — canceling just one commonly frees "
                f"$10-15 a month."
            ),
            "potential_saving": 12.0,
        })
        savings_by_cat["Subscriptions"] = max(
            savings_by_cat.get("Subscriptions", 0), 12.0)

    # ---- Rule D: budget overruns -----------------------------------------
    for cat, limit in budgets.items():
        projected = float(proj_by_cat.get(cat, {}).get("projected_total", 0.0))
        if limit > 0 and projected > limit:
            over = round(projected - limit, 2)
            tips.append({
                "type": "budget",
                "severity": "warn",
                "category": cat,
                "title": f"{cat} budget exceeded",
                "detail": (
                    f"Projected ${projected:,.2f} vs your ${limit:,.2f} budget "
                    f"— ${over:,.2f} over. Pause non-essential spend here."
                ),
                "potential_saving": over,
            })
            savings_by_cat[cat] = max(savings_by_cat.get(cat, 0), over)

    # ---- Rule E: overall pace vs history ---------------------------------
    current = forecast.get("current_month", {})
    projected_total = float(current.get("projected_total", 0.0))
    hist_avg = _completed_avg(
        df.set_index("date")["amount"].resample("MS").sum(), today)
    if hist_avg > 0 and projected_total > hist_avg * 1.10:
        diff = projected_total - hist_avg
        if not any(t["type"] == "rising" for t in tips):
            tips.append({
                "type": "pace",
                "severity": "warn",
                "category": "Overall",
                "title": (
                    f"Pacing ${diff:,.2f} above your typical month"),
                "detail": (
                    f"You're tracking ${projected_total:,.2f} vs a "
                    f"${hist_avg:,.2f} average. Set a daily cap of "
                    f"${projected_total / current.get('days_in_month', 30):,.2f}"
                    f" to stay on target."
                ),
                "potential_saving": round(diff, 2),
            })

    if not tips:
        tips.append({
            "type": "on_track",
            "severity": "good",
            "category": "Overall",
            "title": "You're on track this month",
            "detail": (
                "Spending looks steady versus your recent history. Keep "
                "logging to keep the predictions sharp."
            ),
            "potential_saving": 0.0,
        })

    # Order: warnings first, then biggest saving.
    severity_rank = {"warn": 0, "info": 1, "good": 2}
    tips.sort(key=lambda t: (severity_rank.get(t["severity"], 3),
                             -t["potential_saving"]))

    return {
        "tips": tips,
        "total_potential_saving": round(sum(savings_by_cat.values()), 2),
        "projected_total": round(projected_total, 2),
        "hist_avg": round(hist_avg, 2),
    }