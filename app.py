"""
app.py
AI-Based Expense Tracker — Flask web app + REST API.

Run locally:
    python app.py            # then open http://127.0.0.1:5000

The first run trains the categorization model on the built-in seed corpus,
so the app is fully functional immediately (no external data required).
"""
import os
from datetime import date, datetime

import pandas as pd
from flask import Flask, jsonify, render_template, request

import core.db as db
from core.demo import generate_demo_expenses
from ml.categorizer import TransactionCategorizer, CATEGORIES, get_categorizer
from ml.predictor import forecast
from ml.recommender import recommend

app = Flask(__name__)

# Categorizer is trained once and shared across requests.
categorizer = get_categorizer()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template(
        "index.html",
        categories=CATEGORIES,
        budgets=db.get_budgets(),
    )


# ---------------------------------------------------------------------------
# Expenses
# ---------------------------------------------------------------------------
@app.post("/api/expenses")
def add_expense():
    data = request.get_json(force=True) or {}
    description = (data.get("description") or "").strip()
    amount = data.get("amount")
    when = data.get("date") or date.today().isoformat()

    if not description:
        return jsonify({"error": "description is required"}), 400
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return jsonify({"error": "amount must be a number"}), 400
    if amount <= 0:
        return jsonify({"error": "amount must be positive"}), 400
    # validate date
    try:
        datetime.strptime(when, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "date must be YYYY-MM-DD"}), 400

    # The AI part: predict the category + confidence.
    override = data.get("category")
    if override and override in CATEGORIES:
        category, confidence = override, 1.0
    else:
        pred = categorizer.predict(description)
        category, confidence = pred["category"], pred["confidence"]

    expense_id = db.add_expense(description, amount, category, when,
                                source="manual", confidence=confidence)
    return jsonify({
        "id": expense_id,
        "description": description,
        "amount": amount,
        "category": category,
        "date": when,
        "confidence": confidence,
    }), 201


@app.delete("/api/expenses/<int:expense_id>")
def remove_expense(expense_id):
    rows = db.delete_expense(expense_id)
    if not rows:
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": expense_id})


@app.get("/api/expenses")
def get_expenses():
    limit = request.args.get("limit", type=int)
    rows = db.list_expenses(limit=limit)
    return jsonify({"expenses": rows, "count": len(rows)})


@app.post("/api/expenses/<int:expense_id>/recategorize")
def recategorize(expense_id):
    """Explicit user feedback -> fix category and re-train the model on the
    user's own labelled history."""
    data = request.get_json(force=True) or {}
    category = data.get("category")
    if category not in CATEGORIES:
        return jsonify({"error": "unknown category"}), 400

    expenses = db.list_expenses()
    target = next((e for e in expenses if e["id"] == expense_id), None)
    if not target:
        return jsonify({"error": "not found"}), 404

    # Update the row's category directly.
    with db.get_connection() as conn:
        conn.execute("UPDATE expenses SET category=?, confidence=1.0, "
                     "source='corrected' WHERE id=?",
                     (category, expense_id))

    _retrain()
    return jsonify({"id": expense_id, "category": category})


# ---------------------------------------------------------------------------
# Analytics / AI insights
# ---------------------------------------------------------------------------
@app.get("/api/summary")
def summary():
    today = date.today()
    expenses = db.list_expenses()

    if not expenses:
        return jsonify({
            "month_total": 0.0, "projected_total": 0.0, "count": 0,
            "top_category": None, "avg_daily": 0.0, "total_all": 0.0,
            "month_change_pct": None, "by_category": [],
        })

    df = pd.DataFrame(expenses)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"])

    month_start = pd.Timestamp(today.replace(day=1))
    prev_month_end = month_start - pd.Timedelta(days=1)
    prev_month_start = pd.Timestamp(prev_month_end.replace(day=1))

    this_month = df[(df["date"] >= month_start)]
    prev_month = df[(df["date"] >= prev_month_start) &
                    (df["date"] < month_start)]

    month_total = round(float(this_month["amount"].sum()), 2)
    days_elapsed = today.day
    avg_daily = round(month_total / days_elapsed, 2) if days_elapsed else 0.0

    cat_breakdown = (
        this_month.groupby("category")["amount"].sum()
        .sort_values(ascending=False)
    )
    by_category = [
        {"category": k, "amount": round(float(v), 2)}
        for k, v in cat_breakdown.items()
    ]
    top_category = by_category[0]["category"] if by_category else None

    prev_total = float(prev_month["amount"].sum())
    month_change = None
    if prev_total > 0:
        month_change = round((month_total - prev_total) / prev_total * 100, 1)

    fc = forecast(expenses)["current_month"]

    return jsonify({
        "month_total": month_total,
        "projected_total": fc.get("projected_total", 0.0),
        "days_elapsed": fc.get("days_elapsed", days_elapsed),
        "days_in_month": fc.get("days_in_month", today.day),
        "count": int(len(this_month)),
        "total_count": int(len(df)),
        "top_category": top_category,
        "avg_daily": avg_daily,
        "total_all": round(float(df["amount"].sum()), 2),
        "month_change_pct": month_change,
        "prev_month_total": round(prev_total, 2),
        "by_category": by_category,
    })


@app.get("/api/insights")
def insights():
    expenses = db.list_expenses()
    fc = forecast(expenses)
    rec = recommend(expenses, fc, budgets=db.get_budgets())
    return jsonify({"forecast": fc, "recommendations": rec})


# ---------------------------------------------------------------------------
# Budgets & maintenance
# ---------------------------------------------------------------------------
@app.post("/api/budgets")
def set_budget():
    data = request.get_json(force=True) or {}
    category = data.get("category")
    amount = data.get("amount", 0)
    if category not in CATEGORIES:
        return jsonify({"error": "unknown category"}), 400
    db.set_budget(category, max(0.0, float(amount)))
    return jsonify({"budgets": db.get_budgets()})


@app.get("/api/budgets")
def get_budgets():
    return jsonify({"budgets": db.get_budgets()})


@app.post("/api/reset")
def reset():
    db.clear_all()
    # Reset the model back to the seed-trained baseline.
    global categorizer
    categorizer = TransactionCategorizer().train(persist=True)
    return jsonify({"ok": True, "expenses": 0})


@app.post("/api/seed")
def seed():
    """Populate ~6 months of realistic demo data and re-train on it."""
    global categorizer
    demo = generate_demo_expenses(months=6)
    for e in demo:
        db.add_expense(e["description"], e["amount"], e["category"],
                       e["date"], source="demo", confidence=1.0)
    _retrain()
    return jsonify({"ok": True, "added": len(demo),
                    "expenses": db.count_expenses()})


@app.post("/api/retrain")
def retrain_endpoint():
    _retrain()
    return jsonify({"ok": True, "trained_on": categorizer.trained_on})


@app.get("/api/health")
def health():
    return jsonify({"status": "ok",
                    "model_trained_on": categorizer.trained_on})


@app.get("/api/predict")
def predict_category():
    """Live category preview while the user types a description."""
    desc = (request.args.get("description") or "").strip()
    if not desc:
        return jsonify({"category": None})
    pred = categorizer.predict(desc)
    return jsonify({"category": pred["category"],
                    "confidence": pred["confidence"]})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _retrain():
    """Re-train the categorizer on the seed corpus + user's labelled rows."""
    global categorizer
    labelled = [
        (e["description"], e["category"])
        for e in db.list_expenses()
    ]
    categorizer = TransactionCategorizer().train(extra_data=labelled,
                                                 persist=True)


if __name__ == "__main__":
    db.init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)