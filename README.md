# AI-Based Expense Tracker

A ready-to-run personal finance dashboard that **auto-categorizes** your
transactions, **forecasts your monthly spending**, and gives you **actionable
savings recommendations** — all powered by on-device machine learning, with a
polished dark UI. No cloud services or API keys required.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Flask](https://img.shields.io/badge/flask-3.0-black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![license](https://img.shields.io/badge/license-MIT-green)

## Features

- **Add expenses in seconds** — type a description like *"Morning coffee at
  Starbucks"* and the model picks the category live as you type.
- **AI categorization** — a TF-IDF + Logistic Regression classifier with a
  high-precision keyword layer. Correct a category once and the model
  *re-trains on your own history*.
- **Monthly spending forecast** — a "baseline + deviation" time-series model
  that projects month-end spend and a confidence band, per category, without
  double-counting lumpy costs like rent.
- **Savings recommendations** — detects rising categories, your biggest
  discretionary lever, recurring subscriptions and budget overruns, each with
  a quantified potential saving.
- **Beautiful dashboard** — KPI cards, monthly bar chart, category donut,
  forecast bars and tip cards. Fully responsive.
- **One-click demo data** — loads 6 months of realistic transactions so every
  chart and insight is populated instantly.

## Quick start

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. run
python app.py
```

Open **http://127.0.0.1:5000** and click **Load demo data** to explore with a
full 6-month history, or start adding your own expenses right away.

> The first run trains the categorization model on a built-in seed corpus, so
> the app is fully functional immediately — no setup or external data needed.

### Load demo data from the CLI instead

```bash
python seed.py
```

## Tech stack

| Layer     | Choice                                        |
|-----------|-----------------------------------------------|
| Backend   | Python, Flask                                 |
| Storage   | SQLite (`core/db.py`)                         |
| ML        | scikit-learn, pandas, numpy (categorize/forecast) |
| Frontend  | Vanilla HTML/CSS/JS + Chart.js                |

## How the AI works

**Categorizer** (`ml/categorizer.py`)
A `TfidfVectorizer` (1–2 grams) + `LogisticRegression` classifier trained on a
rich seed corpus per category, blended with a keyword-evidence scorer. It
returns a calibrated confidence and is re-trained (persisted with `joblib`)
whenever you change a category.

**Predictor** (`ml/predictor.py`)
For every category it computes
`projected = baseline + (so_far - expected_by_now)`, where `baseline` is the
category's recent full-month average and `expected_by_now` is pro-rated by day
of month for streamed spend or taken in full for fixed costs (rent,
subscriptions). This avoids the classic mid-month over-projection of lumpy
bills and yields a stable month-end estimate plus an 80% confidence band.

**Recommender** (`ml/recommender.py`)
Rule-based insights over the forecast: increases vs. a 3-month average, the
top discretionary category, recurring subscription load, and per-category
budget overruns — each with a realistic saving estimate.

## Project structure

```
app.py                 Flask app + REST API
seed.py                CLI to load 6 months of demo data
requirements.txt
core/
  db.py                SQLite helpers
  demo.py              reproducible demo-data generator
ml/
  categorizer.py       transaction categorization (ML + rules)
  predictor.py         monthly spending forecast
  recommender.py       savings recommendations
static/                CSS + JS dashboard
templates/index.html   single-page app
data/                  (generated) expenses.db + trained model
```

## REST API

| Method | Endpoint                        | Purpose                          |
|--------|---------------------------------|----------------------------------|
| GET    | `/api/summary`                  | KPIs + this-month breakdown      |
| GET    | `/api/insights`                 | forecast + recommendations        |
| GET    | `/api/expenses`                 | list transactions                |
| POST   | `/api/expenses`                 | add (auto-categorized)           |
| DELETE | `/api/expenses/<id>`            | delete                           |
| POST   | `/api/expenses/<id>/recategorize`| fix category + re-train         |
| GET    | `/api/predict?description=`     | live category preview            |
| GET/POST| `/api/budgets`                 | per-category budgets             |
| POST   | `/api/seed`                     | load demo data                   |
| POST   | `/api/reset`                    | clear everything                 |
| GET    | `/api/health`                   | status / model info              |

## Roadmap / ideas for contributors

- Currency/locale support
- CSV import (bank statements)
- User accounts + multi-device
- Swap SQLite for Postgres behind a thin DAO
- More advanced forecasting (Prophet / ARIMA)
