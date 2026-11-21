"""
core/demo.py
Reproducible demo-data generator.

Produces ~6 months of realistic, varied transactions that map cleanly onto
the categorizer's categories, so the forecast / recommendation engines have
meaningful history the moment someone clicks "Load demo data".
"""
from __future__ import annotations

import random
from datetime import date, timedelta
import calendar

# per category: description templates [verb-ish], amount range, tx/month
CATEGORY_TEMPLATES = {
    "Food & Dining": {
        "desc": ["Coffee at Starbucks", "Lunch at Chipotle", "Dinner out",
                 "Pizza delivery Dominos", "Burger at McDonalds",
                 "Doordash food delivery", "Sushi takeout", "Cafe latte",
                 "Brunch on the weekend", "Taco night", "Ramen nearby"],
        "amt": (6.0, 42.0), "per_month": (10, 16),
    },
    "Groceries": {
        "desc": ["Whole Foods groceries", "Weekly grocery shopping",
                 "Costco bulk run", "Milk eggs and bread",
                 "Trader Joe snacks", "Farmers market produce",
                 "Kroger weekly shop", "Target grocery pickup"],
        "amt": (18.0, 96.0), "per_month": (5, 9),
    },
    "Transportation": {
        "desc": ["Uber ride downtown", "Gas station fill up", "Lyft ride",
                 "Metro card top up", "Parking garage", "Shell fuel",
                 "Oil change", "Bridge toll"],
        "amt": (8.0, 65.0), "per_month": (5, 10),
    },
    "Housing": {
        "desc": ["Monthly rent payment"], "amt": (1245.0, 1255.0),
        "per_month": (1, 1), "fixed_day": 1,
    },
    "Utilities": {
        "desc": ["Electricity bill", "Internet bill Comcast",
                 "Phone bill Verizon", "Water bill", "Gas bill",
                 "AT&T wireless"],
        "amt": (45.0, 130.0), "per_month": (3, 5),
    },
    "Entertainment": {
        "desc": ["Netflix subscription", "Spotify Premium", "Movie night",
                 "Steam game", "Concert tickets", "Disney Plus",
                 "Bowling with friends", "Xbox game pass"],
        "amt": (9.0, 70.0), "per_month": (3, 7),
    },
    "Health & Fitness": {
        "desc": ["Gym membership", "Pharmacy prescription", "Doctor copay",
                 "Vitamins", "Yoga class", "Dentist cleaning"],
        "amt": (15.0, 90.0), "per_month": (2, 5),
    },
    "Shopping": {
        "desc": ["Amazon order", "New shoes Nike", "Headphones BestBuy",
                 "IKEA furniture", "Zara clothes", "Phone accessories"],
        "amt": (14.0, 180.0), "per_month": (3, 7),
    },
    "Travel": {
        "desc": ["Flight tickets", "Hotel booking", "Airbnb stay",
                 "Car rental", "Baggage fee"],
        "amt": (60.0, 420.0), "per_month": (0, 2),
    },
    "Subscriptions": {
        "desc": ["Amazon Prime membership", "iCloud storage",
                 "Adobe Creative Cloud", "Github Copilot", "Dropbox plan",
                 "Notion subscription"],
        "amt": (6.0, 22.0), "per_month": (2, 5),
    },
    "Education": {
        "desc": ["Udemy course", "Online textbook", "Coursera specialization",
                 "Coding bootcamp", "Certification exam"],
        "amt": (10.0, 180.0), "per_month": (0, 3),
    },
    "Personal Care": {
        "desc": ["Haircut at salon", "Spa massage", "Skincare products",
                 "Toiletries", "Barber trim"],
        "amt": (10.0, 70.0), "per_month": (1, 4),
    },
}


def generate_demo_expenses(months=6, seed=42):
    rng = random.Random(seed)
    today = date.today()
    expenses = []

    anchor = today.replace(day=1)
    for offset in range(months - 1, -1, -1):
        # first day of the month that is `offset` months before the anchor
        m = anchor
        for _ in range(offset):
            m = (m - timedelta(days=1)).replace(day=1)
        days_in_month = calendar.monthrange(m.year, m.month)[1]
        max_day = today.day if (m.year == today.year
                                and m.month == today.month) else days_in_month
        # For the in-progress month, scale counts by the elapsed fraction so
        # the partial month reflects a realistic half-done month, not a full
        # month's worth crammed into the days so far.
        month_frac = max_day / days_in_month

        for cat, tmpl in CATEGORY_TEMPLATES.items():
            lo, hi = tmpl["per_month"]
            count = round(rng.randint(lo, hi) * month_frac)
            if "fixed_day" in tmpl and max_day >= tmpl["fixed_day"]:
                count = 1
            for _ in range(count):
                if cat == "Travel" and rng.random() < 0.45:
                    continue  # travel is sporadic
                if cat == "Education" and rng.random() < 0.45:
                    continue
                if "fixed_day" in tmpl:
                    day = tmpl["fixed_day"]
                else:
                    day = rng.randint(1, max_day)
                day = min(day, max_day)
                d = m.replace(day=day)
                amount = round(rng.uniform(*tmpl["amt"]), 2)
                desc = rng.choice(tmpl["desc"])
                # add light variety to merchant names for realistic text
                if rng.random() < 0.4:
                    desc = f"{desc} #{rng.randint(1000, 9999)}"
                expenses.append({
                    "description": desc,
                    "amount": amount,
                    "category": cat,
                    "date": d.isoformat(),
                })

    # sort by date ascending so it reads naturally
    expenses.sort(key=lambda e: e["date"])
    return expenses