"""
ml/categorizer.py
AI transaction categorizer.

Combines two complementary signals so it works out-of-the-box and keeps
improving as the user adds transactions:

  1. A supervised TF-IDF + Logistic Regression model trained on a rich
     built-in seed corpus (one label per category).  Gives calibrated
     probabilities -> a natural "confidence" score.
  2. A high-precision keyword rule layer that boosts the score of a
     category when its merchant/description keywords appear.

The vectorizer + classifier are persisted with joblib so the model can be
re-trained on the user's own labelled history and reloaded instantly.
"""
import os
import re
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from core.db import MODEL_DIR

MODEL_PATH = os.path.join(MODEL_DIR, "categorizer.joblib")
META_PATH = os.path.join(MODEL_DIR, "categorizer_meta.json")

CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Transportation",
    "Housing",
    "Utilities",
    "Entertainment",
    "Health & Fitness",
    "Shopping",
    "Travel",
    "Subscriptions",
    "Education",
    "Personal Care",
    "Other",
]

# High-precision keyword hints per category (lower-cased substrings).
KEYWORDS = {
    "Food & Dining": [
        "coffee", "starbucks", "restaurant", "pizza", "mcdonald", "lunch",
        "dinner", "breakfast", "cafe", "café", "food", "meal", "snack",
        "burger", "chipotle", "subway", "doordash", "ubereats", "grubhub",
        "takeout", "ice cream", "bakery", "dominos", "taco", "sushi", "deli",
        "brunch", "bar", "pub", "diner", "kfc", "wendys", "panera",
    ],
    "Groceries": [
        "grocery", "groceries", "walmart", "whole foods", "target", "kroger",
        "safeway", "trader joe", "costco", "aldi", "supermarket", "produce",
        "veggies", "vegetables", "milk", "eggs", "market", "farmers market",
    ],
    "Transportation": [
        "uber", "lyft", "gas", "fuel", "shell", "chevron", "exxon", "bp",
        "metro", "bus", "train", "parking", "toll", "taxi", "ride", "transit",
        "car wash", "oil change", "auto repair", "car service", "metrocard",
    ],
    "Housing": [
        "rent", "mortgage", "lease", "hoa", "property tax", "apartment",
        "landlord", "housing", "maintenance fee", "cleaning service",
    ],
    "Utilities": [
        "electric", "electricity", "water bill", "gas bill", "internet",
        "comcast", "xfinity", "verizon", "at&t", "t-mobile", "utility",
        "utilities", "phone bill", "broadband", "wifi bill", "sewage",
        "trash bill",
    ],
    "Entertainment": [
        "netflix", "spotify", "hulu", "disney", "movie", "cinema", "theater",
        "concert", "game", "steam", "playstation", "xbox", "amusement",
        "ticket", "show", "bowling", "arcade", "spotify premium", "youtube premium",
    ],
    "Health & Fitness": [
        "gym", "fitness", "yoga", "pilates", "doctor", "dentist", "pharmacy",
        "cvs", "walgreens", "medicine", "prescription", "clinic", "hospital",
        "health", "therapy", "vitamins", "copay", "insurance premium",
    ],
    "Shopping": [
        "amazon", "clothes", "clothing", "shoes", "nike", "adidas", "zara",
        "h&m", "store", "mall", "sephora", "best buy", "electronics",
        "furniture", "ikea", "purchase", "gadget", "headphones", "watch",
        "accessories",
    ],
    "Travel": [
        "flight", "airline", "hotel", "airbnb", "booking", "expedia",
        "vacation", "trip", "baggage", "resort", "boarding pass", "travel",
        "car rental", "hertz", "avis", "hostel", "cruise",
    ],
    "Subscriptions": [
        "subscription", "prime", "membership", "patreon", "adobe", "icloud",
        "dropbox", "github copilot", "renewal", "plan renewal", "saas",
        "streaming plan", "apple music", "google one", "notion",
    ],
    "Education": [
        "course", "udemy", "coursera", "textbook", "tuition", "school",
        "college", "university", "class", "workshop", "degree", "exam fee",
        "certification", "tutorial", "bootcamp",
    ],
    "Personal Care": [
        "haircut", "salon", "barber", "spa", "massage", "cosmetics",
        "toiletries", "shampoo", "manicure", "skincare", "razor",
    ],
}

# Realistic seed sentences per category -> trains a strong base model.
SEED = {
    "Food & Dining": [
        "morning coffee at starbucks", "lunch at the office cafeteria",
        "dinner with friends at italian restaurant", "pizza delivery dominos",
        "burger and fries mcdonalds", "chipotle burrito bowl",
        "sushi takeout", "breakfast bagel and coffee", "cafe latte",
        "doordash food delivery", "grab tacos for dinner",
        "ice cream at the bakery", "brunch on sunday",
        "subway sandwich", "late night ramen", "panera soup",
    ],
    "Groceries": [
        "weekly grocery shopping", "whole foods produce run",
        "walmart groceries", "milk eggs and bread", "costco bulk shopping",
        "trader joe snacks", "kroger weekly shop", "aldi groceries",
        "farmers market vegetables", "safeway grocery pickup",
        "target grocery run", "supermarket weekly stock up",
    ],
    "Transportation": [
        "uber ride downtown", "lyft to the airport", "gas station fill up",
        "shell fuel", "metro card top up", "parking garage fee",
        "bridge toll", "bus pass monthly", "train ticket commute",
        "oil change", "car wash", "taxi late night", "chevron gas",
    ],
    "Housing": [
        "monthly rent payment", "mortgage payment", "apartment lease",
        "hoa fees", "property tax", "landlord rent transfer",
        "maintenance fee", "cleaning service for apartment",
    ],
    "Utilities": [
        "electricity bill", "water bill", "internet bill comcast",
        "phone bill verizon", "at&t wireless", "gas bill",
        "xfinity internet", "utility payment", "trash and sewage bill",
    ],
    "Entertainment": [
        "netflix subscription", "spotify premium", "movie night cinema",
        "concert tickets", "steam game purchase", "playstation store",
        "xbox game pass", "amusement park tickets", "theater show",
        "bowling with friends", "youtube premium", "disney plus",
    ],
    "Health & Fitness": [
        "gym membership", "yoga class", "doctor visit copay",
        "dentist cleaning", "pharmacy prescription", "cvs walgreens medicine",
        "vitamins and supplements", "physical therapy session",
        "health insurance premium", "urgent care visit",
    ],
    "Shopping": [
        "amazon online order", "new shoes nike", "clothing at zara",
        "headphones best buy", "ikea furniture", "electronics purchase",
        "sephora skincare", "h&m clothes", "new watch",
        "phone accessories", "gadget purchase",
    ],
    "Travel": [
        "flight tickets airline", "hotel booking", "airbnb stay",
        "expedia vacation package", "trip to hawaii", "baggage fee",
        "resort booking", "car rental hertz", "boarding pass fee",
        "weekend travel trip",
    ],
    "Subscriptions": [
        "amazon prime membership", "adobe creative cloud", "icloud storage",
        "dropbox plan", "patreon pledge", "github copilot subscription",
        "annual renewal", "google one plan", "notion subscription",
        "apple music subscription",
    ],
    "Education": [
        "udemy course", "coursera specialization", "college textbooks",
        "tuition fee", "online class workshop", "certification exam",
        "coding bootcamp", "school supplies", "university course fee",
    ],
    "Personal Care": [
        "haircut at salon", "barber trim", "spa day massage",
        "cosmetics purchase", "toiletries and shampoo", "manicure",
        "skincare products", "razor refill",
    ],
    "Other": [
        "misc purchase", "donation", "gift for friend", "bank fee",
        "atm withdrawal", "unknown transaction", "cash transfer",
        "government fee", "charity donation", "postage shipping",
    ],
}


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9&+ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_scores(text: str) -> dict:
    text = text.lower()
    scores = {c: 0.0 for c in CATEGORIES}
    for cat, kws in KEYWORDS.items():
        for kw in kws:
            if kw in text:
                # longer keyword => more specific => bigger weight
                scores[cat] += 1.0 + len(kw) / 20.0
    return scores


class TransactionCategorizer:
    def __init__(self):
        self.pipeline = None
        self.trained_on = 0

    # -- training -----------------------------------------------------------
    def _seed_dataset(self, extra=None):
        texts, labels = [], []
        for cat, examples in SEED.items():
            for ex in examples:
                texts.append(_clean(ex))
                labels.append(cat)
        if extra:
            for text, cat in extra:
                texts.append(_clean(text))
                labels.append(cat)
        return texts, labels

    def train(self, extra_data=None, persist=True):
        """(Re)train the model on the seed corpus (+ optional user data)."""
        texts, labels = self._seed_dataset(extra_data)
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ])
        self.pipeline.fit(texts, labels)
        self.trained_on = len(texts)
        if persist:
            self.save()
        return self

    # -- persistence --------------------------------------------------------
    def save(self):
        joblib.dump(
            {"pipeline": self.pipeline, "trained_on": self.trained_on},
            MODEL_PATH,
        )
        with open(META_PATH, "w", encoding="utf-8") as fh:
            json.dump(
                {"categories": CATEGORIES, "trained_on": self.trained_on},
                fh, indent=2,
            )

    @classmethod
    def load(cls):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(MODEL_PATH)
        obj = cls()
        data = joblib.load(MODEL_PATH)
        obj.pipeline = data["pipeline"]
        obj.trained_on = data.get("trained_on", 0)
        return obj

    # -- inference ----------------------------------------------------------
    def predict(self, description: str) -> dict:
        """Return {category, confidence, all_scores} for a description."""
        clean = _clean(description or "")
        model_probs = self.pipeline.predict_proba([clean])[0]
        classes = self.pipeline.classes_
        class_index = {c: i for i, c in enumerate(classes)}

        # Blend ML probabilities with keyword evidence.
        kw = _keyword_scores(clean)
        kw_total = sum(kw.values()) or 1.0
        blended = {}
        for cat in classes:
            ml = float(model_probs[class_index[cat]])
            kw_component = (kw.get(cat, 0.0) / kw_total) * 0.45  # 45% weight
            blended[cat] = ml * 0.55 + kw_component

        total = sum(blended.values()) or 1.0
        blended = {c: v / total for c, v in blended.items()}
        best = max(blended, key=blended.get)

        # If nothing matched well, fall back to the keyword winner or Other.
        if blended[best] < 0.12:
            kw_best = max(kw, key=kw.get)
            if kw[kw_best] > 0:
                best = kw_best

        return {
            "category": best,
            "confidence": round(blended[best], 4),
            "all_scores": {k: round(v, 4)
                           for k, v in sorted(blended.items(),
                                              key=lambda x: -x[1])},
        }


# Module-level singleton so training/prediction is shared across requests.
categorizer = TransactionCategorizer()


def get_categorizer():
    """Load the persisted model, training a fresh one if it's missing."""
    if os.path.exists(MODEL_PATH):
        try:
            return TransactionCategorizer.load()
        except Exception:
            pass
    return TransactionCategorizer().train(persist=True)