"""
seed.py — populate the database with ~6 months of realistic demo data.

Usage:
    python seed.py
"""
import core.db as db
from core.demo import generate_demo_expenses
from ml.categorizer import TransactionCategorizer


def main():
    db.init_db()
    print(f"Current transactions: {db.count_expenses()}")
    if db.count_expenses() > 0:
        ans = input("Database is not empty. Clear and reseed? [y/N] ")
        if ans.strip().lower() != "y":
            print("Aborted.")
            return
        db.clear_all()

    demo = generate_demo_expenses(months=6)
    for e in demo:
        db.add_expense(e["description"], e["amount"], e["category"],
                       e["date"], source="demo", confidence=1.0)
    print(f"Added {len(demo)} demo transactions.")

    # Train the categorizer on the seed corpus + the demo history.
    labelled = [(e["description"], e["category"]) for e in db.list_expenses()]
    cat = TransactionCategorizer().train(extra_data=labelled, persist=True)
    print(f"Categorizer trained on {cat.trained_on} examples.")
    print("Done. Run `python app.py` to start the app.")


if __name__ == "__main__":
    main()