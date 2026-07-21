#!/usr/bin/env python3
"""Initialize or inspect the immutable Moneyball research database."""

from moneyball_predictions.storage import database_summary


if __name__ == "__main__":
    summary = database_summary()
    print("Moneyball research database ready")
    for key, value in summary.items():
        print(f"{key}: {value}")
