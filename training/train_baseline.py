"""Throwaway logistic-regression baseline for entity resolution.

Replaced by training/train_entity_resolution.py, which records provenance,
leakage groups, and a real artifact manifest.
"""

import argparse

from sklearn.linear_model import LogisticRegression


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    print(f"would train a baseline on {args.input}")
    LogisticRegression()


if __name__ == "__main__":
    main()
