from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommend_v2 import train_complexity_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain the prompt complexity classifier from the relabeled CSV."
    )
    parser.add_argument(
        "--training-csv",
        help="Optional training CSV path. Defaults to prompt_complexity_relabeled.csv when present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_csv = Path(args.training_csv) if args.training_csv else None
    result = train_complexity_classifier(training_csv=training_csv)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
