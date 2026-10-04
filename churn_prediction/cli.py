"""Command-line interface for training and using churn models."""

from __future__ import annotations

import argparse
import json

from churn_prediction.pipeline import predict_file, train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train churn models or predict churn from a CSV file."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate models.")
    train_parser.add_argument("--data", required=True, help="Input training CSV.")
    train_parser.add_argument("--target", default="Churn", help="Target column.")
    train_parser.add_argument(
        "--output-dir", default="artifacts", help="Directory for model and metrics."
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)

    predict_parser = subparsers.add_parser("predict", help="Predict churn for a CSV.")
    predict_parser.add_argument("--data", required=True, help="Input prediction CSV.")
    predict_parser.add_argument(
        "--model", default="artifacts/churn_model.joblib", help="Saved model path."
    )
    predict_parser.add_argument(
        "--output", default="artifacts/predictions.csv", help="Predictions CSV path."
    )

    args = parser.parse_args()
    if args.command == "train":
        print(
            json.dumps(
                train_model(
                    args.data,
                    target_column=args.target,
                    output_dir=args.output_dir,
                    test_size=args.test_size,
                    random_state=args.random_state,
                ),
                indent=2,
            )
        )
    else:
        output = predict_file(args.data, model_path=args.model, output_path=args.output)
        print(f"Predictions saved to {output}")


if __name__ == "__main__":
    main()
