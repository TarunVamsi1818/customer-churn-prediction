"""Generate a reproducible synthetic dataset for trying the project."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    if rows < 20:
        raise ValueError("Generate at least 20 rows for a useful demo dataset.")

    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, rows)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=rows,
        p=[0.55, 0.25, 0.20],
    )
    internet = rng.choice(
        ["DSL", "Fiber optic", "No"], size=rows, p=[0.35, 0.45, 0.20]
    )
    monthly_charges = np.clip(
        rng.normal(70 + (internet == "Fiber optic") * 15, 20, rows), 18, 130
    ).round(2)
    total_charges = (monthly_charges * tenure + rng.normal(0, 80, rows)).clip(0).round(2)
    churn_score = (
        -1.7
        + (contract == "Month-to-month") * 1.5
        + (internet == "Fiber optic") * 0.45
        - (contract == "Two year") * 1.0
        - tenure * 0.035
        + (monthly_charges > 95) * 0.35
        + rng.normal(0, 0.65, rows)
    )
    churn_probability = 1 / (1 + np.exp(-churn_score))
    churn = np.where(rng.random(rows) < churn_probability, "Yes", "No")

    return pd.DataFrame(
        {
            "customerID": [f"SYN-{index:05d}" for index in range(rows)],
            "gender": rng.choice(["Female", "Male"], rows),
            "SeniorCitizen": rng.choice([0, 1], rows, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], rows),
            "Dependents": rng.choice(["Yes", "No"], rows, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], rows, p=[0.9, 0.1]),
            "InternetService": internet,
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], rows, p=[0.6, 0.4]),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer",
                    "Credit card",
                ],
                rows,
            ),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/customer_churn_sample.csv")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(rows=args.rows, seed=args.seed).to_csv(output, index=False)
    print(f"Wrote {args.rows} synthetic rows to {output}")


if __name__ == "__main__":
    main()
