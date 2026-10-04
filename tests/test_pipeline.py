from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from churn_prediction.pipeline import _encode_target, predict_file, train_model
from generate_sample_data import generate_dataset


class TargetEncodingTests(unittest.TestCase):
    def test_encodes_yes_as_positive_class(self) -> None:
        encoded = _encode_target(pd.Series(["No", "Yes", " no "]))
        self.assertEqual(encoded.tolist(), [0, 1, 0])

    def test_rejects_non_binary_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two classes"):
            _encode_target(pd.Series(["Yes", "No", "Maybe"]))


class TrainingAndPredictionTests(unittest.TestCase):
    def test_trains_saves_metrics_and_predicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "customers.csv"
            model_dir = root / "model"
            predictions_path = root / "predictions.csv"
            generate_dataset(rows=200, seed=7).to_csv(data_path, index=False)

            report = train_model(data_path, output_dir=model_dir, random_state=7)
            self.assertIn(
                report["selected_model"], {"logistic_regression", "random_forest"}
            )
            self.assertTrue((model_dir / "churn_model.joblib").is_file())
            self.assertEqual(
                json.loads((model_dir / "metrics.json").read_text())["rows"], 200
            )

            predict_file(
                data_path,
                model_path=model_dir / "churn_model.joblib",
                output_path=predictions_path,
            )
            predictions = pd.read_csv(predictions_path)
            self.assertEqual(len(predictions), 200)
            self.assertEqual(
                predictions.columns.tolist(),
                ["churn_prediction", "churn_probability"],
            )
            self.assertTrue(
                predictions["churn_prediction"].isin(["Yes", "No"]).all()
            )
            self.assertTrue(
                predictions["churn_probability"].between(0, 1).all()
            )


if __name__ == "__main__":
    unittest.main()
