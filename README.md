# Customer Churn Prediction

A small, end-to-end machine-learning project that predicts whether a customer
will churn. It compares logistic regression with a random forest, evaluates both
on a stratified holdout set, saves the better model, and can score new CSV files.

> The included sample-data generator creates **synthetic demo data**. Its model
> metrics are not evidence of real-world performance. For real use, train on
> representative, permissioned customer data and validate the model with your
> business and privacy requirements.

## 1. Set up

Use Python 3.10 or later. From the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Generate demo data

```powershell
python generate_sample_data.py
```

This writes `data/customer_churn_sample.csv`. To generate a different-sized
dataset, pass `--rows` and optionally `--seed`.

## 3. Train and compare models

```powershell
python -m churn_prediction.cli train --data data/customer_churn_sample.csv
```

The training command expects a binary target column named `Churn` (`Yes`/`No`,
`true`/`false`, or `1`/`0`). Specify another target with `--target`. It:

1. Checks the input and encodes churn as the positive class.
2. Excludes customer identifier columns, imputes missing values, scales numeric
   columns, and one-hot encodes categorical columns.
3. Makes a stratified train/test split (80/20 by default).
4. Trains logistic regression and random forest models.
5. Reports accuracy, precision, recall, F1, and ROC-AUC; selects the model with
   the highest churn F1 (ROC-AUC breaks ties).
6. Saves the chosen model to `artifacts/churn_model.joblib` and metrics to
   `artifacts/metrics.json`.

Use `--test-size 0.25`, `--random-state 7`, or `--output-dir path` to customize
the run. F1 is the selection metric because churn datasets are often imbalanced;
review precision and recall together against the cost of missed churn and
unnecessary retention offers.

## 4. Score customers

Give the prediction command a CSV with the same customer feature columns used
during training (the target column is not needed):

```powershell
python -m churn_prediction.cli predict `
  --data new_customers.csv `
  --model artifacts/churn_model.joblib `
  --output artifacts/predictions.csv
```

The output CSV contains `churn_prediction` (`Yes`/`No`) and
`churn_probability` for each input row. Unknown categories are handled, and
missing expected columns are imputed by the saved pipeline.

## 5. Run tests

```powershell
python -m unittest discover -s tests -v
```

## Use a real dataset

Replace the demo CSV with a permissioned customer dataset that has a binary
churn outcome and one row per customer. Define churn consistently (for example,
account cancellation within a specified future period), and ensure every
feature would be available at the time a prediction is made. Exclude direct
identifiers and post-churn information. Evaluate on a time-based holdout if
production use will predict future churn, and monitor data drift and model
performance after deployment.
