# src/train/train.py

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def parse_args():
    """Parse command line arguments used by the job.

    Returns
    -------
    argparse.Namespace
        Contains ``inputs`` (path) and ``format`` ("csv" or "mltable").
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, required=True)
    parser.add_argument("--format", type=str, choices=["csv", "mltable"], required=True)
    return parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#adult_csv_path = os.path.join(BASE_DIR, "..", "data", "adult_raw", "adult.csv")
print(f"BASE_DIR={BASE_DIR}")
args = parse_args()
print(f"args={args}")

#print(f"Loading data from {adult_csv_path}")

# 2. Access the data set input
print(f"Input {args.format} path: {args.inputs}")

if args.format == "csv":
    print(f"Loading csv file: {args.inputs}")
    df = pd.read_csv(args.inputs)

print(f"Columns: {df.columns.tolist()}")
for c in df.columns:
    print(c)
print(f"Shape: {df.shape}")
print(f"Size: {df.size}")
print(df.head(10))

y = df["income"]
X = df.drop(columns=["income"])

categorical = X.select_dtypes(include=["object"]).columns
numeric = X.select_dtypes(exclude=["object"]).columns

print("Creating SKLearn Cloumn Trasformer...")
preprocess = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ]
)

print("Creating SKLearn Pipeline...")
clf = Pipeline(
    [
        ("prep", preprocess),
        ("model", LogisticRegression(max_iter=200)),
    ]
)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print("Starting MLFlow...")
mlflow.start_run()
mlflow.sklearn.autolog()

registered_model_name="adultset-classifier"

try:
    print("Training model...")
    clf.fit(X_train, y_train)
    print(f"model train completed: {clf}")
    print("Logging MLFLow model...")
    # mlflow.sklearn.log_model(clf, "model")
    mlflow.sklearn.log_model(
        sk_model=clf,
        registered_model_name=registered_model_name,
        artifact_path=registered_model_name,
    )
    print("Saving MLFLow model...")
    mlflow.sklearn.save_model(
        sk_model=clf,
        path=os.path.join(registered_model_name, "trained_model"),
    )

    print("Done.")
except Exception as e:
    print(f"Classification Model training failed: {e}")

mlflow.end_run()

print("Training complete and model logged to MLflow.")