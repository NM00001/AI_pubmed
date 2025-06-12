"""General machine learning pipeline with model comparison.

This example script performs basic descriptive statistics and evaluates
several classifiers using cross‑validation. It is provided for
educational purposes only and should not be used for any clinical
decision making.
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time


def select_dataset(directory: str = '.') -> str:
    """Interactively select a CSV file from the given directory."""
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError('No CSV files found in the directory.')
    print('Available CSV files:')
    for idx, name in enumerate(csv_files, start=1):
        print(f'{idx}. {name}')
    choice = input('Select a file by number: ')
    try:
        index = int(choice) - 1
        return os.path.join(directory, csv_files[index])
    except (ValueError, IndexError):
        raise ValueError('Invalid selection.')


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame after validating the path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Dataset not found: {file_path}')
    if not file_path.endswith('.csv'):
        raise ValueError('Only CSV files are supported.')
    return pd.read_csv(file_path)


def descriptive_stats(df: pd.DataFrame):
    """Print descriptive statistics for the DataFrame."""
    print('Descriptive statistics:')
    print(df.describe(include='all'))


def run_models(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Run several models with nested cross‑validation."""
    if target not in df.columns:
        raise ValueError(f'Target column `{target}` not found in data.')

    X = df.drop(columns=[target])
    y = df[target].astype(str)

    models = {
        "Baseline": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]),
            "param_grid": {},
        },
        "Random Forest": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(random_state=42)),
            ]),
            "param_grid": {"clf__n_estimators": [100, 300], "clf__max_depth": [5, 10, None]},
        },
        "AdaBoost": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", AdaBoostClassifier(random_state=42)),
            ]),
            "param_grid": {"clf__n_estimators": [50, 100], "clf__learning_rate": [0.5, 1.0]},
        },
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]),
            "param_grid": {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__solver": ["liblinear", "lbfgs"],
            },
        },
    }

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = []

    for name, spec in models.items():
        start = time.time()
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            grid = GridSearchCV(spec["pipeline"], spec["param_grid"], cv=inner_cv, scoring="roc_auc", n_jobs=-1)
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            try:
                y_proba = grid.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

            results.append({
                "model": name,
                "fold": fold,
                "accuracy": acc,
                "auc": auc,
            })

        print(f"{name} completed in {time.time() - start:.1f}s")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Run cross-validated models on a CSV dataset.'
    )
    parser.add_argument('data_file', nargs='?', help='Path to CSV file containing the data.')
    parser.add_argument('--target', required=True, help='Name of the target column.')
    parser.add_argument('--select', action='store_true', help='Interactively select a CSV file from current directory.')
    args = parser.parse_args()

    if args.select:
        data_path = select_dataset()
    elif args.data_file:
        data_path = args.data_file
    else:
        parser.error('Please provide a data file or use --select to choose one.')

    df = load_data(data_path)
    descriptive_stats(df)

    results = run_models(df, args.target)
    print('\nSummary:')
    print(results.groupby('model')[['accuracy', 'auc']].mean().round(3))

    # Plot accuracy distribution
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=results, x='model', y='accuracy', inner='box')
    plt.title('Accuracy distribution by model')
    plt.tight_layout()
    plt.show()

    print('Pipeline completed successfully.')


if __name__ == '__main__':
    main()
