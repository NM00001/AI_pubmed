# AI_pubmed

This repository provides a small demonstration of a machine learning pipeline written in Python.  The script computes descriptive statistics and evaluates several classifiers — Random Forest, AdaBoost and Logistic Regression — using cross‑validation.
To install dependencies, run `pip install -r requirements.txt`.

## Usage

```
python ml_pipeline.py data.csv --target TARGET_COLUMN
```

You can also interactively select a CSV file from the current directory with:

```
python ml_pipeline.py --select --target TARGET_COLUMN
```

The script verifies that the input file is a CSV and prints basic statistics before training each model.  Results of the cross‑validation (accuracy and ROC AUC) are summarised at the end and visualised with a violin plot.

## Data format

- Input must be a CSV file.
- Include a column representing the target variable using the `--target` option.
- Remaining columns are treated as features.

## Disclaimer

This code is provided for educational purposes only. It should not be used for clinical decision making. Ensure that any data you use complies with privacy regulations and institutional guidelines.
