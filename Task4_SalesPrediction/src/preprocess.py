"""
Preprocess Module -- Sales Prediction
======================================
Handles data cleaning and train/test splitting for the advertising dataset.

Since this dataset is relatively clean (no categorical vars, no missing values),
preprocessing is lighter than typical ML pipelines. The focus is on:
    1. Validating data quality
    2. Removing any anomalies
    3. Splitting into training (80%) and testing (20%) sets

Note: We intentionally do NOT scale features for interpretability in linear
regression — the coefficients directly tell us "for every $1k increase in
TV budget, sales increase by X thousand units."

Dataset: https://www.kaggle.com/datasets/bumba5341/advertisingcsv
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the advertising dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw advertising dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    df_clean = df.copy()

    # Check for missing values
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        df_clean.dropna(inplace=True)
        print(f"[OK] Dropped {missing_count} rows with missing values")
    else:
        print("[OK] No missing values found")

    # Check for negative values (budgets and sales should be >= 0)
    numeric_cols = ["TV", "Radio", "Newspaper", "Sales"]
    for col in numeric_cols:
        neg_count = (df_clean[col] < 0).sum()
        if neg_count > 0:
            df_clean = df_clean[df_clean[col] >= 0]
            print(f"[OK] Removed {neg_count} negative values in '{col}'")

    # Check for duplicates
    dups = df_clean.duplicated().sum()
    if dups > 0:
        df_clean.drop_duplicates(inplace=True)
        print(f"[OK] Removed {dups} duplicate rows")
    else:
        print("[OK] No duplicate rows found")

    print(f"[OK] Clean dataset: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
    return df_clean


def prepare_data(df: pd.DataFrame, test_size: float = 0.2,
                 random_state: int = 42) -> dict:
    """
    Full preprocessing pipeline: clean -> separate features/target -> split.

    Parameters
    ----------
    df : pd.DataFrame
        Raw advertising dataset.
    test_size : float
        Proportion of data for testing (default: 20%).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with preprocessed components:
        {X_train, X_test, y_train, y_test, feature_names, df_clean}
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Step 1: Clean
    df_clean = clean_data(df)

    # Step 2: Separate features and target
    feature_cols = ["TV", "Radio", "Newspaper"]
    target_col = "Sales"

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    print(f"\n[OK] Features: {feature_cols}")
    print(f"[OK] Target: {target_col}")

    # Step 3: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"[OK] Split: {X_train.shape[0]} train / {X_test.shape[0]} test")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_cols,
        "df_clean": df_clean,
    }


if __name__ == "__main__":
    from load_data import load_advertising_data
    df = load_advertising_data()
    data = prepare_data(df)
    print(f"\n[DONE] Preprocessing complete.")
    print(f"  Training: {data['X_train'].shape}")
    print(f"  Testing:  {data['X_test'].shape}")
