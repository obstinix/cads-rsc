"""
Load Data Module -- Sales Prediction
=====================================
Loads the Advertising dataset from the local CSV file (sourced from Kaggle).
Dataset: https://www.kaggle.com/datasets/bumba5341/advertisingcsv

The dataset contains advertising budgets (in thousands $) for TV, Radio,
and Newspaper, along with resulting Sales (in thousands of units).

Functions:
    load_advertising_data() -> pd.DataFrame
    inspect_data(df) -> dict
"""

import os
import pandas as pd


def load_advertising_data(data_path: str = None) -> pd.DataFrame:
    """
    Load the Advertising dataset from CSV.

    Parameters
    ----------
    data_path : str, optional
        Path to the CSV file. Defaults to data/advertising.csv.

    Returns
    -------
    pd.DataFrame
        The loaded Advertising dataset with columns: TV, Radio, Newspaper, Sales.
    """
    if data_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data", "advertising.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please run data/prepare_data.py first, or download from:\n"
            "https://www.kaggle.com/datasets/bumba5341/advertisingcsv"
        )

    df = pd.read_csv(data_path)
    print(f"[OK] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Perform initial inspection of the Advertising dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The Advertising dataset.

    Returns
    -------
    dict
        Dictionary containing inspection results.
    """
    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")

    missing = df.isnull().sum()
    print(f"\nMissing Values:\n{missing}")

    print(f"\nStatistical Summary:\n{df.describe()}")

    # Quick insight: budget ranges
    print("\n--- Advertising Budget Ranges (in $1000s) ---")
    for col in ["TV", "Radio", "Newspaper"]:
        if col in df.columns:
            print(f"  {col:12s}: ${df[col].min():.1f}k - ${df[col].max():.1f}k  (mean: ${df[col].mean():.1f}k)")

    if "Sales" in df.columns:
        print(f"\n--- Sales Range ---")
        print(f"  Sales: {df['Sales'].min():.1f}k - {df['Sales'].max():.1f}k  (mean: {df['Sales'].mean():.1f}k units)")

    print(f"\nFirst 5 Rows:\n{df.head()}")

    return {
        "shape": df.shape,
        "missing_values": missing.sum(),
        "stats": df.describe().to_dict(),
    }


if __name__ == "__main__":
    data = load_advertising_data()
    info = inspect_data(data)
    print(f"\n[SUMMARY] Total missing values: {info['missing_values']}")
