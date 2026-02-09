"""
Load Data Module — Iris Flower Classification
==============================================
Loads the Iris dataset from the local CSV file (sourced from Kaggle).
Dataset: https://www.kaggle.com/datasets/saurabh00007/iriscsv

Functions:
    load_iris_data() -> pd.DataFrame
    inspect_data(df) -> None
"""

import os
import pandas as pd


def load_iris_data(data_path: str = None) -> pd.DataFrame:
    """
    Load the Iris dataset from CSV.

    Parameters
    ----------
    data_path : str, optional
        Path to the CSV file. Defaults to data/iris.csv relative to this script.

    Returns
    -------
    pd.DataFrame
        The loaded Iris dataset.
    """
    if data_path is None:
        # Default: look in the data/ folder relative to the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data", "iris.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please run data/prepare_data.py first, or download from:\n"
            "https://www.kaggle.com/datasets/saurabh00007/iriscsv"
        )

    df = pd.read_csv(data_path)
    print(f"[OK] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Perform initial inspection of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The Iris dataset.

    Returns
    -------
    dict
        Dictionary containing inspection results.
    """
    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    # Basic shape
    print(f"\nShape: {df.shape}")

    # Column info
    print(f"\nColumns: {list(df.columns)}")

    # Data types
    print(f"\nData Types:\n{df.dtypes}")

    # Missing values
    missing = df.isnull().sum()
    print(f"\nMissing Values:\n{missing}")

    # Basic statistics
    print(f"\nStatistical Summary:\n{df.describe()}")

    # Target distribution
    if "Species" in df.columns:
        print(f"\nSpecies Distribution:\n{df['Species'].value_counts()}")

    # First few rows
    print(f"\nFirst 5 Rows:\n{df.head()}")

    return {
        "shape": df.shape,
        "missing_values": missing.sum(),
        "species_counts": df["Species"].value_counts().to_dict() if "Species" in df.columns else {},
    }


if __name__ == "__main__":
    data = load_iris_data()
    info = inspect_data(data)
    print(f"\n[SUMMARY] Total missing values: {info['missing_values']}")
    print(f"[SUMMARY] Species: {info['species_counts']}")
