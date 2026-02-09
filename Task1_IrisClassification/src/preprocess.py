"""
Preprocess Module — Iris Flower Classification
===============================================
Handles data cleaning, feature preparation, and train/test splitting.

Steps:
    1. Drop the 'Id' column (not a useful feature)
    2. Encode the target variable (Species) with LabelEncoder
    3. Scale features using StandardScaler
    4. Split into training (80%) and testing (20%) sets

Dataset: https://www.kaggle.com/datasets/saurabh00007/iriscsv
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset by removing unnecessary columns and handling
    any missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Iris dataset from Kaggle CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    df_clean = df.copy()

    # Drop the 'Id' column — it's just a row index, not a feature
    if "Id" in df_clean.columns:
        df_clean.drop("Id", axis=1, inplace=True)
        print("[OK] Dropped 'Id' column (not a feature)")

    # Handle missing values (the Iris dataset is clean, but good practice)
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        df_clean.dropna(inplace=True)
        print(f"[OK] Dropped {missing_count} rows with missing values")
    else:
        print("[OK] No missing values found")

    # Remove duplicate rows if any
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        print(f"[INFO] Found {duplicates} duplicate rows (keeping them — natural in Iris data)")

    return df_clean


def encode_target(df: pd.DataFrame) -> tuple:
    """
    Separate features and target, then encode the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with 'Species' column.

    Returns
    -------
    tuple
        (X, y, label_encoder, feature_names)
        - X: Feature matrix (pd.DataFrame)
        - y: Encoded target array (np.ndarray)
        - label_encoder: Fitted LabelEncoder (for inverse transform)
        - feature_names: List of feature column names
    """
    # Separate features (X) and target (y)
    feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    X = df[feature_cols]
    y_raw = df["Species"]

    # Encode species names to integers: 0, 1, 2
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"[OK] Encoded species: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return X, y, le, feature_cols


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Standardize features using StandardScaler (zero mean, unit variance).

    Why scale? Some algorithms (SVM, KNN) are distance-based and perform
    better when features are on the same scale.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Training and testing feature matrices.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data only
    X_test_scaled = scaler.transform(X_test)          # Transform test data

    print(f"[OK] Features scaled (mean~0, std~1)")
    return X_train_scaled, X_test_scaled, scaler


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Full preprocessing pipeline: clean -> encode -> split -> scale.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Iris dataset.
    test_size : float
        Proportion of data for testing (default: 20%).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with all preprocessed components:
        {X_train, X_test, y_train, y_test, X_train_raw, X_test_raw,
         scaler, label_encoder, feature_names}
    """
    # Step 1: Clean
    df_clean = clean_data(df)

    # Step 2: Encode target
    X, y, le, feature_names = encode_target(df_clean)

    # Step 3: Train/test split (stratified to keep species proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[OK] Split data: {X_train.shape[0]} train / {X_test.shape[0]} test (stratified)")

    # Step 4: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_train_raw": X_train,  # Unscaled — useful for some visualizations
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    from load_data import load_iris_data

    df = load_iris_data()
    data = prepare_data(df)
    print(f"\n[DONE] Preprocessing complete.")
    print(f"  Training set: {data['X_train'].shape}")
    print(f"  Testing set:  {data['X_test'].shape}")
