"""
Main Pipeline — Iris Flower Classification
============================================
Orchestrates the entire pipeline: Load -> Clean -> Train -> Evaluate -> Visualize.

This is the entry point for Task 1. Run this script to execute everything:
    python src/main.py

Dataset: https://www.kaggle.com/datasets/saurabh00007/iriscsv
"""

import os
import sys
import joblib

# Add the src directory to the path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_data import load_iris_data, inspect_data
from preprocess import prepare_data, clean_data
from train_model import train_all_models
from evaluate import evaluate_all_models, save_classification_report, get_best_model
from visualize import generate_all_visualizations


def main():
    """
    Execute the complete Iris Classification pipeline.

    Steps:
        1. Load data from CSV (Kaggle source)
        2. Inspect and explore the dataset
        3. Preprocess (clean, encode, scale, split)
        4. Train 5 classification models with cross-validation
        5. Evaluate all models on the test set
        6. Generate visualizations
        7. Save the best model
    """
    # ── Paths ────────────────────────────────────────────────────────────────
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    models_dir = os.path.join(project_dir, "models")
    results_dir = os.path.join(project_dir, "results")

    print("=" * 60)
    print("  IRIS FLOWER CLASSIFICATION PIPELINE")
    print("  Dataset: Kaggle (saurabh00007/iriscsv)")
    print("=" * 60)

    # ── Step 1: Load Data ────────────────────────────────────────────────────
    print("\n[STEP 1] Loading dataset...")
    df = load_iris_data(os.path.join(data_dir, "iris.csv"))
    info = inspect_data(df)

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    print("\n[STEP 2] Preprocessing data...")
    data = prepare_data(df, test_size=0.2, random_state=42)

    # ── Step 3: Train Models ─────────────────────────────────────────────────
    print("\n[STEP 3] Training models...")
    trained_models, cv_results = train_all_models(data["X_train"], data["y_train"])

    # ── Step 4: Evaluate on Test Set ─────────────────────────────────────────
    print("\n[STEP 4] Evaluating on test set...")
    all_results = evaluate_all_models(
        trained_models, data["X_test"], data["y_test"], data["label_encoder"]
    )

    # Save classification report
    save_classification_report(all_results, results_dir)

    # ── Step 5: Visualizations ───────────────────────────────────────────────
    print("\n[STEP 5] Generating visualizations...")
    # We need the cleaned DataFrame (with Species) for some plots
    df_for_viz = df.copy()
    generate_all_visualizations(
        df=df_for_viz,
        all_results=all_results,
        cv_results=cv_results,
        trained_models=trained_models,
        label_encoder=data["label_encoder"],
        feature_names=data["feature_names"],
        output_dir=results_dir,
    )

    # ── Step 6: Save Best Model ──────────────────────────────────────────────
    print("\n[STEP 6] Saving best model...")
    best_name, best_model, best_results = get_best_model(all_results, trained_models)

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "accuracy": best_results["accuracy"],
        "scaler": data["scaler"],
        "label_encoder": data["label_encoder"],
        "feature_names": data["feature_names"],
    }, model_path)
    print(f"[OK] Best model saved: {best_name} (accuracy: {best_results['accuracy']:.4f})")
    print(f"     Path: {model_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  Best Model: {best_name}")
    print(f"  Test Accuracy: {best_results['accuracy']:.4f}")
    print(f"  Results saved to: {results_dir}")
    print(f"  Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
