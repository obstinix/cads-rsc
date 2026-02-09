"""
Main Pipeline -- Sales Prediction
===================================
Orchestrates the entire pipeline: Load -> Clean -> Train -> Evaluate ->
Analyze Impact -> Visualize -> Save Model.

This is the entry point for Task 4. Run:
    python src/main.py

Dataset: https://www.kaggle.com/datasets/bumba5341/advertisingcsv
"""

import os
import sys
import numpy as np
import joblib
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error
)

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_data import load_advertising_data, inspect_data
from preprocess import prepare_data
from train_model import train_all_models, get_linear_coefficients
from analyze_impact import analyze_channel_impact, compute_roi_analysis, generate_marketing_insights
from visualize import generate_all_visualizations


def evaluate_all_models(trained_models: dict, X_test, y_test) -> dict:
    """
    Evaluate all models on the test set using regression metrics.

    Metrics:
        - R2 Score: How much variance the model explains (1.0 = perfect)
        - MAE: Average absolute error in predictions
        - RMSE: Root mean squared error (penalizes large errors more)

    Parameters
    ----------
    trained_models : dict
        Model name -> fitted model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True test target.

    Returns
    -------
    dict
        Model name -> evaluation metrics.
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION (Test Set)")
    print("=" * 60)

    all_results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        all_results[name] = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "predictions": y_pred,
            "y_test": y_test,
        }

        print(f"\n--- {name} ---")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  MAE:      {mae:.4f} ($1000s)")
        print(f"  RMSE:     {rmse:.4f} ($1000s)")

    # Rank
    print("\n" + "-" * 60)
    print("RANKING (by R2)")
    print("-" * 60)
    ranked = sorted(all_results.items(), key=lambda x: x[1]["r2"], reverse=True)
    for rank, (name, res) in enumerate(ranked, 1):
        print(f"  {rank}. {name:25s} -> R2: {res['r2']:.4f}  MAE: {res['mae']:.4f}")

    return all_results


def save_model_metrics(all_results: dict, output_dir: str) -> str:
    """Save model metrics to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "model_metrics.txt")

    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("SALES PREDICTION -- MODEL EVALUATION REPORT\n")
        f.write("Dataset: https://www.kaggle.com/datasets/bumba5341/advertisingcsv\n")
        f.write("=" * 70 + "\n\n")

        for name, res in all_results.items():
            f.write(f"Model: {name}\n")
            f.write(f"  R2 Score: {res['r2']:.4f}\n")
            f.write(f"  MAE:     {res['mae']:.4f}\n")
            f.write(f"  RMSE:    {res['rmse']:.4f}\n\n")

        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<28} {'R2':>8} {'MAE':>8} {'RMSE':>8}\n")
        f.write("-" * 70 + "\n")
        ranked = sorted(all_results.items(), key=lambda x: x[1]["r2"], reverse=True)
        for name, res in ranked:
            f.write(f"{name:<28} {res['r2']:>8.4f} {res['mae']:>8.4f} {res['rmse']:>8.4f}\n")

    print(f"[OK] Model metrics saved to: {path}")
    return path


def main():
    """Execute the complete Sales Prediction pipeline."""

    # Paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    models_dir = os.path.join(project_dir, "models")
    results_dir = os.path.join(project_dir, "results")

    print("=" * 60)
    print("  SALES PREDICTION PIPELINE")
    print("  Dataset: Kaggle (bumba5341/advertisingcsv)")
    print("=" * 60)

    # -- Step 1: Load Data --
    print("\n[STEP 1] Loading dataset...")
    df = load_advertising_data(os.path.join(data_dir, "advertising.csv"))
    inspect_data(df)

    # -- Step 2: Preprocess --
    print("\n[STEP 2] Preprocessing data...")
    data = prepare_data(df)

    # -- Step 3: Train Models --
    print("\n[STEP 3] Training models...")
    trained_models, cv_results = train_all_models(data["X_train"], data["y_train"])

    # Get linear regression coefficients (interpretability)
    coefficients = get_linear_coefficients(trained_models, data["feature_names"])

    # -- Step 4: Evaluate --
    print("\n[STEP 4] Evaluating on test set...")
    all_results = evaluate_all_models(trained_models, data["X_test"], data["y_test"])
    save_model_metrics(all_results, results_dir)

    # Identify best model
    best_name = max(all_results, key=lambda k: all_results[k]["r2"])
    best_results = all_results[best_name]

    # -- Step 5: Channel Impact & ROI Analysis --
    print("\n[STEP 5] Analyzing advertising channel impact...")
    impact = analyze_channel_impact(data["df_clean"])
    roi = compute_roi_analysis(data["df_clean"], trained_models, data["feature_names"])
    generate_marketing_insights(data["df_clean"], impact, roi, results_dir)

    # -- Step 6: Visualizations --
    print("\n[STEP 6] Generating visualizations...")
    generate_all_visualizations(
        df=data["df_clean"],
        all_results=all_results,
        cv_results=cv_results,
        trained_models=trained_models,
        best_name=best_name,
        best_results=best_results,
        roi=roi,
        feature_names=data["feature_names"],
        output_dir=results_dir,
    )

    # -- Step 7: Save Best Model --
    print("\n[STEP 7] Saving best model...")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "sales_model.pkl")
    joblib.dump({
        "model": trained_models[best_name],
        "model_name": best_name,
        "r2": best_results["r2"],
        "mae": best_results["mae"],
        "rmse": best_results["rmse"],
        "feature_names": data["feature_names"],
        "coefficients": coefficients,
    }, model_path)
    print(f"[OK] Best model saved: {best_name} (R2: {best_results['r2']:.4f})")
    print(f"     Path: {model_path}")

    # -- Summary --
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  Best Model: {best_name}")
    print(f"  Test R2:    {best_results['r2']:.4f}")
    print(f"  Test MAE:   {best_results['mae']:.4f}")
    print(f"  Test RMSE:  {best_results['rmse']:.4f}")
    print(f"  Results:    {results_dir}")
    print(f"  Model:      {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
