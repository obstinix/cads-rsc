"""
Evaluate Module — Iris Flower Classification
=============================================
Evaluates trained models on the test set and generates performance metrics.

Metrics:
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix
    - Classification Report

Dataset: https://www.kaggle.com/datasets/saurabh00007/iriscsv
"""

import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str, label_encoder=None) -> dict:
    """
    Evaluate a single model on the test set.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    X_test : np.ndarray
        Test features (scaled).
    y_test : np.ndarray
        True test labels.
    model_name : str
        Name of the model (for display).
    label_encoder : LabelEncoder, optional
        For decoding class names.

    Returns
    -------
    dict
        Evaluation metrics: accuracy, precision, recall, f1,
        confusion_matrix, predictions, classification_report (str).
    """
    # Predict
    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    # Classification report (readable string)
    target_names = list(label_encoder.classes_) if label_encoder else None
    report = classification_report(y_test, y_pred, target_names=target_names)

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "classification_report": report,
    }


def evaluate_all_models(trained_models: dict, X_test: np.ndarray,
                        y_test: np.ndarray, label_encoder=None) -> dict:
    """
    Evaluate all trained models and compare their performance.

    Parameters
    ----------
    trained_models : dict
        Model name -> fitted sklearn estimator.
    X_test : np.ndarray
        Test features (scaled).
    y_test : np.ndarray
        True test labels.
    label_encoder : LabelEncoder, optional
        For decoding class names.

    Returns
    -------
    dict
        Model name -> evaluation results dict.
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION (Test Set)")
    print("=" * 60)

    all_results = {}
    for name, model in trained_models.items():
        results = evaluate_model(model, X_test, y_test, name, label_encoder)
        all_results[name] = results

        print(f"\n--- {name} ---")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1']:.4f}")

    # Rank by accuracy
    print("\n" + "-" * 60)
    print("TEST SET RANKING (by accuracy)")
    print("-" * 60)
    ranked = sorted(all_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for rank, (name, res) in enumerate(ranked, 1):
        print(f"  {rank}. {name:30s} -> Accuracy: {res['accuracy']:.4f}")

    return all_results


def save_classification_report(all_results: dict, output_dir: str) -> str:
    """
    Save a detailed classification report to a text file.

    Parameters
    ----------
    all_results : dict
        Model name -> evaluation results.
    output_dir : str
        Directory to save the report.

    Returns
    -------
    str
        Path to the saved report file.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "classification_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("IRIS FLOWER CLASSIFICATION — MODEL EVALUATION REPORT\n")
        f.write("Dataset: https://www.kaggle.com/datasets/saurabh00007/iriscsv\n")
        f.write("=" * 70 + "\n\n")

        for name, res in all_results.items():
            f.write(f"{'=' * 50}\n")
            f.write(f"Model: {name}\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Accuracy:  {res['accuracy']:.4f}\n")
            f.write(f"Precision: {res['precision']:.4f}\n")
            f.write(f"Recall:    {res['recall']:.4f}\n")
            f.write(f"F1-Score:  {res['f1']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(res["confusion_matrix"]) + "\n\n")
            f.write("Classification Report:\n")
            f.write(res["classification_report"] + "\n\n")

        # Summary table
        f.write("=" * 70 + "\n")
        f.write("SUMMARY — ALL MODELS\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Model':<32} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        f.write("-" * 72 + "\n")
        ranked = sorted(all_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        for name, res in ranked:
            f.write(
                f"{name:<32} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
                f"{res['recall']:>10.4f} {res['f1']:>10.4f}\n"
            )

    print(f"\n[OK] Classification report saved to: {report_path}")
    return report_path


def get_best_model(all_results: dict, trained_models: dict) -> tuple:
    """
    Identify the best performing model based on test accuracy.

    Returns
    -------
    tuple
        (best_model_name, best_model, best_results)
    """
    best_name = max(all_results, key=lambda k: all_results[k]["accuracy"])
    return best_name, trained_models[best_name], all_results[best_name]
