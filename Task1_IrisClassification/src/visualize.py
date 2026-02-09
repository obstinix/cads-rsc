"""
Visualize Module — Iris Flower Classification
==============================================
Generates publication-quality visualizations for EDA and model results.

Charts generated:
    1. Feature distribution (histograms by species)
    2. Pairplot (scatter matrix of all features)
    3. Correlation heatmap
    4. Confusion matrices for all models
    5. Model comparison bar chart
    6. Feature importance (from Random Forest)

Dataset: https://www.kaggle.com/datasets/saurabh00007/iriscsv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (saves to file without display)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Style Configuration ──────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def plot_feature_distributions(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot histograms of each feature, colored by species.

    This helps us see how each measurement differs across species —
    for example, petal length clearly separates Setosa from the others.
    """
    os.makedirs(output_dir, exist_ok=True)
    features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    pretty_names = ["Sepal Length (cm)", "Sepal Width (cm)",
                    "Petal Length (cm)", "Petal Width (cm)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Feature Distributions by Species", fontsize=16, fontweight="bold")

    colors = {"Iris-setosa": "#2ecc71", "Iris-versicolor": "#3498db", "Iris-virginica": "#e74c3c"}

    for idx, (feature, pretty_name) in enumerate(zip(features, pretty_names)):
        ax = axes[idx // 2, idx % 2]
        for species, color in colors.items():
            subset = df[df["Species"] == species]
            ax.hist(subset[feature], bins=15, alpha=0.6, label=species,
                    color=color, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(pretty_name, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "feature_distributions.png")
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_pairplot(df: pd.DataFrame, output_dir: str) -> str:
    """
    Create a scatter matrix (pairplot) showing relationships between
    all features, colored by species.

    This is one of the most insightful visualizations — it reveals
    which feature combinations best separate the species.
    """
    os.makedirs(output_dir, exist_ok=True)
    features_df = df.drop("Id", axis=1, errors="ignore")

    palette = {"Iris-setosa": "#2ecc71", "Iris-versicolor": "#3498db", "Iris-virginica": "#e74c3c"}
    g = sns.pairplot(features_df, hue="Species", palette=palette,
                     diag_kind="kde", corner=False,
                     plot_kws={"alpha": 0.6, "s": 40, "edgecolor": "white", "linewidth": 0.5})
    g.fig.suptitle("Iris Features — Pairplot", y=1.01, fontsize=16, fontweight="bold")

    path = os.path.join(output_dir, "pairplot.png")
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot a correlation heatmap of the numeric features.

    Strong positive correlation between petal length and petal width
    is expected — they grow together.
    """
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, square=True, linewidths=1, ax=ax,
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)

    # Prettier labels
    labels = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)

    path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_confusion_matrices(all_results: dict, label_encoder, output_dir: str) -> str:
    """
    Plot confusion matrices for all models in a grid layout.

    The confusion matrix shows where models make mistakes — for Iris data,
    most errors are between Versicolor and Virginica (they're similar).
    """
    os.makedirs(output_dir, exist_ok=True)
    n_models = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight="bold")

    class_names = label_encoder.classes_
    # Shorter names for display
    short_names = [name.replace("Iris-", "") for name in class_names]

    for idx, (name, res) in enumerate(all_results.items()):
        ax = axes[idx // 3, idx % 3]
        cm = res["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=short_names, yticklabels=short_names,
                    cbar=False, linewidths=0.5, linecolor="gray")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(f"{name}\n(Acc: {res['accuracy']:.2%})", fontsize=11, fontweight="bold")

    # Hide the 6th subplot (we only have 5 models)
    if n_models < 6:
        for i in range(n_models, 6):
            axes[i // 3, i % 3].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_model_comparison(all_results: dict, cv_results: dict, output_dir: str) -> str:
    """
    Bar chart comparing test accuracy and CV accuracy for all models.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(all_results.keys())
    # Use shorter display names
    display_names = [n.replace("K-Nearest Neighbors", "KNN").replace(
        "Support Vector Machine", "SVM").replace(
        "Logistic Regression", "Log. Reg.").replace(
        "Decision Tree", "Dec. Tree").replace(
        "Random Forest", "Rand. Forest") for n in model_names]

    test_acc = [all_results[n]["accuracy"] for n in model_names]
    cv_acc = [cv_results[n]["mean_score"] for n in model_names]
    cv_std = [cv_results[n]["std_score"] for n in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, test_acc, width, label="Test Accuracy",
                   color="#3498db", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, cv_acc, width, label="CV Accuracy (5-fold)",
                   color="#2ecc71", alpha=0.85, yerr=cv_std,
                   capsize=5, edgecolor="white")

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0.85, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_feature_importance(trained_models: dict, feature_names: list,
                            output_dir: str) -> str:
    """
    Plot feature importance from the Random Forest model.

    This tells us which measurements are most useful for classification.
    Typically, petal measurements dominate in the Iris dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    if "Random Forest" not in trained_models:
        print("[SKIP] Random Forest not found, skipping feature importance.")
        return None

    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    pretty_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    sorted_names = [pretty_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    bars = ax.barh(range(len(sorted_names)), sorted_importances,
                   color=colors, edgecolor="white", height=0.6)

    # Add values on bars
    for bar, val in zip(bars, sorted_importances):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=11)

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # Highest importance at top

    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def generate_all_visualizations(df: pd.DataFrame, all_results: dict,
                                 cv_results: dict, trained_models: dict,
                                 label_encoder, feature_names: list,
                                 output_dir: str) -> list:
    """
    Generate all visualizations and save to the output directory.

    Parameters
    ----------
    df : pd.DataFrame
        Original cleaned dataset (with Species column).
    all_results : dict
        Model evaluation results.
    cv_results : dict
        Cross-validation results.
    trained_models : dict
        Trained model instances.
    label_encoder : LabelEncoder
        For class name decoding.
    feature_names : list
        Feature column names.
    output_dir : str
        Directory to save all plots.

    Returns
    -------
    list
        Paths to all saved visualization files.
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    paths = []
    paths.append(plot_feature_distributions(df, output_dir))
    paths.append(plot_pairplot(df, output_dir))
    paths.append(plot_correlation_heatmap(df, output_dir))
    paths.append(plot_confusion_matrices(all_results, label_encoder, output_dir))
    paths.append(plot_model_comparison(all_results, cv_results, output_dir))
    paths.append(plot_feature_importance(trained_models, feature_names, output_dir))

    print(f"\n[OK] All {len([p for p in paths if p])} visualizations saved to: {output_dir}")
    return paths
