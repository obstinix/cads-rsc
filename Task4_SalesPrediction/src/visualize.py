"""
Visualize Module -- Sales Prediction
=====================================
Generates publication-quality visualizations for EDA and model results.

Charts generated:
    1. Scatter plots: Each channel vs Sales (with regression line)
    2. Correlation heatmap
    3. Pairplot of all features and target
    4. Residual plots for the best model
    5. Actual vs Predicted scatter plot
    6. Model comparison bar chart
    7. Advertising channel ROI comparison
    8. Sales distribution histogram

Dataset: https://www.kaggle.com/datasets/bumba5341/advertisingcsv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -- Style Configuration --
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def plot_channel_vs_sales(df: pd.DataFrame, output_dir: str) -> str:
    """
    Scatter plots for each advertising channel vs Sales with regression lines.

    This is the most important EDA plot -- it immediately shows which channels
    have the strongest linear relationship with sales.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Advertising Spend vs Sales", fontsize=16, fontweight="bold")

    channels = ["TV", "Radio", "Newspaper"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for idx, (channel, color) in enumerate(zip(channels, colors)):
        ax = axes[idx]
        ax.scatter(df[channel], df["Sales"], alpha=0.5, color=color,
                   edgecolors="white", linewidths=0.5, s=50)

        # Add regression line
        z = np.polyfit(df[channel], df["Sales"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[channel].min(), df[channel].max(), 100)
        ax.plot(x_line, p(x_line), "--", color="black", alpha=0.6, linewidth=1.5)

        # Correlation in corner
        corr = df[channel].corr(df["Sales"])
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=12, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        ax.set_xlabel(f"{channel} Budget ($1000s)", fontsize=12)
        ax.set_ylabel("Sales (1000s units)", fontsize=12)
        ax.set_title(f"{channel} vs Sales", fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(output_dir, "advertising_impact.png")
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """
    Correlation heatmap showing relationships between all variables.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[["TV", "Radio", "Newspaper", "Sales"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="RdYlBu_r",
                center=0, square=True, linewidths=1.5, ax=ax,
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.8},
                annot_kws={"fontsize": 13, "fontweight": "bold"})

    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)

    path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_actual_vs_predicted(y_test, y_pred, model_name: str, output_dir: str) -> str:
    """
    Scatter plot of actual vs predicted sales values.

    A perfect model would have all points on the diagonal line.
    Points far from the line are where the model struggles.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_test, y_pred, alpha=0.6, color="#3498db",
               edgecolors="white", linewidths=0.5, s=70)

    # Perfect prediction line
    min_val = min(y_test.min(), min(y_pred))
    max_val = max(y_test.max(), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="#e74c3c",
            linewidth=2, label="Perfect Prediction")

    ax.set_xlabel("Actual Sales (1000s)", fontsize=13)
    ax.set_ylabel("Predicted Sales (1000s)", fontsize=13)
    ax.set_title(f"Actual vs Predicted Sales ({model_name})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    # Add R2 text
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.95, f"R2 = {r2:.4f}", transform=ax.transAxes,
            fontsize=13, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

    path = os.path.join(output_dir, "prediction_accuracy.png")
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_residuals(y_test, y_pred, model_name: str, output_dir: str) -> str:
    """
    Plot residuals (errors) to check model assumptions.

    Good model: residuals should be randomly scattered around zero
    with no clear patterns.
    """
    os.makedirs(output_dir, exist_ok=True)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Residual Analysis ({model_name})", fontsize=14, fontweight="bold")

    # Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.6, color="#e74c3c",
                edgecolors="white", linewidths=0.5, s=50)
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Predicted Sales", fontsize=11)
    ax1.set_ylabel("Residuals", fontsize=11)
    ax1.set_title("Residuals vs Predicted", fontsize=12)

    # Residual distribution
    ax2 = axes[1]
    ax2.hist(residuals, bins=20, alpha=0.7, color="#3498db",
             edgecolor="white", linewidth=0.5)
    ax2.axvline(x=0, color="#e74c3c", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Residual Value", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Residual Distribution", fontsize=12)

    mean_res = np.mean(residuals)
    ax2.text(0.95, 0.95, f"Mean: {mean_res:.3f}\nStd: {np.std(residuals):.3f}",
             transform=ax2.transAxes, fontsize=10, verticalalignment="top",
             ha="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(output_dir, "residual_plot.png")
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_model_comparison(all_results: dict, cv_results: dict, output_dir: str) -> str:
    """
    Bar chart comparing R2 and MAE for all models.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(all_results.keys())
    display_names = [n.replace("Regression", "Reg.").replace("Random Forest", "Rand. Forest")
                     .replace("Gradient Boosting", "Grad. Boost") for n in model_names]

    test_r2 = [all_results[n]["r2"] for n in model_names]
    cv_r2 = [cv_results[n]["r2_mean"] for n in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, test_r2, width, label="Test R2",
                   color="#3498db", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, cv_r2, width, label="CV R2 (5-fold)",
                   color="#2ecc71", alpha=0.85, edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("R2 Score", fontsize=12)
    ax.set_title("Model Performance Comparison (R2)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0.7, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_roi_analysis(roi: dict, output_dir: str) -> str:
    """
    Horizontal bar chart showing ROI per advertising channel.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not roi:
        print("[SKIP] No ROI data available")
        return None

    channels = list(roi.keys())
    coefficients = [roi[ch]["coefficient"] for ch in channels]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if c > 0.03 else "#f39c12" if c > 0.01 else "#95a5a6"
              for c in coefficients]

    bars = ax.barh(channels, coefficients, color=colors, edgecolor="white", height=0.5)

    for bar, val in zip(bars, coefficients):
        label = f"{val:.4f} ({abs(val)*1000:.0f} units per $1k)"
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=10)

    ax.set_xlabel("Coefficient (Sales increase per $1k spent)", fontsize=12)
    ax.set_title("Advertising ROI by Channel", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    path = os.path.join(output_dir, "roi_analysis.png")
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_sales_forecast(df: pd.DataFrame, best_model, feature_names: list,
                        output_dir: str) -> str:
    """
    Show how sales change with varying TV budget (holding others at median).

    This is a practical tool: "If I spend $X on TV, how much can I expect to sell?"
    """
    os.makedirs(output_dir, exist_ok=True)

    # Vary TV budget from 0 to 300, hold Radio and Newspaper at median
    radio_med = df["Radio"].median()
    news_med = df["Newspaper"].median()

    tv_range = np.linspace(0, 300, 100)
    X_forecast = pd.DataFrame({
        "TV": tv_range,
        "Radio": [radio_med] * len(tv_range),
        "Newspaper": [news_med] * len(tv_range),
    })

    predicted_sales = best_model.predict(X_forecast[feature_names])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tv_range, predicted_sales, color="#e74c3c", linewidth=2.5, label="Predicted Sales")
    ax.fill_between(tv_range, predicted_sales * 0.9, predicted_sales * 1.1,
                    alpha=0.15, color="#e74c3c", label="~10% confidence band")

    # Mark some key points
    for tv_val in [50, 100, 150, 200, 250]:
        idx = np.argmin(np.abs(tv_range - tv_val))
        ax.plot(tv_val, predicted_sales[idx], "ko", markersize=6)
        ax.annotate(f"  ${tv_val}k -> {predicted_sales[idx]:.1f}k units",
                    (tv_val, predicted_sales[idx]),
                    fontsize=9, color="#2c3e50")

    ax.set_xlabel("TV Advertising Budget ($1000s)", fontsize=12)
    ax.set_ylabel("Predicted Sales (1000s units)", fontsize=12)
    ax.set_title(f"Sales Forecast vs TV Budget\n(Radio=${radio_med:.0f}k, Newspaper=${news_med:.0f}k fixed)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    path = os.path.join(output_dir, "sales_forecast.png")
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def generate_all_visualizations(df, all_results, cv_results, trained_models,
                                 best_name, best_results, roi, feature_names,
                                 output_dir) -> list:
    """
    Generate all visualizations for Task 4.

    Returns
    -------
    list
        Paths to all saved visualization files.
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    paths = []
    paths.append(plot_channel_vs_sales(df, output_dir))
    paths.append(plot_correlation_heatmap(df, output_dir))
    paths.append(plot_actual_vs_predicted(
        all_results[best_name]["y_test"],
        all_results[best_name]["predictions"],
        best_name, output_dir
    ))
    paths.append(plot_residuals(
        all_results[best_name]["y_test"],
        all_results[best_name]["predictions"],
        best_name, output_dir
    ))
    paths.append(plot_model_comparison(all_results, cv_results, output_dir))
    paths.append(plot_roi_analysis(roi, output_dir))
    paths.append(plot_sales_forecast(
        df, trained_models[best_name], feature_names, output_dir
    ))

    saved = [p for p in paths if p]
    print(f"\n[OK] All {len(saved)} visualizations saved to: {output_dir}")
    return paths
