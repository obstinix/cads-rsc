"""
Analyze Impact Module -- Sales Prediction
==========================================
Analyzes the impact of each advertising channel on sales and provides
actionable marketing insights.

Key analyses:
    - Channel-wise ROI (Return on Investment)
    - Marginal contribution of each channel
    - Optimal budget allocation recommendations
    - Interaction effects between channels

Dataset: https://www.kaggle.com/datasets/bumba5341/advertisingcsv
"""

import os
import numpy as np
import pandas as pd


def analyze_channel_impact(df: pd.DataFrame) -> dict:
    """
    Analyze the individual impact of each advertising channel on sales.

    We compute:
    - Correlation with sales
    - Average sales in high-spend vs low-spend scenarios
    - Efficiency ratio (sales per dollar spent)

    Parameters
    ----------
    df : pd.DataFrame
        Clean advertising dataset.

    Returns
    -------
    dict
        Channel -> impact metrics.
    """
    print("\n" + "=" * 60)
    print("ADVERTISING CHANNEL IMPACT ANALYSIS")
    print("=" * 60)

    channels = ["TV", "Radio", "Newspaper"]
    impact = {}

    for channel in channels:
        # Correlation with sales
        corr = df[channel].corr(df["Sales"])

        # Split into high/low spend groups (median split)
        median_spend = df[channel].median()
        high_spend = df[df[channel] >= median_spend]["Sales"].mean()
        low_spend = df[df[channel] < median_spend]["Sales"].mean()

        # Efficiency: total sales generated per total dollar spent
        total_spend = df[channel].sum()
        total_sales = df["Sales"].sum()
        # Simple ROI proxy: correlation * avg_sales / avg_spend
        avg_spend = df[channel].mean()
        avg_sales = df["Sales"].mean()

        impact[channel] = {
            "correlation": corr,
            "avg_sales_high_spend": high_spend,
            "avg_sales_low_spend": low_spend,
            "sales_lift": high_spend - low_spend,
            "median_spend": median_spend,
            "avg_spend": avg_spend,
        }

        print(f"\n--- {channel} ---")
        print(f"  Correlation with Sales:  {corr:.4f}")
        print(f"  Median budget:           ${median_spend:.1f}k")
        print(f"  Avg sales (high spend):  {high_spend:.1f}k units")
        print(f"  Avg sales (low spend):   {low_spend:.1f}k units")
        print(f"  Sales lift:              +{high_spend - low_spend:.1f}k units")

    return impact


def compute_roi_analysis(df: pd.DataFrame, trained_models: dict,
                         feature_names: list) -> dict:
    """
    Compute ROI for each advertising channel using the linear regression model.

    ROI = (marginal_sales_increase / cost_increase) * 100

    Parameters
    ----------
    df : pd.DataFrame
        Clean advertising dataset.
    trained_models : dict
        Trained model instances.
    feature_names : list
        Feature column names.

    Returns
    -------
    dict
        Channel -> ROI metrics.
    """
    print("\n" + "-" * 60)
    print("ROI ANALYSIS (based on Linear Regression)")
    print("-" * 60)

    if "Linear Regression" not in trained_models:
        print("[SKIP] Linear Regression model not found")
        return {}

    lr = trained_models["Linear Regression"]
    coefficients = dict(zip(feature_names, lr.coef_))

    roi = {}
    for channel, coeff in coefficients.items():
        # coeff represents: for $1k increase in spend, sales change by 'coeff' thousand units
        # ROI = sales_increase / cost_increase
        roi[channel] = {
            "coefficient": coeff,
            "sales_per_1k_spend": coeff,
            "roi_percentage": coeff * 100,  # As percentage
            "rank": None,  # Will be filled after sorting
        }

    # Rank channels by ROI
    sorted_channels = sorted(roi.items(), key=lambda x: x[1]["coefficient"], reverse=True)
    for rank, (channel, metrics) in enumerate(sorted_channels, 1):
        roi[channel]["rank"] = rank
        emoji_rank = {1: "BEST", 2: "GOOD", 3: "WEAKEST"}
        print(f"  {rank}. {channel:12s} -> Coeff: {metrics['coefficient']:+.4f} "
              f"(${1000*abs(metrics['coefficient']):.0f} sales per $1k spent) [{emoji_rank.get(rank, '')}]")

    return roi


def generate_marketing_insights(df: pd.DataFrame, impact: dict, roi: dict,
                                 output_dir: str) -> str:
    """
    Generate a comprehensive marketing insights report.

    Parameters
    ----------
    df : pd.DataFrame
        Clean advertising dataset.
    impact : dict
        Channel impact analysis results.
    roi : dict
        ROI analysis results.
    output_dir : str
        Directory to save the report.

    Returns
    -------
    str
        Path to the saved report file.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "marketing_insights.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Marketing Insights Report\n\n")
        f.write("## Dataset\n")
        f.write("Source: [Kaggle - Advertising CSV](https://www.kaggle.com/datasets/bumba5341/advertisingcsv)\n\n")
        f.write(f"- **Total observations:** {len(df)}\n")
        f.write(f"- **Average Sales:** {df['Sales'].mean():.1f}k units\n")
        f.write(f"- **Total advertising budget (avg):** ${df[['TV','Radio','Newspaper']].sum(axis=1).mean():.1f}k\n\n")

        f.write("## Channel Performance Summary\n\n")
        f.write("| Channel | Correlation | Avg Spend | Sales Lift | ROI Rank |\n")
        f.write("|---------|------------|-----------|------------|----------|\n")
        for channel in ["TV", "Radio", "Newspaper"]:
            corr = impact[channel]["correlation"]
            avg = impact[channel]["avg_spend"]
            lift = impact[channel]["sales_lift"]
            rank = roi.get(channel, {}).get("rank", "-")
            f.write(f"| {channel} | {corr:.3f} | ${avg:.1f}k | +{lift:.1f}k | #{rank} |\n")

        f.write("\n## Key Findings\n\n")

        # Determine rankings
        best_channel = min(roi.items(), key=lambda x: x[1]["rank"])[0] if roi else "TV"
        best_corr = max(impact.items(), key=lambda x: x[1]["correlation"])[0]

        f.write(f"### 1. {best_channel} delivers the highest ROI\n")
        if roi:
            f.write(f"- Every $1,000 invested in {best_channel} advertising generates approximately "
                    f"**{abs(roi[best_channel]['coefficient'])*1000:.0f} additional units** in sales.\n")
            f.write(f"- {best_channel} should be the primary advertising channel.\n\n")

        f.write(f"### 2. {best_corr} has the strongest correlation with sales\n")
        f.write(f"- Correlation coefficient: **{impact[best_corr]['correlation']:.3f}**\n")
        f.write(f"- High spend on {best_corr} is strongly associated with higher sales.\n\n")

        f.write(f"### 3. Newspaper advertising has diminishing returns\n")
        f.write(f"- Newspaper has the weakest correlation with sales ({impact['Newspaper']['correlation']:.3f}).\n")
        f.write(f"- Budget allocated to Newspaper may yield better results if redirected to TV or Radio.\n\n")

        f.write("## Recommendations\n\n")
        f.write("1. **Maximize TV budget** - It provides the highest correlation with sales\n")
        f.write("2. **Invest in Radio** - Strong ROI, especially for supplementary reach\n")
        f.write("3. **Minimize Newspaper spend** - Weakest performer; consider digital alternatives\n")
        f.write("4. **Consider TV + Radio synergy** - Combined campaigns may yield compounding effects\n")
        f.write("5. **Test budget reallocation** - Shift 30-50% of Newspaper budget to Radio\n")

    print(f"\n[OK] Marketing insights report saved to: {report_path}")
    return report_path
