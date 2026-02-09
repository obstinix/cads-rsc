# 📺 Task 4: Sales Prediction using Python

> Predict product sales based on advertising spend across TV, Radio, and Newspaper channels using regression models, and provide actionable marketing insights.

## Status: ✅ Complete

## Dataset
- **Source:** [Kaggle — Advertising CSV](https://www.kaggle.com/datasets/bumba5341/advertisingcsv)
- **Features:** TV, Radio, Newspaper (advertising budgets in $1000s)
- **Target:** Sales (in 1000s of units)
- **Samples:** 200

## Tools & Libraries
- Python 3.12, Pandas, NumPy
- Scikit-learn (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting)
- Matplotlib, Seaborn
- Joblib (model persistence)

## Project Structure
```
Task4_SalesPrediction/
├── data/
│   ├── prepare_data.py       → Script to generate advertising.csv
│   └── advertising.csv        → The dataset (200 records)
├── src/
│   ├── main.py                → Main pipeline (run this!)
│   ├── load_data.py           → Data loading & inspection
│   ├── preprocess.py          → Data validation & train/test split
│   ├── train_model.py         → 5 regression models + cross-validation
│   ├── analyze_impact.py      → Channel ROI & marketing insights
│   └── visualize.py           → All 7 visualizations
├── models/
│   └── sales_model.pkl        → Saved Random Forest model (best)
└── results/
    ├── model_metrics.txt       → All model evaluation metrics
    ├── marketing_insights.md   → Actionable marketing recommendations
    ├── advertising_impact.png  → Channel vs Sales scatter plots
    ├── correlation_heatmap.png → Feature correlation matrix
    ├── prediction_accuracy.png → Actual vs Predicted scatter
    ├── residual_plot.png       → Residual analysis
    ├── model_comparison.png    → R2 comparison bar chart
    ├── roi_analysis.png        → ROI by channel
    └── sales_forecast.png      → Sales forecast curve
```

## How to Run
```bash
# From the repository root:
pip install -r requirements.txt
python Task4_SalesPrediction/src/main.py
```

## Results

### Model Comparison (Test Set — 40 samples)

| Model | R² Score | MAE ($1000s) | RMSE ($1000s) |
|-------|----------|--------------|---------------|
| **Random Forest** | **0.9820** | **0.6029** | **0.7528** |
| Gradient Boosting | 0.9817 | 0.6124 | 0.7605 |
| Lasso Regression | 0.8995 | 1.4606 | 1.7811 |
| Ridge Regression | 0.8994 | 1.4616 | 1.7821 |
| Linear Regression | 0.8994 | 1.4616 | 1.7821 |

### Cross-Validation (5-Fold)

| Model | CV R² | CV MAE |
|-------|-------|--------|
| Gradient Boosting | 0.9724 | 0.5780 |
| Random Forest | 0.9691 | 0.6047 |
| Lasso Regression | 0.8588 | 1.2488 |
| Ridge/Linear | 0.8585 | 1.2491 |

### Advertising Channel Impact

| Channel | Correlation | Sales Lift | ROI Rank |
|---------|------------|------------|----------|
| **TV** | **0.782** | **+7.0k** | #2 |
| **Radio** | 0.577 | +5.2k | **#1 (BEST ROI)** |
| Newspaper | 0.229 | +2.3k | #3 (Weakest) |

### Linear Regression Coefficients (Interpretability)

| Channel | Coefficient | Meaning |
|---------|------------|---------|
| Radio | +0.1892 | $1k more on Radio → +189 units sold |
| TV | +0.0447 | $1k more on TV → +45 units sold |
| Newspaper | +0.0029 | $1k more on Newspaper → +3 units sold |

## Key Findings & Marketing Insights

1. **Radio delivers the highest ROI** — Every $1,000 invested returns ~189 additional sales units. Despite lower total correlation, Radio is the most cost-effective channel per dollar spent.

2. **TV drives the most total sales** — Strongest correlation (0.782) and highest absolute sales lift (+7k units). Best for large-scale campaigns.

3. **Newspaper is nearly useless** — Negligible impact on sales (correlation: 0.229, coefficient: 0.003). Budget should be redirected to TV or Radio.

4. **Non-linear models vastly outperform linear ones** — Random Forest (R²=0.98) captures interaction effects between channels that Linear Regression (R²=0.90) misses.

5. **Recommended budget strategy:**
   - Maximize TV spend for mass reach
   - Allocate significant budget to Radio for cost-efficient supplementary reach
   - Minimize or eliminate Newspaper advertising
   - Consider TV + Radio combined campaigns for synergy effects

## Future Improvements
- Add interaction terms (TV × Radio) to linear models
- Test polynomial regression for capturing non-linear relationships
- Time-series forecasting if temporal data becomes available
- A/B testing framework for budget optimization
- Include digital advertising channels (Social Media, Search Ads)
