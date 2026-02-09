# 📊 CADS-RSC — Data Science Projects

> A collection of data science projects covering classification, regression, data analysis, and visualization.

---

## 🏗️ Repository Structure

```
cads-rsc/
├── README.md                        ← You are here
├── requirements.txt                 ← All Python dependencies
├── .gitignore
│
├── Task1_IrisClassification/        ← Iris Flower Classification
│   ├── data/       → Dataset files
│   ├── notebooks/  → Jupyter notebooks
│   ├── src/        → Python source code
│   ├── models/     → Trained model files
│   └── results/    → Visualizations & metrics
│
├── Task2_UnemploymentAnalysis/      ← Unemployment in India Analysis
│   ├── data/       → Dataset files
│   ├── notebooks/  → Jupyter notebooks
│   ├── src/        → Python source code
│   └── results/    → Visualizations & insights
│
├── Task3_CarPricePrediction/        ← Car Price Prediction (ML)
│   ├── data/       → Dataset files
│   ├── notebooks/  → Jupyter notebooks
│   ├── src/        → Python source code
│   ├── models/     → Trained model files
│   └── results/    → Visualizations & metrics
│
└── Task4_SalesPrediction/           ← Sales Prediction from Advertising
    ├── data/       → Dataset files
    ├── notebooks/  → Jupyter notebooks
    ├── src/        → Python source code
    ├── models/     → Trained model files
    └── results/    → Visualizations & metrics
```

---

## 📋 Task Overview

| # | Task | Description | Status |
|---|------|-------------|--------|
| 1 | **Iris Flower Classification** | Classify iris species using sepal/petal measurements with multiple ML models | ✅ Complete — SVM best (96.67%) |
| 2 | **Unemployment Analysis** | Analyze unemployment trends in India, including Covid-19 impact | 🔲 Not Started |
| 3 | **Car Price Prediction** | Predict used car prices using regression and feature engineering | 🔲 Not Started |
| 4 | **Sales Prediction** | Forecast sales from advertising spend across TV, Radio, and Newspaper | ✅ Complete — Random Forest best (R²=0.982) |

---

## 🛠️ Tech Stack

- **Language:** Python 3.12+
- **Data:** Pandas, NumPy
- **ML:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Environment:** Jupyter Notebook

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/cads-rsc.git
cd cads-rsc

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Navigate to any task and explore!
cd Task1_IrisClassification
python src/main.py
```

## 📄 License

This project is for educational purposes. Datasets are sourced from [Kaggle](https://www.kaggle.com/) and belong to their respective owners.
