# 🌸 Task 1: Iris Flower Classification

> Classify iris flower species (Setosa, Versicolor, Virginica) based on sepal and petal measurements using multiple machine learning models.

## 📝 Status: ✅ Complete

## 📊 Dataset
- **Source:** [Kaggle — Iris CSV](https://www.kaggle.com/datasets/saurabh00007/iriscsv)
- **Features:** SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
- **Target:** Species (Iris-setosa, Iris-versicolor, Iris-virginica)
- **Samples:** 150 (50 per species, perfectly balanced)

## 🛠️ Tools & Libraries
- Python 3.12, Pandas, NumPy
- Scikit-learn (Logistic Regression, KNN, SVM, Decision Tree, Random Forest)
- Matplotlib, Seaborn
- Joblib (model persistence)

## 📁 Structure
```
Task1_IrisClassification/
├── data/
│   ├── prepare_data.py      → Script to generate iris.csv
│   └── iris.csv              → The dataset (150 samples)
├── src/
│   ├── main.py               → Main pipeline (run this!)
│   ├── load_data.py           → Data loading & inspection
│   ├── preprocess.py          → Cleaning, encoding, scaling, splitting
│   ├── train_model.py         → Model training & cross-validation
│   ├── evaluate.py            → Evaluation metrics & reporting
│   └── visualize.py           → All visualizations
├── models/
│   └── best_model.pkl         → Saved SVM model (best performer)
└── results/
    ├── classification_report.txt
    ├── feature_distributions.png
    ├── pairplot.png
    ├── correlation_heatmap.png
    ├── confusion_matrices.png
    ├── model_comparison.png
    └── feature_importance.png
```

## 🚀 How to Run
```bash
# From the repository root:
pip install -r requirements.txt
python Task1_IrisClassification/src/main.py
```

## 📈 Results

### Model Comparison (Test Set — 30 samples)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Support Vector Machine** | **96.67%** | **96.97%** | **96.67%** | **96.66%** |
| Logistic Regression | 93.33% | 93.33% | 93.33% | 93.33% |
| K-Nearest Neighbors | 93.33% | 94.44% | 93.33% | 93.27% |
| Decision Tree | 93.33% | 93.33% | 93.33% | 93.33% |
| Random Forest | 93.33% | 93.33% | 93.33% | 93.33% |

### Cross-Validation (5-Fold, Training Set)

| Model | CV Accuracy | Std Dev |
|-------|-------------|---------|
| K-Nearest Neighbors | 96.67% | ±3.12% |
| Support Vector Machine | 96.67% | ±3.12% |
| Logistic Regression | 95.83% | ±2.64% |
| Random Forest | 95.00% | ±1.67% |
| Decision Tree | 94.17% | ±2.04% |

### Feature Importance (Random Forest)
Petal measurements are far more important than sepal measurements for species classification:
- **Petal Width** and **Petal Length** dominate importance
- Sepal features contribute less to classification accuracy

## 💡 Key Takeaways

1. **Iris-setosa is perfectly separable** — All models classify it with 100% accuracy. Its petal measurements are distinctly smaller.
2. **Versicolor vs Virginica is the challenge** — These species have overlapping feature ranges, causing most misclassifications.
3. **SVM performs best** on the test set (96.67%), handling the non-linear decision boundary between Versicolor and Virginica effectively.
4. **Petal features > Sepal features** — Petal length and width are the most discriminative features.
5. **All models perform well** (93%+) — The Iris dataset is relatively simple, making it an excellent starter classification problem.

## 🔮 Future Improvements
- Try hyperparameter tuning with GridSearchCV
- Experiment with ensemble methods (Voting Classifier)
- Add dimensionality reduction (PCA) for visualization
- Test with neural network approaches
