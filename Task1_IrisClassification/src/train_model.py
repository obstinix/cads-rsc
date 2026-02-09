"""
Train Model Module — Iris Flower Classification
=================================================
Trains multiple classification models and compares their performance.

Models trained:
    1. Logistic Regression — Simple linear classifier
    2. K-Nearest Neighbors (KNN) — Distance-based
    3. Support Vector Machine (SVM) — Finds optimal hyperplane
    4. Decision Tree — Rule-based, interpretable
    5. Random Forest — Ensemble of decision trees

Dataset: https://www.kaggle.com/datasets/saurabh00007/iriscsv
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def get_models() -> dict:
    """
    Create a dictionary of classification models to train.

    Returns
    -------
    dict
        Model name -> sklearn estimator instance.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=200,           # Ensure convergence
            random_state=42,
            multi_class="multinomial",  # Multi-class classification
            solver="lbfgs"
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,          # Classic choice for small datasets
            weights="uniform",
            metric="euclidean"
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",           # Radial basis function kernel
            C=1.0,                  # Regularization parameter
            gamma="scale",
            random_state=42,
            probability=True        # Enable probability estimates
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,            # Prevent overfitting
            random_state=42,
            criterion="gini"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,       # Number of trees
            max_depth=5,
            random_state=42,
            n_jobs=-1               # Use all CPU cores
        ),
    }
    return models


def train_single_model(model, X_train: np.ndarray, y_train: np.ndarray, name: str):
    """
    Train a single model and return it.

    Parameters
    ----------
    model : sklearn estimator
        The model to train.
    X_train : np.ndarray
        Training features (scaled).
    y_train : np.ndarray
        Training labels.
    name : str
        Model name for logging.

    Returns
    -------
    sklearn estimator
        The fitted model.
    """
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    print(f"  [OK] {name}: Training accuracy = {train_accuracy:.4f}")
    return model


def cross_validate_model(model, X_train: np.ndarray, y_train: np.ndarray,
                         cv: int = 5) -> dict:
    """
    Perform k-fold cross-validation on a model.

    Cross-validation gives us a more reliable estimate of model performance
    by training and testing on different subsets of the data.

    Parameters
    ----------
    model : sklearn estimator
        The model to evaluate.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    cv : int
        Number of folds (default: 5).

    Returns
    -------
    dict
        {mean_score, std_score, all_scores}
    """
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    return {
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "all_scores": scores,
    }


def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Train all models and perform cross-validation.

    Parameters
    ----------
    X_train : np.ndarray
        Training features (scaled).
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    tuple
        (trained_models, cv_results)
        - trained_models: dict of name -> fitted model
        - cv_results: dict of name -> cross-validation results
    """
    models = get_models()
    trained_models = {}
    cv_results = {}

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    for name, model in models.items():
        print(f"\nTraining: {name}")

        # Train the model
        trained_model = train_single_model(model, X_train, y_train, name)
        trained_models[name] = trained_model

        # Cross-validate
        cv = cross_validate_model(model, X_train, y_train, cv=5)
        cv_results[name] = cv
        print(f"  [CV]  5-fold CV accuracy: {cv['mean_score']:.4f} (+/- {cv['std_score']:.4f})")

    # Summary
    print("\n" + "-" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("-" * 60)
    for name, cv in sorted(cv_results.items(), key=lambda x: x[1]["mean_score"], reverse=True):
        print(f"  {name:30s} -> {cv['mean_score']:.4f} (+/- {cv['std_score']:.4f})")

    # Identify best model
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_score"])
    print(f"\n  >>> Best model: {best_name} ({cv_results[best_name]['mean_score']:.4f})")

    return trained_models, cv_results
