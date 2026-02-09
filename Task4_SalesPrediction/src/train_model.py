"""
Train Model Module -- Sales Prediction
=======================================
Trains multiple regression models to predict sales from advertising spend.

Models trained:
    1. Linear Regression       -- Simple baseline, highly interpretable
    2. Ridge Regression        -- L2 regularization to prevent overfitting
    3. Lasso Regression        -- L1 regularization, performs feature selection
    4. Random Forest Regressor -- Non-linear, captures interactions
    5. Gradient Boosting       -- Advanced ensemble, often best performer

Dataset: https://www.kaggle.com/datasets/bumba5341/advertisingcsv
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


def get_models() -> dict:
    """
    Create a dictionary of regression models to train.

    Returns
    -------
    dict
        Model name -> sklearn estimator instance.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(
            alpha=1.0,              # Regularization strength
            random_state=42
        ),
        "Lasso Regression": Lasso(
            alpha=0.1,              # Lighter regularization for small dataset
            random_state=42,
            max_iter=10000
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        ),
    }
    return models


def train_single_model(model, X_train, y_train, name: str):
    """
    Train a single regression model.

    Parameters
    ----------
    model : sklearn estimator
        The model to train.
    X_train : array-like
        Training features.
    y_train : array-like
        Training target (Sales).
    name : str
        Model name for logging.

    Returns
    -------
    sklearn estimator
        The fitted model.
    """
    model.fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    print(f"  [OK] {name}: Training R2 = {train_r2:.4f}")
    return model


def cross_validate_model(model, X_train, y_train, cv: int = 5) -> dict:
    """
    Perform k-fold cross-validation on a regression model.

    Uses both R2 and negative MAE as scoring metrics.

    Parameters
    ----------
    model : sklearn estimator
        The model to evaluate.
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    cv : int
        Number of folds (default: 5).

    Returns
    -------
    dict
        {r2_mean, r2_std, mae_mean, mae_std}
    """
    # R-squared scores
    r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

    # MAE scores (returned as negative, so we negate)
    mae_scores = -cross_val_score(model, X_train, y_train, cv=cv,
                                   scoring="neg_mean_absolute_error")

    return {
        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(),
        "mae_mean": mae_scores.mean(),
        "mae_std": mae_scores.std(),
    }


def train_all_models(X_train, y_train) -> tuple:
    """
    Train all models and perform cross-validation.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.

    Returns
    -------
    tuple
        (trained_models, cv_results)
    """
    models = get_models()
    trained_models = {}
    cv_results = {}

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    for name, model in models.items():
        print(f"\nTraining: {name}")

        # Train
        trained_model = train_single_model(model, X_train, y_train, name)
        trained_models[name] = trained_model

        # Cross-validate
        cv = cross_validate_model(model, X_train, y_train, cv=5)
        cv_results[name] = cv
        print(f"  [CV]  5-fold R2: {cv['r2_mean']:.4f} (+/- {cv['r2_std']:.4f})")
        print(f"  [CV]  5-fold MAE: {cv['mae_mean']:.4f} (+/- {cv['mae_std']:.4f})")

    # Summary
    print("\n" + "-" * 60)
    print("CROSS-VALIDATION SUMMARY (by R2)")
    print("-" * 60)
    for name, cv in sorted(cv_results.items(), key=lambda x: x[1]["r2_mean"], reverse=True):
        print(f"  {name:25s} -> R2: {cv['r2_mean']:.4f}  MAE: {cv['mae_mean']:.4f}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["r2_mean"])
    print(f"\n  >>> Best model (CV): {best_name} (R2: {cv_results[best_name]['r2_mean']:.4f})")

    return trained_models, cv_results


def get_linear_coefficients(trained_models: dict, feature_names: list) -> dict:
    """
    Extract and display coefficients from the linear regression model.

    These coefficients are directly interpretable:
      - TV coefficient of 0.045 means: for every $1k more spent on TV ads,
        sales increase by ~45 units (0.045 * 1000).

    Parameters
    ----------
    trained_models : dict
        Trained model instances.
    feature_names : list
        Feature column names.

    Returns
    -------
    dict
        Feature name -> coefficient value.
    """
    if "Linear Regression" not in trained_models:
        return {}

    lr = trained_models["Linear Regression"]
    coeffs = dict(zip(feature_names, lr.coef_))
    intercept = lr.intercept_

    print("\n" + "-" * 60)
    print("LINEAR REGRESSION COEFFICIENTS")
    print("-" * 60)
    print(f"  Intercept: {intercept:.4f}")
    for feat, coeff in sorted(coeffs.items(), key=lambda x: abs(x[1]), reverse=True):
        impact = "increases" if coeff > 0 else "decreases"
        print(f"  {feat:12s}: {coeff:+.4f}  -> $1k more on {feat} {impact} sales by {abs(coeff)*1000:.0f} units")

    return coeffs
