from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "california_housing_processed.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the processed dataset and separate features and target."""
    print("Loading processed dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Processed dataset not found. "
            "Run python -m src.data_preprocessing first."
        )

    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    return X, y


def train_models(X_train, y_train):
    """Train four regression models."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=15,
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        trained_models[name] = model

        print(f"{name} training completed!")

    return trained_models


def evaluate_models(trained_models, X_test, y_test):
    """Evaluate trained models using regression metrics."""
    results = []

    for name, model in trained_models.items():
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2,
        })

        print(f"\n{name}")
        print(f"MAE:      {mae:.4f}")
        print(f"MSE:      {mse:.4f}")
        print(f"RMSE:     {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="RMSE",
        ascending=True
    )

    results_file = METRICS_DIR / "model_comparison.csv"
    results_df.to_csv(results_file, index=False)

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))
    print("\nMetrics saved to:", results_file)

    return results_df


def save_models(trained_models):
    """Save all trained models."""
    for name, model in trained_models.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        model_path = MODELS_DIR / filename

        joblib.dump(model, model_path)
        print("Saved model:", model_path)


def main():
    """Run the model training and evaluation workflow."""
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    trained_models = train_models(X_train, y_train)

    evaluate_models(
        trained_models,
        X_test,
        y_test
    )

    save_models(trained_models)

    print("\nModel training and evaluation completed successfully!")


if __name__ == "__main__":
    main()