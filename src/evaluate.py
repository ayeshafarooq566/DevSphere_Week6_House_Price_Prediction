from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "california_housing_processed.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data():
    """Load the dataset and recreate the test split."""
    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    return X_test, y_test


def evaluate_saved_models(X_test, y_test):
    """Evaluate saved models and save their predictions."""
    results = []
    predictions_df = pd.DataFrame({
        "Actual_House_Value": y_test.to_numpy()
    })

    model_files = {
        "Linear Regression": "linear_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Random Forest": "random_forest.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
    }

    for model_name, filename in model_files.items():
        model_path = MODELS_DIR / filename

        if not model_path.exists():
            print(f"Skipping {model_name}: model file not found.")
            continue

        model = joblib.load(model_path)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2,
        })

        predictions_df[
            f"{model_name}_Prediction"
        ] = predictions

        print(f"\n{model_name}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")

    if not results:
        raise FileNotFoundError(
            "No saved models found. Run python -m src.train first."
        )

    results_df = pd.DataFrame(results).sort_values(
        by="RMSE",
        ascending=True,
    )

    results_df.to_csv(
        METRICS_DIR / "final_model_evaluation.csv",
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_DIR / "test_predictions.csv",
        index=False,
    )

    print("\n========== FINAL MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    print("\nSaved evaluation metrics.")
    print("Saved test predictions.")

    return results_df


def create_evaluation_charts(results_df, X_test, y_test):
    """Create model comparison and actual-versus-predicted charts."""
    # Model RMSE comparison
    plt.figure(figsize=(9, 5))

    plt.bar(
        results_df["Model"],
        results_df["RMSE"],
    )

    plt.title("Model Comparison by RMSE")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("RMSE (units of $100,000)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "05_model_rmse_comparison.png",
        dpi=300,
    )
    plt.close()

    # R² comparison
    plt.figure(figsize=(9, 5))

    plt.bar(
        results_df["Model"],
        results_df["R2 Score"],
    )

    plt.title("Model Comparison by R² Score")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("R² Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "06_model_r2_comparison.png",
        dpi=300,
    )
    plt.close()

    # Actual vs. predicted for the best model by RMSE
    best_name = results_df.iloc[0]["Model"]

    filename = best_name.lower().replace(" ", "_") + ".joblib"
    best_model = joblib.load(MODELS_DIR / filename)

    predictions = best_model.predict(X_test)

    plt.figure(figsize=(7, 7))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.4,
        s=12,
    )

    minimum = min(y_test.min(), predictions.min())
    maximum = max(y_test.max(), predictions.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    plt.title(f"Actual vs. Predicted House Values\n{best_name}")
    plt.xlabel("Actual House Value (units of $100,000)")
    plt.ylabel("Predicted House Value (units of $100,000)")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "07_actual_vs_predicted.png",
        dpi=300,
    )
    plt.close()

    print("\nEvaluation charts saved in:", FIGURES_DIR)
    print("Best model by RMSE:", best_name)


def main():
    """Run the complete evaluation workflow."""
    X_test, y_test = load_test_data()

    results_df = evaluate_saved_models(X_test, y_test)

    create_evaluation_charts(
        results_df,
        X_test,
        y_test,
    )

    print("\nModel evaluation completed successfully!")


if __name__ == "__main__":
    main()