from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the California Housing dataset."""
    print("Loading California Housing dataset...")

    dataset = fetch_california_housing(as_frame=True)

    X = dataset.data.copy()
    y = dataset.target.copy()
    y.name = "MedHouseVal"

    print("\nDataset loaded successfully!")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    return X, y


def inspect_data(X, y):
    """Inspect the dataset and check data quality."""
    print("\n========== DATA INSPECTION ==========")

    print("\nDataset information:")
    X.info()

    print("\nMissing values:")
    print(X.isnull().sum())

    print("\nDuplicate feature rows:", X.duplicated().sum())

    print("\nTarget summary:")
    print(y.describe())

    if X.isnull().values.any():
        raise ValueError("Missing values found in features.")

    if y.isnull().any():
        raise ValueError("Missing values found in target.")

    if not X.index.equals(y.index):
        raise ValueError("Feature and target indices do not match.")

    print("\nData quality checks passed!")


def save_processed_data(X, y):
    """Save the processed dataset as a CSV file."""
    combined = X.copy()
    combined["MedHouseVal"] = y

    output_file = PROCESSED_DIR / "california_housing_processed.csv"
    combined.to_csv(output_file, index=False)

    print("\nProcessed dataset saved to:")
    print(output_file)


def split_data(X, y):
    """Split the dataset into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\n========== TRAIN-TEST SPLIT ==========")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training percentage: 80%")
    print("Testing percentage: 20%")

    return X_train, X_test, y_train, y_test


def main():
    """Run the complete preprocessing workflow."""
    X, y = load_data()

    inspect_data(X, y)
    save_processed_data(X, y)
    split_data(X, y)

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()