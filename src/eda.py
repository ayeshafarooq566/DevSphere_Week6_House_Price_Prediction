from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "california_housing_processed.csv"
)

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the processed housing dataset."""
    print("Loading processed dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run python -m src.data_preprocessing first."
        )

    df = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    return df


def create_target_distribution(df):
    """Plot the distribution of house values."""
    plt.figure(figsize=(9, 5))

    sns.histplot(
        df["MedHouseVal"],
        bins=40,
        kde=True
    )

    plt.title("Distribution of California House Values")
    plt.xlabel("Median House Value (units of $100,000)")
    plt.ylabel("Number of Districts")
    plt.tight_layout()

    output_file = FIGURES_DIR / "01_house_value_distribution.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print("Saved:", output_file)


def create_correlation_heatmap(df):
    """Plot correlations between dataset features."""
    plt.figure(figsize=(11, 8))

    correlation = df.corr(numeric_only=True)

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    output_file = FIGURES_DIR / "02_correlation_heatmap.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print("Saved:", output_file)


def create_income_vs_price(df):
    """Plot median income against median house value."""
    plt.figure(figsize=(9, 6))

    sample = df.sample(
        n=min(3000, len(df)),
        random_state=42
    )

    sns.scatterplot(
        data=sample,
        x="MedInc",
        y="MedHouseVal",
        alpha=0.5,
        s=25
    )

    plt.title("Median Income vs. Median House Value")
    plt.xlabel("Median Income")
    plt.ylabel("Median House Value (units of $100,000)")
    plt.tight_layout()

    output_file = FIGURES_DIR / "03_income_vs_house_value.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print("Saved:", output_file)


def create_geographical_plot(df):
    """Visualize house values by geographical location."""
    plt.figure(figsize=(10, 7))

    sample = df.sample(
        n=min(5000, len(df)),
        random_state=42
    )

    scatter = plt.scatter(
        sample["Longitude"],
        sample["Latitude"],
        c=sample["MedHouseVal"],
        cmap="viridis",
        s=10,
        alpha=0.7
    )

    plt.colorbar(
        scatter,
        label="Median House Value (units of $100,000)"
    )

    plt.title("Geographical Distribution of House Values")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    output_file = FIGURES_DIR / "04_geographical_house_values.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print("Saved:", output_file)


def main():
    """Run the complete exploratory data analysis."""
    df = load_data()

    print("\n========== DATASET SUMMARY ==========")
    print(df.describe())

    print("\n========== CREATING VISUALIZATIONS ==========")

    create_target_distribution(df)
    create_correlation_heatmap(df)
    create_income_vs_price(df)
    create_geographical_plot(df)

    print("\nEDA completed successfully!")
    print("All figures are saved in:", FIGURES_DIR)


if __name__ == "__main__":
    main()