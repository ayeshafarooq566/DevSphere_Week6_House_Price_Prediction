# California House Price Prediction

A machine learning project that analyzes California housing data, compares regression models, and predicts median house values through an interactive Streamlit web application.

## Project Overview

This project was developed as part of the DevSphere Week 6 Machine Learning Internship task.

It demonstrates the complete machine learning workflow, including data preprocessing, exploratory data analysis, model training, model evaluation, and deployment through a user-friendly web application.

## Objectives

* Explore and understand California housing data.
* Perform data preprocessing and prepare the dataset for machine learning.
* Visualize housing prices and relationships between features.
* Train and compare multiple regression algorithms.
* Evaluate model performance using regression metrics.
* Build an interactive application for house price prediction.

## Dataset

The project uses the California Housing dataset available through `scikit-learn`.

The dataset contains **20,640 observations and 8 input features**.

### Input Features

| Feature    | Description                   |
| ---------- | ----------------------------- |
| MedInc     | Median income in the district |
| HouseAge   | Median house age              |
| AveRooms   | Average number of rooms       |
| AveBedrms  | Average number of bedrooms    |
| Population | District population           |
| AveOccup   | Average household occupancy   |
| Latitude   | Geographic latitude           |
| Longitude  | Geographic longitude          |

The target variable is `MedHouseVal`, representing median house value in units of **$100,000**.

For example, a target value of 2.5 corresponds to approximately $250,000.

## Machine Learning Models

Four regression algorithms are trained and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

The application selects the model with the lowest RMSE in the generated evaluation results.

## Evaluation Metrics

The models are evaluated using:

* **MAE (Mean Absolute Error):** Average absolute prediction error.
* **MSE (Mean Squared Error):** Average squared prediction error.
* **RMSE (Root Mean Squared Error):** Square root of the average squared error.
* **R² Score:** Measures how well the model explains variation in the target values.

The generated metrics are saved in the `outputs/metrics/` directory.

## Exploratory Data Analysis

The project generates visualizations including:

* House value distribution
* Feature correlation heatmap
* Median income versus house value
* Geographic distribution of house values
* Model RMSE comparison
* Model R² comparison
* Actual versus predicted values

## Project Structure

```text
DevSphere_Week6_House_Price_Prediction/
│
├── data/
│   └── processed/
│       └── california_housing_processed.csv
│
├── models/
│   ├── linear_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   └── gradient_boosting.joblib
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
│
├── screenshots/
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── train.py
│   └── evaluate.py
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ayeshafarooq566/DevSphere_Week6_House_Price_Prediction.git
```

### 2. Open the project directory

```bash
cd DevSphere_Week6_House_Price_Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows Command Prompt:**

```bat
.venv\Scripts\activate.bat
```

**PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt or continue using your already activated environment.

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Run these commands from the project root.

### Generate the processed dataset

```bash
python -m src.data_preprocessing
```

### Generate exploratory visualizations

```bash
python -m src.eda
```

### Train the regression models

```bash
python -m src.train
```

### Evaluate the models

```bash
python -m src.evaluate
```

### Launch the Streamlit application

```bash
python -m streamlit run app.py
```

The application opens in your browser. Enter housing-related feature values to generate a prediction.

**Note:** Generate the dataset, train the models, and run evaluation before launching the application if the generated files are not already available.

## Screenshots

Screenshots of the application, predictions, model performance, and visualizations are available in the `screenshots/` directory.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit
* Git and GitHub

## Important Note

This application produces estimates from district-level housing statistics. Its predictions are not guaranteed property valuations or current real-estate market prices.

Model performance depends on the dataset, selected features, and evaluation results.

## Author

**Ayesha Farooq**

AI/ML Engineer & Junior Python Developer

GitHub: https://github.com/ayeshafarooq566

## License

This project is available for educational and learning purposes.
