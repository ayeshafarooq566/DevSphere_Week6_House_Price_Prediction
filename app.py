from pathlib import Path

import joblib
import streamlit as st
import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_FILE = PROJECT_ROOT / "outputs" / "metrics" / "final_model_evaluation.csv"


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏡",
    layout="wide",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fb;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #4338ca;
        text-align: center;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        background-color: #e0e7ff;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD THE BEST TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    """Load the best model according to the saved RMSE results."""

    if not METRICS_FILE.exists():
        return None, None

    results = pd.read_csv(METRICS_FILE)

    if results.empty:
        return None, None

    best_model_name = results.sort_values("RMSE").iloc[0]["Model"]

    model_filename = best_model_name.lower().replace(" ", "_") + ".joblib"
    model_path = MODELS_DIR / model_filename

    if not model_path.exists():
        return None, best_model_name

    model = joblib.load(model_path)

    return model, best_model_name


model, best_model_name = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🏡 California House Price Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning-powered house value estimation'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("About This Project")

    st.write(
        """
        This application uses machine learning models trained
        on the California Housing dataset.

        **Project features**
        - Exploratory data analysis
        - Four regression models
        - Model evaluation
        - Interactive predictions
        """
    )

    st.caption(
        "Educational project. Predictions are estimates, "
        "not professional property valuations."
    )


# --------------------------------------------------
# MODEL STATUS
# --------------------------------------------------

if model is None:
    st.error(
        "The trained model could not be loaded. "
        "Please run `python -m src.train` and "
        "`python -m src.evaluate` from the project folder."
    )
    st.stop()

st.success(f"Ready to predict! Selected model: **{best_model_name}**")


# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

st.subheader("Enter Housing Information")

st.info(
    "Enter the following district-level statistics. "
    "The model predicts the median house value for a district, "
    "not the exact price of an individual house."
)

with st.form("house_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.number_input(
            "Median Income",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1,
            help="Median income in units of tens of thousands of dollars.",
        )

        house_age = st.number_input(
            "Median House Age (years)",
            min_value=1.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
        )

        ave_rooms = st.number_input(
            "Average Rooms",
            min_value=0.5,
            max_value=100.0,
            value=5.0,
            step=0.1,
        )

        ave_bedrms = st.number_input(
            "Average Bedrooms",
            min_value=0.1,
            max_value=20.0,
            value=1.0,
            step=0.1,
        )

    with col2:
        population = st.number_input(
            "District Population",
            min_value=1.0,
            max_value=50000.0,
            value=1000.0,
            step=100.0,
        )

        ave_occup = st.number_input(
            "Average Occupancy",
            min_value=0.1,
            max_value=1000.0,
            value=3.0,
            step=0.1,
        )

        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.5,
            value=34.05,
            step=0.01,
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=-118.25,
            step=0.01,
        )

    submitted = st.form_submit_button(
        "Predict House Value",
        use_container_width=True,
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    input_data = pd.DataFrame(
        [
            {
                "MedInc": med_inc,
                "HouseAge": house_age,
                "AveRooms": ave_rooms,
                "AveBedrms": ave_bedrms,
                "Population": population,
                "AveOccup": ave_occup,
                "Latitude": latitude,
                "Longitude": longitude,
            }
        ]
    )

    prediction = float(model.predict(input_data)[0])

    # The dataset target is expressed in units of $100,000.
    predicted_dollars = prediction * 100000

    st.markdown(
        '<div class="result-box">'
        '<h3>Predicted Median House Value</h3>'
        f'<h1>${predicted_dollars:,.0f}</h1>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"Raw model output: {prediction:.4f} units of $100,000. "
        "The displayed amount is an estimate based on the dataset."
    )

    st.subheader("Your Submitted Inputs")
    st.dataframe(input_data, use_container_width=True)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()
st.subheader("Model Performance")

if METRICS_FILE.exists():

    metrics = pd.read_csv(METRICS_FILE)
    metrics = metrics.sort_values("RMSE")

    st.dataframe(
        metrics.style.format(
            {
                "MAE": "{:.4f}",
                "MSE": "{:.4f}",
                "RMSE": "{:.4f}",
                "R2 Score": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

else:
    st.warning("Model evaluation metrics have not been generated yet.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "California House Price Prediction | "
    "DevSphere Week 6 Machine Learning Project"
)