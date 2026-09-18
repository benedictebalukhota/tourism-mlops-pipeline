"""
app.py
------------------------------------------------------------------
STEP 4 of the MLOps pipeline: MODEL DEPLOYMENT (Streamlit front-end).

Loads the registered model from the Hugging Face model hub, collects
customer attributes from the user, assembles them into a single-row
DataFrame matching the training schema, and returns a purchase
prediction for the Wellness Tourism Package.
"""

import os

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HF_USERNAME = os.getenv("HF_USERNAME", "<your-hf-username>")
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-model"
MODEL_FILENAME = "tourism_model.joblib"

st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="🌴",
    layout="centered",
)


# ----------------------------------------------------------------------
# Model loading (cached so the hub download happens only once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Download and deserialize the registered model from the HF model hub."""
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        token=os.getenv("HF_TOKEN"),  # only needed if the repo is private
    )
    return joblib.load(model_path)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🌴 Wellness Tourism Package Predictor")
st.markdown(
    "Predicts whether a customer is likely to purchase the **Wellness Tourism "
    "Package**, so the sales team can prioritise outreach before making contact."
)

try:
    model = load_model()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the model from the Hugging Face hub: {exc}")
    st.stop()

st.subheader("Customer details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    type_of_contact = st.selectbox(
        "Type of Contact", ["Self Enquiry", "Company Invited"]
    )
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox(
        "Marital Status", ["Married", "Divorced", "Single", "Unmarried"]
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=1000, max_value=100000, value=23000, step=500
    )

with col2:
    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=10, value=3, step=1
    )
    number_of_children_visiting = st.number_input(
        "Number of Children Visiting (under 5)",
        min_value=0,
        max_value=5,
        value=1,
        step=1,
    )
    preferred_property_star = st.selectbox(
        "Preferred Property Star", [3.0, 4.0, 5.0], index=0
    )
    number_of_trips = st.number_input(
        "Number of Trips per Year", min_value=1, max_value=25, value=3, step=1
    )
    passport = st.selectbox("Holds a Passport", ["No", "Yes"])
    own_car = st.selectbox("Owns a Car", ["No", "Yes"])
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )
    duration_of_pitch = st.number_input(
        "Duration of Pitch (minutes)", min_value=1, max_value=130, value=15, step=1
    )

number_of_followups = st.slider("Number of Follow-ups", 1, 6, 4)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)

# ----------------------------------------------------------------------
# Assemble inputs into a single-row DataFrame
# ----------------------------------------------------------------------
# Column names and order must match the training schema exactly, otherwise
# the fitted ColumnTransformer will not recognise them.
input_df = pd.DataFrame(
    [
        {
            "Age": float(age),
            "TypeofContact": type_of_contact,
            "CityTier": int(city_tier),
            "DurationOfPitch": float(duration_of_pitch),
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": int(number_of_person_visiting),
            "NumberOfFollowups": float(number_of_followups),
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": float(preferred_property_star),
            "MaritalStatus": marital_status,
            "NumberOfTrips": float(number_of_trips),
            "Passport": 1 if passport == "Yes" else 0,
            "PitchSatisfactionScore": int(pitch_satisfaction_score),
            "OwnCar": 1 if own_car == "Yes" else 0,
            "NumberOfChildrenVisiting": float(number_of_children_visiting),
            "Designation": designation,
            "MonthlyIncome": float(monthly_income),
        }
    ]
)

with st.expander("View model input"):
    st.dataframe(input_df.T.rename(columns={0: "value"}))

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if st.button("Predict", type="primary"):
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    st.markdown("---")
    if prediction == 1:
        st.success(
            f"**Likely to purchase** the Wellness Tourism Package "
            f"(probability: {probability:.1%})"
        )
        st.markdown("**Recommendation:** prioritise this customer for outreach.")
    else:
        st.info(
            f"**Unlikely to purchase** the Wellness Tourism Package "
            f"(probability: {probability:.1%})"
        )
        st.markdown(
            "**Recommendation:** deprioritise, or approach with a different package."
        )

    st.progress(probability)
    st.caption(f"Predicted purchase probability: {probability:.4f}")

st.markdown("---")
st.caption(
    f"Model source: [{MODEL_REPO_ID}](https://huggingface.co/{MODEL_REPO_ID}) | "
    "Built with an automated MLOps pipeline (GitHub Actions + Hugging Face Hub)."
)
