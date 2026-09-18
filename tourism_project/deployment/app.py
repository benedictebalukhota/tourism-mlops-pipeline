"""
app.py
------------------------------------------------------------------
STEP 4 of the MLOps pipeline: MODEL DEPLOYMENT (Gradio front-end).

Loads the registered model from the Hugging Face model hub, collects
customer attributes from the user, assembles them into a single-row
DataFrame matching the training schema, and returns a purchase
prediction for the Wellness Tourism Package.

NOTE ON SDK CHOICE: this app is built with Gradio rather than Streamlit.
As of mid-2026, Hugging Face Spaces only allow Docker and Streamlit-SDK
Spaces to run on compute for accounts with a paid (PRO) plan; free
personal accounts can still host Gradio Spaces on the free ZeroGPU tier.
Gradio was chosen here specifically so the app can be deployed and
graded without requiring a paid subscription. A Streamlit version of
this same app is kept in `app_streamlit_legacy.py` for reference, and
`Dockerfile` documents how it would be containerised on a paid plan.
"""

import os

import gradio as gr
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HF_USERNAME = os.getenv("HF_USERNAME", "<your-hf-username>")
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-model"
MODEL_FILENAME = "tourism_model.joblib"

_model = None  # lazy-loaded, cached at module scope


def load_model():
    """Download and deserialize the registered model from the HF model hub."""
    global _model
    if _model is None:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            token=os.getenv("HF_TOKEN"),  # only needed if the repo is private
        )
        _model = joblib.load(model_path)
    return _model


# ----------------------------------------------------------------------
# Prediction function
# ----------------------------------------------------------------------
def predict(
    age,
    type_of_contact,
    city_tier,
    occupation,
    gender,
    marital_status,
    designation,
    monthly_income,
    number_of_person_visiting,
    number_of_children_visiting,
    preferred_property_star,
    number_of_trips,
    passport,
    own_car,
    product_pitched,
    duration_of_pitch,
    number_of_followups,
    pitch_satisfaction_score,
):
    try:
        model = load_model()
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Could not load the model from the Hugging Face hub: {exc}"

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

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    if prediction == 1:
        headline = f"✅ Likely to purchase the Wellness Tourism Package"
        recommendation = "Recommendation: prioritise this customer for outreach."
    else:
        headline = f"ℹ️ Unlikely to purchase the Wellness Tourism Package"
        recommendation = "Recommendation: deprioritise, or approach with a different package."

    result_md = (
        f"### {headline}\n\n"
        f"**Predicted purchase probability:** {probability:.1%}\n\n"
        f"{recommendation}"
    )
    return result_md


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
with gr.Blocks(title="Wellness Tourism Package Predictor") as demo:
    gr.Markdown(
        "# 🌴 Wellness Tourism Package Predictor\n"
        "Predicts whether a customer is likely to purchase the **Wellness "
        "Tourism Package**, so the sales team can prioritise outreach before "
        "making contact."
    )

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age", value=35, minimum=18, maximum=100)
            type_of_contact = gr.Dropdown(
                ["Self Enquiry", "Company Invited"], label="Type of Contact",
                value="Self Enquiry",
            )
            city_tier = gr.Dropdown([1, 2, 3], label="City Tier", value=1)
            occupation = gr.Dropdown(
                ["Salaried", "Small Business", "Large Business", "Free Lancer"],
                label="Occupation", value="Salaried",
            )
            gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
            marital_status = gr.Dropdown(
                ["Married", "Divorced", "Single", "Unmarried"],
                label="Marital Status", value="Married",
            )
            designation = gr.Dropdown(
                ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
                label="Designation", value="Manager",
            )
            monthly_income = gr.Number(
                label="Monthly Income", value=23000, minimum=1000, maximum=100000
            )

        with gr.Column():
            number_of_person_visiting = gr.Number(
                label="Number of Persons Visiting", value=3, minimum=1, maximum=10
            )
            number_of_children_visiting = gr.Number(
                label="Number of Children Visiting (under 5)",
                value=1, minimum=0, maximum=5,
            )
            preferred_property_star = gr.Dropdown(
                [3.0, 4.0, 5.0], label="Preferred Property Star", value=3.0
            )
            number_of_trips = gr.Number(
                label="Number of Trips per Year", value=3, minimum=1, maximum=25
            )
            passport = gr.Dropdown(["No", "Yes"], label="Holds a Passport", value="No")
            own_car = gr.Dropdown(["No", "Yes"], label="Owns a Car", value="No")
            product_pitched = gr.Dropdown(
                ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
                label="Product Pitched", value="Basic",
            )
            duration_of_pitch = gr.Number(
                label="Duration of Pitch (minutes)", value=15, minimum=1, maximum=130
            )

    with gr.Row():
        number_of_followups = gr.Slider(1, 6, value=4, step=1, label="Number of Follow-ups")
        pitch_satisfaction_score = gr.Slider(
            1, 5, value=3, step=1, label="Pitch Satisfaction Score"
        )

    predict_btn = gr.Button("Predict", variant="primary")
    output_md = gr.Markdown()

    predict_btn.click(
        fn=predict,
        inputs=[
            age, type_of_contact, city_tier, occupation, gender, marital_status,
            designation, monthly_income, number_of_person_visiting,
            number_of_children_visiting, preferred_property_star, number_of_trips,
            passport, own_car, product_pitched, duration_of_pitch,
            number_of_followups, pitch_satisfaction_score,
        ],
        outputs=output_md,
    )

    gr.Markdown(
        f"---\nModel source: [{MODEL_REPO_ID}](https://huggingface.co/{MODEL_REPO_ID}) | "
        "Built with an automated MLOps pipeline (GitHub Actions + Hugging Face Hub)."
    )

if __name__ == "__main__":
    demo.launch()
