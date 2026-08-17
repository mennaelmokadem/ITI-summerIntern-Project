"""
Streamlit app: predicts the probability that a data scientist candidate
will look for a new job, given a trained model.pkl (produced by train_model.py).

Run:
    pip install streamlit joblib pandas numpy scikit-learn xgboost lightgbm catboost
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

from preprocessing import engineer_features

st.set_page_config(page_title="Job Change Predictor", layout="centered")


@st.cache_resource
def load_model():
    pipe = joblib.load("model.pkl")
    meta = joblib.load("model_meta.pkl")
    return pipe, meta


try:
    pipe, meta = load_model()
except FileNotFoundError:
    st.error(
        "model.pkl / model_meta.pkl not found. Run `python train_model.py` "
        "first (with aug_train.csv in this folder), then relaunch the app."
    )
    st.stop()

st.title("Smart Recruitment Assistant")
st.write("Fill in a candidate's details to estimate the probability they're looking for a new job.")

known_cities = meta["known_cities"]

with st.form("candidate_form"):
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", options=known_cities, index=known_cities.index("city_103") if "city_103" in known_cities else 0)
        city_development_index = st.slider("City development index", 0.4, 1.0, 0.9, 0.001)
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        relevent_experience = st.selectbox(
            "Relevant experience", ["Has relevent experience", "No relevent experience"]
        )
        enrolled_university = st.selectbox(
            "University enrollment", ["no_enrollment", "Full time course", "Part time course", "Unknown"]
        )
        education_level = st.selectbox(
            "Education level", ["Graduate", "Masters", "High School", "Phd", "Primary School", "Unknown"]
        )
        major_discipline = st.selectbox(
            "Major discipline", ["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other", "Unknown"]
        )

    with col2:
        experience = st.slider("Years of experience", 0, 21, 5, help="21 represents '>20 years'")
        company_size = st.selectbox(
            "Company size", ["<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+", "Unknown"]
        )
        company_type = st.selectbox(
            "Company type",
            ["Pvt Ltd", "Funded Startup", "Public Sector", "Early Stage Startup", "NGO", "Other", "Unknown"],
        )
        last_new_job = st.selectbox(
            "Years since last job change", ["never", "1", "2", "3", "4", ">4"]
        )
        training_hours = st.number_input("Training hours completed", min_value=1, max_value=336, value=50)

    submitted = st.form_submit_button("Predict")

if submitted:
    # "Unknown" / "Prefer not to say" selections map back to real NaN so the
    # model's own missing-value handling (median/mode fill it learned in training) applies.
    raw_input = pd.DataFrame([{
        "city": city,
        "city_development_index": city_development_index,
        "gender": None if gender in ("Unknown", "Prefer not to say") else gender,
        "relevent_experience": relevent_experience,
        "enrolled_university": None if enrolled_university == "Unknown" else enrolled_university,
        "education_level": None if education_level == "Unknown" else education_level,
        "major_discipline": None if major_discipline == "Unknown" else major_discipline,
        "experience": experience,
        "company_size": None if company_size == "Unknown" else company_size,
        "company_type": None if company_type == "Unknown" else company_type,
        "last_new_job": last_new_job,
        "training_hours": training_hours,
    }])

    processed = engineer_features(raw_input, numeric_medians=pd.Series(meta["numeric_medians"]), fit=False)
    proba = pipe.predict_proba(processed)[0, 1]
    prediction = int(proba >= meta["threshold"])

    st.divider()
    st.metric("Probability of looking for a new job", f"{proba:.1%}")

    if prediction == 1:
        st.warning("Prediction: **Likely looking for a new job**")
    else:
        st.success("Prediction: **Likely to stay**")

    st.caption(
        f"Classified using the tuned threshold ({meta['threshold']:.3f}); "
        "the probability above is the underlying model score."
    )
