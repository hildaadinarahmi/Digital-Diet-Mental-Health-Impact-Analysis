import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load model (assumes you've trained and saved it as .pkl)
model_path = "mental_health_rf_model.pkl"  # Update this path if necessary
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    st.error("Model file not found. Please check the file path.")
    st.stop()  # Stop the app if the model is not found

# Title & description
st.set_page_config(page_title="Digital Diet & Mental Health Classifier", layout="centered")

st.title("🧠 Digital Diet & Mental Health Risk Predictor")
st.markdown("""
This app uses a machine learning model to predict the likelihood of **mental health risk** based on your **digital usage patterns** and **lifestyle habits**.  
Developed by **Hilda Adina Rahmi** — Junior Data Scientist.
""")

# Collect user input
screen_time = st.sidebar.slider("📱 Daily Screen Time (hours)", 0.0, 15.0, 5.0)
sleep_quality = st.sidebar.slider("😴 Sleep Quality (1 = Poor, 10 = Excellent)", 1, 10, 5)
social_media = st.sidebar.slider("📲 Social Media Use (hours)", 0.0, 8.0, 2.0)
depression = st.sidebar.slider("📉 Weekly Depression Score (0–10)", 0.0, 10.0, 4.0)
anxiety = st.sidebar.slider("😟 Weekly Anxiety Score (0–10)", 0.0, 10.0, 4.0)
stress = st.sidebar.slider("😰 Stress Level (0–10)", 0.0, 10.0, 5.0)
age = st.sidebar.slider("🎂 Age", 13, 65, 25)
gender = st.sidebar.radio("⚧️ Gender", ["Male", "Female"])
location = st.sidebar.radio("🏡 Living Environment", ["Urban", "Rural"])

# Prepare input
input_dict = {
    "daily_screen_time_hours": screen_time,
    "sleep_quality": sleep_quality,
    "social_media_hours": social_media,
    "weekly_depression_score": depression,
    "weekly_anxiety_score": anxiety,
    "stress_level": stress,
    "age": age,
    "gender_Female": 1 if gender == "Female" else 0,
    "gender_Male": 1 if gender == "Male" else 0,
    "location_type_Rural": 1 if location == "Rural" else 0,
    "location_type_Urban": 1 if location == "Urban" else 0
}

input_df = pd.DataFrame([input_dict])

# Predict
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)

# Display results
st.subheader("🎯 Prediction Result")
risk_label = "🟢 Low Risk" if prediction == 0 else "🔴 At-Risk"
st.markdown(f"### **Mental Health Risk Level: {risk_label}**")

st.subheader("📈 Risk Probability")
st.write(f"Low Risk: {proba[0][0]*100:.1f}%")
st.write(f"High Risk: {proba[0][1]*100:.1f}%")

with st.expander("📘 How does this work?"):
    st.markdown("""
    The model was trained using a **Random Forest Classifier** on a dataset of 1,000+ individuals.  
    It considers behavioral patterns such as:
    - Screen time
    - Social media usage
    - Sleep quality
    - Stress, anxiety, and depression levels
    """)
