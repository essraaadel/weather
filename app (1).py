
import streamlit as st
import joblib
import numpy as np

# Load model and label encoders
model = joblib.load("random_forest_model.pkl")
rain_today_encoder = joblib.load("RainToday_LabelEncoder.joblib")  # example encoder

st.title("🌦️ Weather Prediction App")

# Layout: two columns for better organization
col1, col2 = st.columns(2)

with col1:
    MinTemp = st.number_input("Min Temperature (°C)")
    MaxTemp = st.number_input("Max Temperature (°C)")
    Rainfall = st.number_input("Rainfall (mm)")
    RainToday = st.selectbox("Rain Today", rain_today_encoder.classes_)

with col2:
    Humidity3pm = st.number_input("Humidity at 3pm (%)")
    Pressure3pm = st.number_input("Pressure at 3pm (hPa)")
    WindSpeed3pm = st.number_input("Wind Speed at 3pm (km/h)")
    Temp3pm = st.number_input("Temperature at 3pm (°C)")

# Validation check
if not (0 <= Humidity3pm <= 100):
    st.warning("⚠️ Humidity should be between 0 and 100.")

if st.button("Predict"):
    # Encode RainToday
    rain_today_encoded = rain_today_encoder.transform([RainToday])[0]
    
    # Prepare input
    input_data = np.array([[MinTemp, MaxTemp, Rainfall, Humidity3pm, Pressure3pm, WindSpeed3pm, Temp3pm, rain_today_encoded]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Optional: Show probability if supported
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1]
        st.write(f"Probability of rain tomorrow: **{prob:.2%}**")
    
    # Output result
    st.success("🌧️ It will rain tomorrow." if prediction[0] == 1 else "☀️ It will not rain tomorrow.")
