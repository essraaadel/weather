import streamlit as st
import joblib
import numpy as np
import pickle
# Load model and label encoders
model = pickle.load("random_forest_model.pkl")
rain_today_encoder = joblib.load("RainToday_LabelEncoder.joblib")
location_encoder = joblib.load("Location_LabelEncoder.joblib")
wind_dir_encoder = joblib.load("WindDir3pm_LabelEncoder.joblib")
wind_gust_encoder = joblib.load("WindGustDir_LabelEncoder.joblib")

st.title("🌦️ Weather Prediction App")

# Layout: two columns
col1, col2 = st.columns(2)

with col1:
    MinTemp = st.number_input("Min Temperature (°C)")
    MaxTemp = st.number_input("Max Temperature (°C)")
    Rainfall = st.number_input("Rainfall (mm)")
    RainToday = st.selectbox("Rain Today", rain_today_encoder.classes_)
    Location = st.selectbox("Location", location_encoder.classes_)
    WindGustDir = st.selectbox("Wind Gust Direction", wind_gust_encoder.classes_)

with col2:
    Humidity3pm = st.number_input("Humidity at 3pm (%)")
    Pressure3pm = st.number_input("Pressure at 3pm (hPa)")
    WindSpeed3pm = st.number_input("Wind Speed at 3pm (km/h)")
    Temp3pm = st.number_input("Temperature at 3pm (°C)")
    WindDir3pm = st.selectbox("Wind Direction at 3pm", wind_dir_encoder.classes_)

# Humidity validation
if not (0 <= Humidity3pm <= 100):
    st.warning("⚠️ Humidity should be between 0 and 100.")

if st.button("Predict"):
    try:
        # Encode categorical variables
        rain_today_encoded = rain_today_encoder.transform([RainToday])[0]
        location_encoded = location_encoder.transform([Location])[0]
        wind_dir_encoded = wind_dir_encoder.transform([WindDir3pm])[0]
        wind_gust_encoded = wind_gust_encoder.transform([WindGustDir])[0]

        # Prepare input in expected order
        input_data = np.array([[MinTemp, MaxTemp, Rainfall, 
                                Humidity3pm, Pressure3pm, WindSpeed3pm, Temp3pm,
                                rain_today_encoded, location_encoded, wind_dir_encoded, wind_gust_encoded]])

        # Prediction
        prediction = model.predict(input_data)

        # Probability (if supported)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0][1]
            st.write(f"Probability of rain tomorrow: **{prob:.2%}**")

        # Result
        st.success("🌧️ It will rain tomorrow." if prediction[0] == 1 else "☀️ It will not rain tomorrow.")
    
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
