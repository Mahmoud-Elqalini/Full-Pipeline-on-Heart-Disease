# app.py
import streamlit as st
import pandas as pd
import joblib

st.markdown("<h1 style='text-align: center;'>Heart Disease Prediction 🫀</h1>", unsafe_allow_html=True)
st.write("")
st.write("")  
st.write("")  
st.write("")  

# -------------------------------
# Load the saved model
# -------------------------------
try:
    model = joblib.load(r"model/final_model.pkl")
except FileNotFoundError:
    st.error("❌ Could not find 'model/final_model.pkl'. Please check the path.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ An error occurred while loading the model: {e}")
    st.stop()

# -------------------------------
# Instructions for mobile users
# -------------------------------

# -------------------------------
# Centered subtitle for patient info
# -------------------------------
st.markdown("<h3 style='text-align: center;'>📋 Enter Patient Information</h3>", unsafe_allow_html=True)

# -------------------------------
# Inputs in 2 columns
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 0, 120, 0)
    Resting_Blood_Pressure = st.number_input("Resting Blood Pressure", 0, 200, 0)
    Cholesterol_Level = st.number_input("Cholesterol Level", 0, 600, 0)
    ST_Depression = st.number_input("ST Depression", 0.0, 10.0, 0.0)
    Number_of_Major_Arteries = st.selectbox("Number of Major Arteries", [0, 1, 2, 3])

with col2:
    Chest_Pain_Type = st.selectbox("Chest Pain Type", [0, 1, 2, 3, 4])
    Maximum_Heart_Rate_Achieved = st.number_input("Maximum Heart Rate Achieved", 0, 220, 0)
    Exercise_Induced_Angina = st.selectbox("Exercise Induced Angina", [0, 1])
    Slope = st.selectbox("Slope of ST Segment", [0, 1, 2, 3])
    Thalassemia = st.selectbox("Thalassemia", [0, 3, 6, 7])

# -------------------------------
# Collect input into DataFrame
# -------------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "Chest Pain Type": Chest_Pain_Type,
    "Resting Blood Pressure": Resting_Blood_Pressure,
    "Cholesterol Level": Cholesterol_Level,
    "Maximum Heart Rate Achieved": Maximum_Heart_Rate_Achieved,
    "Exercise Induced Angina": Exercise_Induced_Angina,
    "ST Depression": ST_Depression,
    "Slope": Slope,
    "Number of Major Arteries": Number_of_Major_Arteries,
    "Thalassemia": Thalassemia
}])


# -------------------------------
# Prediction with try-except
# -------------------------------
if st.button("Predict Heart Disease"):
    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        if prediction == 1:
            st.error(f"The patient is likely to have heart disease ❤️ (Probability: {prob:.2f})")
        else:
            st.success(f"The patient is unlikely to have heart disease 💚 (Probability: {prob:.2f})")
    except ValueError as ve:
        st.error(f"ValueError: {ve}. Check that the input features match the model features and order.")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")

