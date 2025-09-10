# app.py
import streamlit as st
import pandas as pd
import joblib

st.title("Heart Disease Prediction 🫀")

# -------------------------------
# Load the saved pipeline with try-except
# -------------------------------
try:
    model = joblib.load(r"model/final_model.pkl")
except FileNotFoundError:
    st.error("FileNotFoundError: Could not find 'model/final_model.pkl'. Please check the path.")
    st.stop()  # Stop further execution
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# -------------------------------
# Sidebar inputs
# -------------------------------
st.sidebar.header("Enter Patient Information")
Age = st.sidebar.number_input("Age", 20, 100, 50)
Chest_Pain_Type = st.sidebar.selectbox("Chest Pain Type", [1, 2, 3, 4])
Resting_Blood_Pressure = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
Cholesterol_Level = st.sidebar.number_input("Cholesterol Level", 100, 400, 200)
Maximum_Heart_Rate_Achieved = st.sidebar.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
Exercise_Induced_Angina = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
ST_Depression = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.0)
Slope = st.sidebar.selectbox("Slope of ST Segment", [1, 2, 3])
Number_of_Major_Arteries = st.sidebar.selectbox("Number of Major Arteries", [0, 1, 2, 3])
Thalassemia = st.sidebar.selectbox("Thalassemia", [3, 6, 7])

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
