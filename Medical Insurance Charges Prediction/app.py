import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model
with open('insurance_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the scaler (ensure the scaler is fitted as in preprocessing step)
scaler = StandardScaler()
scaler.fit([[19, 27.9, 0], [64, 47.6, 5]])  # Example fit, replace with your actual fitting data

# Streamlit app
st.title("Medical Insurance Charges Prediction")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", options=["Male", "Female"])
smoker = st.selectbox("Smoker", options=["Yes", "No"])

# Preprocess the input
sex_encoded = 1 if sex == "Male" else 0
smoker_encoded = 1 if smoker == "Yes" else 0

# Create a DataFrame
input_data = pd.DataFrame([[age, bmi, children, sex_encoded, smoker_encoded]], 
                          columns=['age', 'bmi', 'children', 'sex_male', 'smoker_yes'])

# Standardize the numerical features
input_data[['age', 'bmi', 'children']] = scaler.transform(input_data[['age', 'bmi', 'children']])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.write(f"Predicted Insurance Charges: ${prediction:.2f}")
