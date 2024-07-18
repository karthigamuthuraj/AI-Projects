import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('best_kidney_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title('Chronic Kidney Disease Prediction')

# Input form
st.header('Input Patient Data')

# Define the input features based on training data with descriptive labels
feature_labels = {
    'age': 'Age',
    'bp': 'Blood Pressure',
    'al': 'Albumin',
    'su': 'Sugar',
    'bgr': 'Blood Glucose Random',
    'bu': 'Blood Urea',
    'sc': 'Serum Creatinine',
    'sod': 'Sodium',
    'pot': 'Potassium',
    'hrmo': 'Hemoglobin',
    'pcv': 'Packed Cell Volume',
    'wc': 'White Blood Cell Count',
    'rc': 'Red Blood Cell Count',
    'sg_b': 'Specific Gravity B',
    'sg_c': 'Specific Gravity C',
    'sg_d': 'Specific Gravity D',
    'sg_e': 'Specific Gravity E',
    'rbc_normal': 'Red Blood Cells (Normal)',
    'pc_normal': 'Pus Cell (Normal)',
    'pcc_present': 'Pus Cell Clumps (Present)',
    'ba_present': 'Bacteria (Present)',
    'htn_yes': 'Hypertension (Yes)',
    'dm_yes': 'Diabetes Mellitus (Yes)',
    'cad_yes': 'Coronary Artery Disease (Yes)',
    'appet_yes': 'Appetite (Yes)',
    'pe_yes': 'Pedal Edema (Yes)',
    'ane_yes': 'Anemia (Yes)'
}

# Organize input fields into rows of four
col1, col2, col3, col4 = st.columns(4)

# Create input fields for the user
input_data = {}
for feature in feature_labels:
    if feature.startswith('sg_'):
        sg_options = {'sg_b': 1.005, 'sg_c': 1.010, 'sg_d': 1.015, 'sg_e': 1.020}
        with col1 if len(input_data) % 4 == 0 else col2 if len(input_data) % 4 == 1 else col3 if len(input_data) % 4 == 2 else col4:
            input_data[feature] = st.selectbox(feature_labels[feature], list(sg_options.values()), key=feature)
    elif feature in ['al', 'su']:
        with col1 if len(input_data) % 4 == 0 else col2 if len(input_data) % 4 == 1 else col3 if len(input_data) % 4 == 2 else col4:
            input_data[feature] = st.selectbox(feature_labels[feature], [0, 1, 2, 3, 4, 5], key=feature)
    elif feature in ['rbc_normal', 'pc_normal', 'pcc_present', 'ba_present', 'htn_yes', 'dm_yes', 'cad_yes', 'appet_yes', 'pe_yes', 'ane_yes']:
        with col1 if len(input_data) % 4 == 0 else col2 if len(input_data) % 4 == 1 else col3 if len(input_data) % 4 == 2 else col4:
            input_data[feature] = st.selectbox(feature_labels[feature], ['no', 'yes'], key=feature)
    elif feature in ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hrmo', 'pcv', 'wc', 'rc']:
        with col1 if len(input_data) % 4 == 0 else col2 if len(input_data) % 4 == 1 else col3 if len(input_data) % 4 == 2 else col4:
            input_data[feature] = st.number_input(feature_labels[feature], value=0, key=feature)

# Convert categorical variables to match the training data encoding
input_data_transformed = {}
for feature in feature_labels:
    if feature.startswith('sg_'):
        for key in ['sg_b', 'sg_c', 'sg_d', 'sg_e']:
            input_data_transformed[key] = 1 if sg_options[key] == input_data[feature] else 0
    elif feature in ['al', 'su']:
        input_data_transformed[feature] = input_data[feature]
    elif feature in ['rbc_normal', 'pc_normal', 'pcc_present', 'ba_present', 'htn_yes', 'dm_yes', 'cad_yes', 'appet_yes', 'pe_yes', 'ane_yes']:
        input_data_transformed[feature] = 1 if input_data[feature] == 'yes' else 0
    else:
        input_data_transformed[feature] = input_data[feature]

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data_transformed])

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.success('The patient is likely to have Chronic Kidney Disease (CKD).')
    else:
        st.success('The patient is not likely to have Chronic Kidney Disease (CKD).')

    st.subheader('Prediction Probability:')
    st.write(f"Probability of CKD: {prediction_proba[0][1]:.4f}")
    st.write(f"Probability of No CKD: {prediction_proba[0][0]:.4f}")
