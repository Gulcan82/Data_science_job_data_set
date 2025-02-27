import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model, preprocessing objects, and column names
model = joblib.load("xgboost_model.pkl")
encoder = joblib.load("onehot_encoder.pkl")
scaler = joblib.load("scaler.pkl")
training_columns = joblib.load("training_columns.pkl")

# Streamlit UI
st.title("Data Science Job Prediction")

# Collecting all feature inputs
experience = st.slider("Years of Experience", 0, 20, 5)
training_hours = st.slider("Training Hours", 0, 200, 50)
city_development_index = st.slider("City Development Index", 0.5, 1.0, 0.75)
education_level = st.selectbox("Education Level", ["Unknown", "Primary School", "High School", "Graduate", "Masters", "PhD"])
company_size = st.selectbox("Company Size", ["Unknown", "<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"])
company_type = st.selectbox("Company Type", ["Unknown", "Startup", "Product", "Service", "Consulting", "Government", "NGO"])
major_discipline = st.selectbox("Major Discipline", ["Unknown", "STEM", "Business", "Arts", "Humanities", "Other"])
gender = st.selectbox("Gender", ["Unknown", "Male", "Female", "Other"])
relevant_experience = st.selectbox("Relevant Experience", ["Has relevant experience", "No relevant experience"])
enrolled_university = st.selectbox("Enrolled University", ["Unknown", "Full-time", "Part-time", "No Enrollment"])

def align_features(input_data, training_columns):
    """Aligns input data with the expected feature set, filling missing columns with 0."""
    aligned_data = pd.DataFrame(0, index=[0], columns=training_columns)  # Initialize with all training columns set to 0
    
    # Update values for columns present in input_data
    for col in input_data.columns:
        if col in aligned_data.columns:
            aligned_data[col] = input_data[col].iloc[0]  # Assign the value from the first row
    return aligned_data

# Prediction
if st.button("Predict"):
    # Create input data
    input_data = pd.DataFrame({
        'city_development_index': [city_development_index],
        'education_level': [education_level],
        'company_size': [company_size],
        'company_type': [company_type],
        'major_discipline': [major_discipline],
        'gender': [gender],
        'relevant_experience': [relevant_experience],
        'enrolled_university': [enrolled_university],
        'experience': [experience],
        'training_hours': [training_hours]
    })

    # Encode categorical features using the saved encoder
    categorical_cols = ['education_level', 'company_size', 'company_type', 'major_discipline', 'gender', 'relevant_experience', 'enrolled_university']
    encoded_features = encoder.transform(input_data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

    # Combine numerical and encoded categorical features
    numerical_cols = ['city_development_index', 'experience', 'training_hours']
    input_processed = pd.concat([input_data[numerical_cols].reset_index(drop=True), encoded_df], axis=1)

    # Align features with training data
    aligned_input = align_features(input_processed, training_columns)

    # Scale numerical features
    aligned_input[numerical_cols] = scaler.transform(aligned_input[numerical_cols])

    # Make prediction
    prediction = model.predict(aligned_input)
    st.write("Prediction:", "Looking for a Job" if prediction[0] == 1 else "Not Looking for a Job")