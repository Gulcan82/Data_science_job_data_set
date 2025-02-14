import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Streamlit UI
st.title("Data Science Job Prediction")

# Collecting all 11 feature inputs
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

# Encode categorical values
input_data = np.array([[experience, training_hours, city_development_index, education_level, company_size,
                        company_type, major_discipline, gender, relevant_experience, enrolled_university]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write("Prediction:", "Looking for a Job" if prediction[0] == 1 else "Not Looking for a Job")
