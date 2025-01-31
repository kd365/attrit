import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # Ensure the scaler is consistent
from tensorflow.keras.models import load_model  # Assuming the model is saved or reloaded

# Assuming the scaler and model are already trained and loaded:
scaler = joblib.load('scaler.pkl') # Replace with the scaler fitted on your training data
best_model_fnn = load_model("attrition_model.h5")  # Replace with your actual model path

# Feature mapping for user input prompts
feature_prompts = {
    "PercentSalaryHike": "Percent Salary Hike: ",
    "MonthlyRate": "Monthly Rate: ",
    "Department_Sales": "Is the employee in the Sales Department? (1 = Yes, 0 = No): ",
    "JobRole_Sales_Executive": "Is the employee a Sales Executive? (1 = Yes, 0 = No): ",
    "Gender_Female": "Gender (1 = Female, 0 = Male): ",
    "WorkLifeBalance": "Worklife Balance Score (1-4): ",
    "JobRole_Research_Director": "Is the employee a Research Director? (1 = Yes, 0 = No): ",
    "Department_HumanResources": "Is the employee in the Human Resources Department? (1 = Yes, 0 = No): ",
    "JobInvolvement": "Job Involvement Score (1-4): ",
    "YearsAtCompany": "Years At Company: ",
    "EducationField_Other": "Is the employee's Education Field identified as Other? (1 = Yes, 0 = No): ",
    "EnvironmentSatisfaction": "Environment Satisfaction Score (1-4): ",
    "JobRole_Healthcare_Representative": "Is the employee a Healthcare Representative? (1 = Yes, 0 = No): ",
    "EducationField_TechnicalDegree": "Is the employee's Education Field identified as Technical Degree? (1 = Yes, 0 = No): ",
    "TrainingTimesLastYear": "Number of Training Events last year: ",
    "AgeGroupEncoded": "Age Group (18-25, 26-35, 36-45, 46-55, or 55+): ",
    "JobRole_Research_Scientist": "Is the employee a Research Scientist? (1 = Yes, 0 = No): ",
    "Department_ResearchNDevelopment": "Is the employee in the Research & Development Department? (1 = Yes, 0 = No): ",
    "MonthlyIncome": "Monthly Income: ",
}

# Mapping of Age Groups to numeric values
agegroup_mapping = {'18-25': 1, '26-35': 2, '36-45': 3, '46-55': 4, '55+': 5}

# Streamlit form to take user input
def get_user_input():
    st.title("Employee Attrition Prediction")
    
    # Create a dictionary to store user inputs
    user_input_dict = {}
    
    # First, handle the Age Group input separately
    agegroup_input = st.selectbox("Age Group (18-25, 26-35, 36-45, 46-55, or 55+):", list(agegroup_mapping.keys()))
    user_input_dict["AgeGroupEncoded"] = agegroup_mapping[agegroup_input]

    # Conditional inputs for features (e.g., depending on 'Department_Sales' input)
    department_sales = st.selectbox("Is the employee in the Sales Department?", [1, 0])
    user_input_dict["Department_Sales"] = department_sales
    if department_sales == 1:
        user_input_dict["Department_HumanResources"] = 0
        user_input_dict["Department_ResearchNDevelopment"] = 0
    else:
        user_input_dict["Department_HumanResources"] = st.selectbox("Is the employee in the Human Resources Department?", [1, 0])
        user_input_dict["Department_ResearchNDevelopment"] = st.selectbox("Is the employee in the Research & Development Department?", [1, 0])

    job_role_sales_exec = st.selectbox("Is the employee a Sales Executive?", [1, 0])
    user_input_dict["JobRole_Sales_Executive"] = job_role_sales_exec
    if job_role_sales_exec == 1:
        user_input_dict["JobRole_Research_Director"] = 0
        user_input_dict["JobRole_Healthcare_Representative"] = 0
        user_input_dict["JobRole_Research_Scientist"] = 0
    else:
        user_input_dict["JobRole_Research_Director"] = st.selectbox("Is the employee a Research Director?", [1, 0])
        user_input_dict["JobRole_Healthcare_Representative"] = st.selectbox("Is the employee a Healthcare Representative?", [1, 0])
        user_input_dict["JobRole_Research_Scientist"] = st.selectbox("Is the employee a Research Scientist?", [1, 0])

    # Other general inputs
    user_input_dict["PercentSalaryHike"] = st.number_input("Percent Salary Hike: ", min_value=0, max_value=100)
    user_input_dict["MonthlyRate"] = st.number_input("Monthly Rate: ", min_value=0)
    user_input_dict["WorkLifeBalance"] = st.number_input("Worklife Balance Score (1-4): ", min_value=1, max_value=4)
    user_input_dict["JobInvolvement"] = st.number_input("Job Involvement Score (1-4): ", min_value=1, max_value=4)
    user_input_dict["YearsAtCompany"] = st.number_input("Years At Company: ", min_value=0)
    user_input_dict["EnvironmentSatisfaction"] = st.number_input("Environment Satisfaction Score (1-4): ", min_value=1, max_value=4)
    user_input_dict["TrainingTimesLastYear"] = st.number_input("Number of Training Events last year: ", min_value=0)
    user_input_dict["MonthlyIncome"] = st.number_input("Monthly Income: ", min_value=0)

    return user_input_dict

# Get user input from Streamlit form
user_input_dict = get_user_input()

# Convert user input into numpy array for prediction
column_names = list(feature_prompts.keys())  # Column names should match the ones used in training
user_input = np.array([[user_input_dict.get(col, 0) for col in column_names]])

# Feature scaling
try:
    user_input_scaled = scaler.transform(user_input)
    st.write("Scaled Input: ", user_input_scaled)
except Exception as e:
    st.error(f"Error during scaling: {str(e)}")

# Make prediction
prediction = best_model_fnn.predict(user_input_scaled)
prediction_class = prediction.argmax(axis=-1)

# Display the prediction result
if prediction_class[0] == 1:
    st.write("Attrition Prediction: Person will leave")
else:
    st.write("Attrition Prediction: Person will not leave")







